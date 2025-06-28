function graphic_visualization_ex_1_8()
    % Parameters
    sigma_h = 9.5298e-4;
    ft = 0.2383; % Threshold potential

    % Diseased regions and severity factors
    diseased_regions = {@(x,y)(x-0.3).^2 + (y-0.7).^2 < 0.1^2, ... % Ωd1: most severe
                        @(x,y)(x-0.7).^2 + (y-0.3).^2 < 0.15^2, ... % Ωd2: moderate
                        @(x,y)(x-0.5).^2 + (y-0.5).^2 < 0.1^2};     % Ωd3: mild
    D_factors = [0.01, 0.2, 0.5];  % Visualize severity

    % Run simulation with stored history
    [u_final, activation_times, ~, u_history] = monodomain_solver_with_history();

    % Grid setup
    [nvy, nvx, nt] = size(u_history); % Note order is nvy, nvx now
    [X,Y] = meshgrid(linspace(0,1,nvx), linspace(1,0,nvy)); % Y from 1 to 0

    % Create figure
    fig = figure('Position', [100 100 1200 800], 'Color', 'w');

    % Main simulation plot (left)
    ax1 = subplot(2,2,[1 3]);
    h_surf = surf(X, Y, u_history(:,:,1), 'EdgeColor', 'none');
    hold on;

    % Overlay diseased regions
    for i = 1:length(diseased_regions)
        mask = diseased_regions{i}(X, Y);
        Z = ones(size(X));
        Z(~mask) = NaN;  % Transparent outside region

        surf(X, Y, Z, 'FaceColor', 'red', ...
             'FaceAlpha', 0.2 + 0.1*i, 'EdgeColor', 'none', ...
             'DisplayName', sprintf('Ωd%d (×%.2f D)', i, D_factors(i)));
    end

    zlim([0 1]); caxis([0 1]);
    colormap(ax1, jet);
    colorbar('Location', 'eastoutside');
    title('Cardiac Tissue Activation');
    xlabel('X position (cm)');
    ylabel('Y position (cm)');
    zlabel('Potential (u)');
    view(2);
    axis equal tight;
    grid off;

    % Activation status plot
    ax2 = subplot(2,2,2);
    activation_map = activation_times < inf;
    h_activation = imagesc(activation_map);
    title('Activated Nodes');
    colormap(ax2, [1 1 1; 0 0.5 0]);
    colorbar('Ticks', [0 1], 'TickLabels', {'Inactive', 'Active'});
    axis equal tight;

    % Activation fraction over time
    ax3 = subplot(2,2,4);
    h_time = plot(0, 0, 'b-', 'LineWidth', 2);
    xlim([0 35]); ylim([0 1]);
    title('Activation Progress');
    xlabel('Time (ms)');
    ylabel('Fraction Activated');
    grid on;

    % Setup video writer
    videoFile = 'enhanced_monodomain_simulation.mp4';
    writerObj = VideoWriter(videoFile, 'MPEG-4');
    writerObj.FrameRate = 10;
    open(writerObj);

    % Animation loop
    time_points = [];
    frac_points = [];
    for n = 1:5:nt
        t = n*0.1;

        % Update surface plot
        set(h_surf, 'ZData', u_history(:,:,n));
        title(ax1, sprintf('Cardiac Tissue Activation (t = %.1f ms)', t));

        % Update activation map
        current_activation = activation_times <= t;
        set(h_activation, 'CData', reshape(current_activation, nvy, nvx)); % Note order

        % Update time series plot
        frac_activated = sum(current_activation(:)) / numel(current_activation);
        time_points = [time_points t];
        frac_points = [frac_points frac_activated];
        set(h_time, 'XData', time_points, 'YData', frac_points);

        % Optional annotations
        if abs(t - 1.25) < 0.05
            annotation('textbox', [0.6 0.7 0.2 0.1], 'String', 'First Activation!', ...
                       'FitBoxToText', 'on', 'BackgroundColor', 'y');
        end

        % Capture frame
        frame = getframe(fig);
        writeVideo(writerObj, frame);

        pause(0.01);
    end

    % Finalize
    close(writerObj);
    close(fig);
    disp(['Enhanced animation saved as ' videoFile]);
end

function [u_final, activation_times, M_matrix_check, u_history] = monodomain_solver_with_history()
    % Parameters from your project
    sigma_h = 9.5298e-4;
    a = 18.515; fr = 0; ft = 0.2383; fd = 1;
    T = 35; dt = 0.1;
    nvx = 33; nvy = 33;
    
    % Setup mesh - note we swap nvx and nvy in meshgrid to get correct orientation
    hx = 1/(nvx-1); hy = 1/(nvy-1);
    nv = nvx*nvy; nt = T/dt;
    [X,Y] = meshgrid(linspace(0,1,nvx), linspace(1,0,nvy)); % Y goes from 1 to 0
    
    % Initialize history storage
    u_history = zeros(nvy, nvx, nt+1); % Note order: nvy first
    
    % Setup initial condition - TOP RIGHT CORNER (x≥0.9, y≥0.9)
    u0 = zeros(nvy,nvx);
    u0(Y >= 0.9 & X >= 0.9) = 1;  % Note we use Y first for rows
    u = u0(:);
    u_history(:,:,1) = u0;
    
    % Define diseased regions (element centers) - adjust for new orientation
    elem_centers_x = linspace(hx/2, 1-hx/2, nvx-1);
    elem_centers_y = linspace(1-hy/2, hy/2, nvy-1); % Reversed y-axis
    [ElemX, ElemY] = meshgrid(elem_centers_x, elem_centers_y);
    
    % Initialize element conductivity (all healthy by default)
    sigma_elements = sigma_h * ones((nvx-1)*(nvy-1), 1);
    
    % Apply different conductivity to diseased regions (coordinates now match)
    % Ωd1 - 10Σₕ (hyper-conductive) - top-left
    d1_mask = (ElemX-0.3).^2 + (ElemY-0.7).^2 < 0.1^2;
    sigma_elements(d1_mask(:)) = 10*sigma_h;
    
    % Ωd2 - Σₕ (same as healthy) - bottom-right
    d2_mask = (ElemX-0.7).^2 + (ElemY-0.3).^2 < 0.15^2;
    sigma_elements(d2_mask(:)) = sigma_h;
    
    % Ωd3 - 0.1Σₕ (hypo-conductive) - center
    d3_mask = (ElemX-0.5).^2 + (ElemY-0.5).^2 < 0.1^2;
    sigma_elements(d3_mask(:)) = 0.1*sigma_h;
    
    % Assemble matrices (no changes needed here)
    M = assembleMass(nvx, nvy, hx, hy);
    K = assembleDiffusion_modified(nvx, nvy, hx, hy, sigma_elements);
    A = M + dt*K;
    
    % Check M-matrix property
    diagonal_positive = all(diag(A) > 0);
    off_diagonal_nonpositive = all(A(~eye(size(A))) <= 0);
    M_matrix_check = diagonal_positive && off_diagonal_nonpositive;
    
    % Initialize activation tracking
    activation_times = inf(nv,1);
    
    % Time integration loop
    for n = 1:nt
        f_u = a*(u-fr).*(u-ft).*(u-fd);
        rhs = M*u - dt*M*f_u;
        u_new = A\rhs;
        
        newly_activated = (u <= ft) & (u_new > ft);
        activation_times(newly_activated) = n*dt;
        
        u = u_new;
        u_history(:,:,n+1) = reshape(u, nvy, nvx); % Note order
    end
    
    u_final = reshape(u, nvy, nvx);
end