function [u_final, activation_times, M_matrix_check] = monodomain_heterogeneous_ex_1_6(sigma_d_ratio, dt, nvx, nvy)
% MATLAB solver for heterogeneous monodomain equation
% sigma_d_ratio: ratio of diseased to healthy conductivity (10, 1, or 0.1)

if nargin < 4
    nvx = 33; nvy = 33;  % Default grid
end
if nargin < 3
    dt = 0.1;  % Default timestep
end

% Problem parameters
sigma_h = 9.5298e-4;  % Healthy tissue conductivity
sigma_d = sigma_d_ratio * sigma_h;  % Diseased tissue conductivity
a = 18.515; fr = 0; ft = 0.2383; fd = 1;  % Reaction parameters
T = 35;  % Final time

% Setup mesh
hx = 1/(nvx-1); hy = 1/(nvy-1);
nv = nvx*nvy; nt = T/dt;
ne = (nvx-1)*(nvy-1);

% Setup initial condition
[X,Y] = meshgrid(linspace(0,1,nvx), linspace(0,1,nvy));
u0 = zeros(nvx,nvy);
u0(X >= 0.9 & Y >= 0.9) = 1;
u = u0(:);  % Vectorize

% Define diseased regions (element centers)
sigma_elements = zeros(ne, 1);
element_idx = 1;

for j = 1:(nvy-1)  % y-direction elements
    for i = 1:(nvx-1)  % x-direction elements
        % Element center coordinates
        x_center = (i-1)*hx + hx/2;
        y_center = (j-1)*hy + hy/2;
        
        % Check if element center is in diseased regions
        in_d1 = (x_center - 0.3)^2 + (y_center - 0.7)^2 < 0.1^2;
        in_d2 = (x_center - 0.7)^2 + (y_center - 0.3)^2 < 0.15^2;
        in_d3 = (x_center - 0.5)^2 + (y_center - 0.5)^2 < 0.1^2;
        
        if in_d1 || in_d2 || in_d3
            sigma_elements(element_idx) = sigma_d;
        else
            sigma_elements(element_idx) = sigma_h;
        end
        
        element_idx = element_idx + 1;
    end
end

% Report conductivity distribution
diseased_elements = sum(sigma_elements == sigma_d);
healthy_elements = sum(sigma_elements == sigma_h);
fprintf('Conductivity setup: σ_d/σ_h = %.1f\n', sigma_d_ratio);
fprintf('Diseased elements: %d, Healthy elements: %d\n', diseased_elements, healthy_elements);

% Assemble matrices
fprintf('Assembling matrices...\n');
M = assembleMass(nvx, nvy, hx, hy);
K = assembleDiffusion_modified(nvx, nvy, hx, hy, sigma_elements);
A = M + dt*K;  % System matrix

% Check if A is an M-matrix
% M-matrix: A_ii > 0 and A_ij <= 0 for i≠j
diagonal_positive = all(diag(A) > 0);
off_diagonal_nonpositive = all(A(~eye(size(A))) <= 0);
M_matrix_check = diagonal_positive && off_diagonal_nonpositive;

fprintf('M-matrix check: %s\n', string(M_matrix_check));

% Initialize activation tracking
activation_times = inf(nv,1);
min_activation_time = inf;

% Time integration loop
fprintf('Starting time integration...\n');
for n = 1:nt
    t = n*dt;
    
    % Compute reaction term f(u) = a(u-fr)(u-ft)(u-fd)
    f_u = a*(u-fr).*(u-ft).*(u-fd);
    
    % Right-hand side: M*u - dt*M*f(u)
    rhs = M*u - dt*M*f_u;
    
    % Solve linear system: (M + dt*K)*u_new = rhs
    u_new = A\rhs;
    
    % Track activation times (when u crosses ft threshold)
    newly_activated = (u <= ft) & (u_new > ft);
    activation_times(newly_activated) = t;
    
    % Track minimum activation time for reporting
    if any(newly_activated)
        min_activation_time = min(min_activation_time, t);
    end
    
    % Update solution
    u = u_new;
    
    % Progress report
    if mod(n, nt/10) == 0
        fprintf('Step %d/%d (t=%.1f), max(u)=%.3f, min(u)=%.3f\n', ...
                n, nt, t, max(u), min(u));
    end
end

% Final results
u_final = reshape(u, nvx, nvy);
fprintf('Solution completed. Final time: %.1f\n', T);
fprintf('Solution bounds: [%.6f, %.6f]\n', min(u), max(u));

% Check if solution stays in [0,1]
bounds_satisfied = (min(u) >= -1e-10 && max(u) <= 1+1e-10);
fprintf('Constraint u ∈ [0,1]: %s\n', string(bounds_satisfied));

% Report activation time
if min_activation_time < inf
    fprintf('First activation time: %.2f ms\n', min_activation_time);
else
    fprintf('No activation detected\n');
end

% Summary for table
fprintf('\n--- SUMMARY ---\n');
fprintf('σ_d/σ_h ratio: %.1f\n', sigma_d_ratio);
fprintf('Δt: %.3f\n', dt);
fprintf('Grid: %d elements\n', ne);
fprintf('First activation: %.2f ms\n', min_activation_time);
fprintf('M-matrix: %s\n', string(M_matrix_check));
fprintf('u ∈ [0,1]: %s\n', string(bounds_satisfied));
fprintf('----------------\n');

end