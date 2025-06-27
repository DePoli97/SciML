function results_table = run_all_simulations_ex_1_7()
% Run all required simulations for the project table

% Define simulation parameters
dt_values = [0.1, 0.05, 0.025];
ne_values = [64, 128, 256];  % Number of elements
sigma_ratios = [10, 1, 0.1];  % σ_d/σ_h ratios

% Grid sizes corresponding to ne_values (approximately)
% For ne = (nvx-1)*(nvy-1), we need nvx = nvy = sqrt(ne) + 1
nvx_values = [9, 12, 17];  % Gives ne ≈ 64, 121, 256

% Initialize results storage
results = [];

fprintf('=== RUNNING ALL SIMULATIONS ===\n\n');

% Loop through all parameter combinations
sim_count = 1;
total_sims = length(dt_values) *    length(ne_values) * length(sigma_ratios);

for i = 1:length(dt_values)
    dt = dt_values(i);
    
    for j = 1:length(ne_values)
        ne_target = ne_values(j);
        nvx = nvx_values(j);
        nvy = nvx;  % Square grid
        ne_actual = (nvx-1)*(nvy-1);
        
        for k = 1:length(sigma_ratios)
            sigma_ratio = sigma_ratios(k);
            
            fprintf('Simulation %d/%d: dt=%.3f, ne=%d, σ_d/σ_h=%.1f\n', ...
                    sim_count, total_sims, dt, ne_actual, sigma_ratio);
            
            try
                % Run simulation
                [u_final, activation_times, is_M_matrix] = ...
                    monodomain_heterogeneous(sigma_ratio, dt, nvx, nvy);
                
                % Extract first activation time
                first_activation = min(activation_times(activation_times < inf));
                if isempty(first_activation)
                    first_activation = NaN;  % No activation
                end
                
                % Check bounds constraint
                bounds_ok = (min(u_final(:)) >= -1e-10) && (max(u_final(:)) <= 1+1e-10);
                
                % Store results
                results = [results; dt, ne_actual, sigma_ratio, first_activation, is_M_matrix, bounds_ok];
                
                fprintf('  → Activation: %.2f ms, M-matrix: %s, Bounds: %s\n\n', ...
                        first_activation, string(is_M_matrix), string(bounds_ok));
                
            catch ME
                fprintf('  → ERROR: %s\n\n', ME.message);
                % Store error case
                results = [results; dt, ne_actual, sigma_ratio, NaN, false, false];
            end
            
            sim_count = sim_count + 1;
        end
    end
end

% Create results table
fprintf('=== SIMULATION RESULTS TABLE ===\n\n');
fprintf('%8s %6s %15s %12s %12s\n', ...
        'Δt', 'ne', 'Activation time', 'M-matrix?', 'u ∈[0,1]');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:size(results, 1)
    dt = results(i, 1);
    ne = results(i, 2);
    sigma_ratio = results(i, 3);
    activation = results(i, 4);
    m_matrix = results(i, 5);
    bounds = results(i, 6);
    
    if isnan(activation)
        activation_str = 'No activation';
    else
        activation_str = sprintf('%.2f ms', activation);
    end
    
    fprintf('%8.3f %6d %15s %12s %12s\n', ...
            dt, ne, activation_str, ...
            string(logical(m_matrix)), string(logical(bounds)));
end

% Convert to table for further use
results_table = array2table(results, ...
    'VariableNames', {'dt', 'ne', 'sigma_ratio', ...
                      'activation_time', 'is_M_matrix', 'bounds_ok'});

% Save to CSV
writetable(results_table, 'simulation_results.csv');

fprintf('\n=== ANALYSIS COMPLETE ===\n');

fprintf('\n=== GRAPHICAL VISUALIZATION ===\n');
% Filter significant results (where activation occurred)
activated = ~isnan(results_table.activation_time);

% Define color coding based on bounds_ok
colors = zeros(height(results_table), 3);  % RGB
colors(results_table.bounds_ok == 1, :) = repmat([0 0.6 0], sum(results_table.bounds_ok == 1), 1);  % Green
colors(results_table.bounds_ok == 0, :) = repmat([0.8 0 0], sum(results_table.bounds_ok == 0), 1);  % Red

% Marker size based on activation
markerSizes = 30 + 70 * activated;  % Bigger if activation happened

% Separate data by bounds_ok
idx_ok = results_table.bounds_ok == 1;
idx_bad = results_table.bounds_ok == 0;

% Plot separately for legend to work
figure;
hold on;

scatter3(results_table.dt(idx_ok), ...
         results_table.ne(idx_ok), ...
         results_table.sigma_ratio(idx_ok), ...
         markerSizes(idx_ok), ...
         'g', 'filled');

scatter3(results_table.dt(idx_bad), ...
         results_table.ne(idx_bad), ...
         results_table.sigma_ratio(idx_bad), ...
         markerSizes(idx_bad), ...
         'r', 'filled');

xlabel('\Delta t');
ylabel('n_e');
zlabel('\sigma_d / \sigma_h');
title('Simulation Results: Activation and Bounds Check');
grid on;
view(135, 20);
legend({'Bounds OK (u∈[0,1])', 'Bounds Violated'}, 'Location', 'bestoutside');

% Filter rows where activation occurred
valid_rows = ~isnan(results_table.activation_time);

% Compute average activation time grouped by sigma_ratio and ne
summary = groupsummary(results_table(valid_rows,:), ...
                      {'sigma_ratio', 'ne'}, ...
                      'mean', 'activation_time');

% Create pivot table
sigma_vals = unique(results_table.sigma_ratio);
ne_vals = unique(results_table.ne);
activation_grid = NaN(length(ne_vals), length(sigma_vals));

for i = 1:height(summary)
    row = find(ne_vals == summary.ne(i));
    col = find(sigma_vals == summary.sigma_ratio(i));
    activation_grid(row, col) = summary.mean_activation_time(i);
end

% Create heatmap
figure;
heatmap(sigma_vals, ne_vals, activation_grid, ...
        'XLabel', '\sigma_d / \sigma_h', ...
        'YLabel', 'n_e', ...
        'Title', 'Avg Activation Time (ms)', ...
        'ColorbarVisible', 'on');

end