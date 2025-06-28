function [u_final, activation_times] = monodomain_solver_ex_1_5()
% MATLAB solver for the monodomain equation using IMEX scheme
% Approximately 50 lines as requested

% Problem parameters
sigma_h = 9.5298e-4;  % Healthy tissue diffusivity
a = 18.515; fr = 0; ft = 0.2383; fd = 1;  % Reaction parameters
T = 35; dt = 0.1;  % Time parameters
nvx = 33; nvy = 33;  % Grid size (matching ne = 64x64 elements approximately)

% Setup mesh
hx = 1/(nvx-1); hy = 1/(nvy-1);
nv = nvx*nvy; nt = T/dt;

% Setup initial condition
[X,Y] = meshgrid(linspace(0,1,nvx), linspace(0,1,nvy));
u0 = zeros(nvx,nvy);
u0(X >= 0.9 & Y >= 0.9) = 1;
u = u0(:);  % Vectorize

% Setup uniform diffusivity (homogeneous case)
ne = (nvx-1)*(nvy-1);
sigma_elements = sigma_h * ones(ne,1);

% Assemble matrices
fprintf('Assembling matrices...\n');
M = assembleMass(nvx, nvy, hx, hy);
K = assembleDiffusion_modified(nvx, nvy, hx, hy, sigma_elements);
A = M + dt*K;  % System matrix

% Initialize activation tracking
activation_times = inf(nv,1);

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
if min(u) >= -1e-10 && max(u) <= 1+1e-10
    fprintf('Constraint u ∈ [0,1]: SATISFIED\n');
else
    fprintf('Constraint u ∈ [0,1]: VIOLATED\n');
end

end