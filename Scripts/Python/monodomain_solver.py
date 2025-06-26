# Complete MATLAB/Python Solver for Monodomain Equation
# This implements the IMEX scheme for the cardiac tissue simulation

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def assembleMass(nvx, nvy, hx, hy):
    """Assemble mass matrix"""
    Mref = np.array([[1/3, 1/6], [1/6, 1/3]])
    
    Mx = hx * Mref
    My = hy * Mref
    
    Aloc = np.kron(My, Mx)
    
    nv = nvx * nvy
    ne = (nvx - 1) * (nvy - 1)
    
    # Create connectivity matrix
    id_matrix = np.arange(1, nv + 1).reshape(nvx, nvy, order='F')
    
    a = id_matrix[:-1, :-1].flatten(order='F') - 1
    b = id_matrix[1:, :-1].flatten(order='F') - 1
    c = id_matrix[:-1, 1:].flatten(order='F') - 1
    d = id_matrix[1:, 1:].flatten(order='F') - 1
    
    conn = np.array([a, b, c, d])
    
    # Create sparse matrix
    I = []
    J = []
    V = []
    
    for e in range(ne):
        for i in range(4):
            for j in range(4):
                I.append(conn[i, e])
                J.append(conn[j, e])
                V.append(Aloc[i, j])
    
    A = sp.csr_matrix((V, (I, J)), shape=(nv, nv))
    return A

def assembleDiffusion(nvx, nvy, hx, hy, sigma_elements):
    """Assemble diffusion matrix with element-wise diffusivity"""
    Aref = np.array([[1, -1], [-1, 1]])
    Mref = np.array([[1/3, 1/6], [1/6, 1/3]])
    
    Ax = (1/hx) * Aref
    Ay = (1/hy) * Aref
    
    Mx = hx * Mref
    My = hy * Mref
    
    nv = nvx * nvy
    ne = (nvx - 1) * (nvy - 1)
    
    if len(sigma_elements) != ne:
        raise ValueError(f'sigma_elements must have length equal to number of elements ({ne})')
    
    # Create connectivity matrix
    id_matrix = np.arange(1, nv + 1).reshape(nvx, nvy, order='F')
    
    a = id_matrix[:-1, :-1].flatten(order='F') - 1
    b = id_matrix[1:, :-1].flatten(order='F') - 1
    c = id_matrix[:-1, 1:].flatten(order='F') - 1
    d = id_matrix[1:, 1:].flatten(order='F') - 1
    
    conn = np.array([a, b, c, d])
    
    # Create sparse matrix
    I = []
    J = []
    V = []
    
    for e in range(ne):
        Aloc = sigma_elements[e] * (np.kron(My, Ax) + np.kron(Ay, Mx))
        
        for i in range(4):
            for j in range(4):
                I.append(conn[i, e])
                J.append(conn[j, e])
                V.append(Aloc[i, j])
    
    A = sp.csr_matrix((V, (I, J)), shape=(nv, nv))
    return A

def cubic_reaction(u, a, fr, ft, fd):
    """Cubic reaction term f(u) = a(u - fr)(u - ft)(u - fd)"""
    return a * (u - fr) * (u - ft) * (u - fd)

def setup_initial_condition(nvx, nvy):
    """Setup initial condition u0"""
    nv = nvx * nvy
    x = np.linspace(0, 1, nvx)
    y = np.linspace(0, 1, nvy)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Initial condition: u0 = 1 if x >= 0.9 and y >= 0.9, else 0
    u0 = np.zeros((nvx, nvy))
    u0[X >= 0.9] = 1.0
    u0[Y >= 0.9] = 1.0
    u0 = u0.flatten(order='F')
    
    return u0

def setup_diffusivity(nvx, nvy, sigma_h, sigma_d_factor):
    """Setup element-wise diffusivity based on diseased regions"""
    ne = (nvx - 1) * (nvy - 1)
    sigma_elements = np.full(ne, sigma_h)  # Default to healthy tissue
    
    # Define diseased regions (circular regions as specified in the problem)
    x_centers = np.linspace(0, 1, nvx-1) + 0.5/(nvx-1)  # Element centers
    y_centers = np.linspace(0, 1, nvy-1) + 0.5/(nvy-1)
    X_centers, Y_centers = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    # Diseased regions (circles)
    # Ωd1: (x-0.3)² + (y-0.7)² < 0.1²
    # Ωd2: (x-0.7)² + (y-0.3)² < 0.15²  
    # Ωd3: (x-0.5)² + (y-0.5)² < 0.1²
    
    for i in range(nvx-1):
        for j in range(nvy-1):
            e = i * (nvy-1) + j  # Element index
            x_c = X_centers[i, j]
            y_c = Y_centers[i, j]
            
            # Check if element center is in any diseased region
            in_d1 = (x_c - 0.3)**2 + (y_c - 0.7)**2 < 0.1**2
            in_d2 = (x_c - 0.7)**2 + (y_c - 0.3)**2 < 0.15**2
            in_d3 = (x_c - 0.5)**2 + (y_c - 0.5)**2 < 0.1**2
            
            if in_d1 or in_d2 or in_d3:
                sigma_elements[e] = sigma_d_factor * sigma_h
    
    return sigma_elements

def monodomain_solver(nvx, nvy, T, dt, sigma_h, sigma_d_factor, a, fr, ft, fd):
    """
    Main solver for the monodomain equation using IMEX scheme
    
    Parameters:
    nvx, nvy: number of vertices in x and y directions
    T: final time
    dt: time step
    sigma_h: healthy tissue diffusivity
    sigma_d_factor: diseased tissue diffusivity factor (relative to sigma_h)
    a, fr, ft, fd: reaction term parameters
    """
    
    # Setup mesh
    hx = 1.0 / (nvx - 1)
    hy = 1.0 / (nvy - 1)
    nv = nvx * nvy
    nt = int(T / dt)
    
    print(f"Solver setup:")
    print(f"  Grid: {nvx}x{nvy} vertices")
    print(f"  Mesh spacing: hx={hx:.4f}, hy={hy:.4f}")
    print(f"  Time steps: {nt}, dt={dt:.4f}")
    print(f"  Final time: T={T}")
    
    # Setup diffusivity
    sigma_elements = setup_diffusivity(nvx, nvy, sigma_h, sigma_d_factor)
    print(f"  Diseased elements: {np.sum(sigma_elements != sigma_h)}/{len(sigma_elements)}")
    
    # Assemble matrices
    print("Assembling matrices...")
    M = assembleMass(nvx, nvy, hx, hy)
    K = assembleDiffusion(nvx, nvy, hx, hy, sigma_elements)
    
    # System matrix (M + dt*K)
    A = M + dt * K
    
    # Initial condition
    u = setup_initial_condition(nvx, nvy)
    
    # Storage for results
    u_history = [u.copy()]
    times = [0.0]
    activation_times = np.full(nv, np.inf)  # Time when u > ft for each node
    
    print("Starting time integration...")
    start_time = time.time()
    
    # Time integration loop
    for n in range(nt):
        t = (n + 1) * dt
        
        # Compute reaction term at current time
        f_reaction = cubic_reaction(u, a, fr, ft, fd)
        
        # Right-hand side: M*u - dt*f(u)
        rhs = M.dot(u) - dt * M.dot(f_reaction)
        
        # Solve linear system: (M + dt*K) * u_new = rhs
        u_new = spsolve(A, rhs)
        
        # Update activation times
        activated = (u <= ft) & (u_new > ft)
        activation_times[activated] = t
        
        # Update solution
        u = u_new
        
        # Store results (every 10th step to save memory)
        if n % 10 == 0 or n == nt - 1:
            u_history.append(u.copy())
            times.append(t)
        
        # Progress update
        if (n + 1) % (nt // 10) == 0:
            print(f"  Step {n+1}/{nt} (t={t:.3f}), max(u)={np.max(u):.3f}, min(u)={np.min(u):.3f}")
    
    elapsed_time = time.time() - start_time
    print(f"Time integration completed in {elapsed_time:.2f} seconds")
    
    # Check constraints
    u_min = np.min([np.min(u_step) for u_step in u_history])
    u_max = np.max([np.max(u_step) for u_step in u_history])
    print(f"Solution bounds: u ∈ [{u_min:.6f}, {u_max:.6f}]")
    print(f"Constraint u ∈ [0,1] satisfied: {u_min >= -1e-10 and u_max <= 1 + 1e-10}")
    
    # Check if matrix is M-matrix
    # A matrix is M-matrix if diagonal entries are positive and off-diagonal entries are non-positive
    A_dense = A.toarray()
    diag_positive = np.all(np.diag(A_dense) > 0)
    off_diag_nonpositive = np.all(A_dense - np.diag(np.diag(A_dense)) <= 1e-12)
    is_M_matrix = diag_positive and off_diag_nonpositive
    print(f"Matrix is M-matrix: {is_M_matrix}")
    
    return {
        'u_history': u_history,
        'times': times,
        'activation_times': activation_times,
        'nvx': nvx,
        'nvy': nvy,
        'sigma_elements': sigma_elements,
        'is_M_matrix': is_M_matrix,
        'u_bounds': (u_min, u_max)
    }

# Test the solver
if __name__ == "__main__":
    # Problem parameters from the PDF
    sigma_h = 9.5298e-4  # Healthy tissue diffusivity
    a = 18.515
    fr = 0.2383
    ft = 0.0
    fd = 1.0
    T = 35.0  # Final time
    
    # Test with different parameters
    test_cases = [
        {'nvx': 33, 'nvy': 33, 'dt': 0.1, 'sigma_d_factor': 10.0},   # 10*Σh
        {'nvx': 33, 'nvy': 33, 'dt': 0.1, 'sigma_d_factor': 1.0},    # Σh  
        {'nvx': 33, 'nvy': 33, 'dt': 0.1, 'sigma_d_factor': 0.1},    # 0.1*Σh
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Running test case {i+1}: σd = {case['sigma_d_factor']}*σh")
        print(f"{'='*60}")
        
        result = monodomain_solver(
            nvx=case['nvx'],
            nvy=case['nvy'], 
            T=T,
            dt=case['dt'],
            sigma_h=sigma_h,
            sigma_d_factor=case['sigma_d_factor'],
            a=a, fr=fr, ft=ft, fd=fd
        )
        
        results.append(result)
        
        # Compute activation time statistics
        finite_activation_times = result['activation_times'][result['activation_times'] < np.inf]
        if len(finite_activation_times) > 0:
            print(f"Activation statistics:")
            print(f"  Nodes activated: {len(finite_activation_times)}/{len(result['activation_times'])}")
            print(f"  First activation: {np.min(finite_activation_times):.3f}")
            print(f"  Last activation: {np.max(finite_activation_times):.3f}")
        else:
            print("No nodes activated during simulation")
    
    print(f"\n{'='*60}")
    print("All test cases completed!")
    print(f"{'='*60}")