# Complete Improved PINN Training and Create New Comparisons
# This script finishes the training and creates new GIFs

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import os
from pinn_improved import ImprovedPINN, train_improved_pinn, evaluate_improved_pinn
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Set up matplotlib
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.dpi'] = 150

def assembleMass(nvx, nvy, hx, hy):
    """Assemble mass matrix for bilinear finite elements."""
    Mref = np.array([[1/3, 1/6], [1/6, 1/3]])
    Mx = hx * Mref
    My = hy * Mref
    Aloc = np.kron(My, Mx)
    
    nv = nvx * nvy
    ne = (nvx - 1) * (nvy - 1)
    
    id_matrix = np.arange(1, nv + 1).reshape(nvx, nvy, order='F')
    a = id_matrix[:-1, :-1].flatten(order='F') - 1
    b = id_matrix[1:, :-1].flatten(order='F') - 1
    c = id_matrix[:-1, 1:].flatten(order='F') - 1
    d = id_matrix[1:, 1:].flatten(order='F') - 1
    conn = np.array([a, b, c, d])
    
    I, J, V = [], [], []
    for e in range(ne):
        for i in range(4):
            for j in range(4):
                I.append(conn[i, e])
                J.append(conn[j, e])
                V.append(Aloc[i, j])
    
    A = sp.csr_matrix((V, (I, J)), shape=(nv, nv))
    return A

def assembleDiffusion(nvx, nvy, hx, hy, sigma_elements):
    """Assemble diffusion matrix with element-wise varying diffusivity."""
    Aref = np.array([[1, -1], [-1, 1]])
    Mref = np.array([[1/3, 1/6], [1/6, 1/3]])
    
    Ax = (1/hx) * Aref
    Ay = (1/hy) * Aref
    Mx = hx * Mref
    My = hy * Mref
    
    nv = nvx * nvy
    ne = (nvx - 1) * (nvy - 1)
    
    id_matrix = np.arange(1, nv + 1).reshape(nvx, nvy, order='F')
    a = id_matrix[:-1, :-1].flatten(order='F') - 1
    b = id_matrix[1:, :-1].flatten(order='F') - 1
    c = id_matrix[:-1, 1:].flatten(order='F') - 1
    d = id_matrix[1:, 1:].flatten(order='F') - 1
    conn = np.array([a, b, c, d])
    
    I, J, V = [], [], []
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
    """Cubic reaction term."""
    return a * (u - fr) * (u - ft) * (u - fd)

def setup_initial_condition(nvx, nvy):
    """Setup initial condition."""
    x = np.linspace(0, 1, nvx)
    y = np.linspace(0, 1, nvy)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u0 = np.zeros((nvx, nvy))
    u0[X >= 0.9] = 1.0
    u0[Y >= 0.9] = 1.0
    u0 = u0.flatten(order='F')
    
    return u0

def setup_diffusivity(nvx, nvy, sigma_h, sigma_d_factor):
    """Setup element-wise diffusivity."""
    ne = (nvx - 1) * (nvy - 1)
    sigma_elements = np.full(ne, sigma_h)
    
    x_centers = np.linspace(0, 1, nvx-1) + 0.5/(nvx-1)
    y_centers = np.linspace(0, 1, nvy-1) + 0.5/(nvy-1)
    X_centers, Y_centers = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    for i in range(nvx-1):
        for j in range(nvy-1):
            e = i * (nvy-1) + j
            x_c = X_centers[i, j]
            y_c = Y_centers[i, j]
            
            in_d1 = (x_c - 0.3)**2 + (y_c - 0.7)**2 < 0.1**2
            in_d2 = (x_c - 0.7)**2 + (y_c - 0.3)**2 < 0.15**2
            in_d3 = (x_c - 0.5)**2 + (y_c - 0.5)**2 < 0.1**2
            
            if in_d1 or in_d2 or in_d3:
                sigma_elements[e] = sigma_d_factor * sigma_h
    
    return sigma_elements

def run_fem_simulation(nvx, nvy, T, dt, sigma_h, sigma_d_factor, a, fr, ft, fd):
    """Run FEM simulation."""
    print(f"Running FEM simulation...")
    
    hx = 1.0 / (nvx - 1)
    hy = 1.0 / (nvy - 1)
    nt = int(T / dt)
    
    sigma_elements = setup_diffusivity(nvx, nvy, sigma_h, sigma_d_factor)
    
    M = assembleMass(nvx, nvy, hx, hy)
    K = assembleDiffusion(nvx, nvy, hx, hy, sigma_elements)
    A = M + dt * K
    
    u = setup_initial_condition(nvx, nvy)
    
    u_history = [u.copy()]
    times = [0.0]
    
    for n in range(nt):
        t = (n + 1) * dt
        f_reaction = cubic_reaction(u, a, fr, ft, fd)
        rhs = M.dot(u) - dt * M.dot(f_reaction)
        u_new = spsolve(A, rhs)
        u = u_new
        
        u_history.append(u.copy())
        times.append(t)
        
        if (n + 1) % (nt // 5) == 0:
            print(f"  FEM step {n+1}/{nt} (t={t:.1f})")
    
    fem_solutions = []
    for u_vec in u_history:
        u_grid = u_vec.reshape(nvx, nvy, order='F')
        fem_solutions.append(u_grid)
    
    return np.array(fem_solutions), np.array(times)

def create_improved_comparison_gif(fem_solutions, pinn_solutions, times, case_name, output_dir='improved_comparison_gifs'):
    """Create improved side-by-side comparison GIF."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    nvx, nvy = fem_solutions.shape[1], fem_solutions.shape[2]
    
    print(f"Creating IMPROVED comparison GIF for {case_name}...")
    
    # Determine global min/max
    all_fem = fem_solutions.flatten()
    all_pinn = pinn_solutions.flatten()
    vmin = min(np.min(all_fem), np.min(all_pinn))
    vmax = max(np.max(all_fem), np.max(all_pinn))
    
    print(f"  Value range: [{vmin:.3f}, {vmax:.3f}]")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    def animate(frame):
        for ax in axes:
            ax.clear()
        
        x = np.linspace(0, 1, nvx)
        y = np.linspace(0, 1, nvy)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        levels = np.linspace(vmin, vmax, 21)
        
        # FEM plot
        contour1 = axes[0].contourf(X, Y, fem_solutions[frame], levels=levels, cmap='viridis', extend='both')
        axes[0].set_title(f'FEM Solution\\nTime = {times[frame]:.2f} ms', fontsize=14, fontweight='bold')
        
        # PINN plot
        contour2 = axes[1].contourf(X, Y, pinn_solutions[frame], levels=levels, cmap='viridis', extend='both')
        axes[1].set_title(f'Improved PINN Solution\\nTime = {times[frame]:.2f} ms', fontsize=14, fontweight='bold')
        
        # Difference plot
        diff = np.abs(fem_solutions[frame] - pinn_solutions[frame])
        contour3 = axes[2].contourf(X, Y, diff, levels=np.linspace(0, np.max(diff), 21), cmap='Reds', extend='max')
        axes[2].set_title(f'Absolute Difference\\nMax = {np.max(diff):.3f}', fontsize=14, fontweight='bold')
        
        # Add diseased regions to all plots
        for ax in axes:
            circle1 = patches.Circle((0.3, 0.7), 0.1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            circle2 = patches.Circle((0.7, 0.3), 0.15, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            circle3 = patches.Circle((0.5, 0.5), 0.1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.add_patch(circle3)
            
            rect = patches.Rectangle((0.9, 0.9), 0.1, 0.1, linewidth=2, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
            
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'FEM vs Improved PINN Comparison - {case_name}', fontsize=16, fontweight='bold')
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(fem_solutions), interval=200, blit=False, repeat=True)
    
    # Save as GIF
    gif_filename = os.path.join(output_dir, f'{case_name}_FEM_vs_ImprovedPINN.gif')
    writer = PillowWriter(fps=5)
    anim.save(gif_filename, writer=writer)
    
    plt.close(fig)
    print(f"  Saved: {gif_filename}")
    
    return gif_filename

def main():
    """Main function to complete training and create improved comparisons."""
    
    print("Completing Improved PINN Training and Creating New Comparisons")
    print("=" * 60)
    
    # Parameters
    sigma_h = 9.5298e-4
    a = 18.515
    fr = 0.2383
    ft = 0.0
    fd = 1.0
    T = 6.0  # Shorter for faster comparison
    dt = 0.06
    nvx = nvy = 33
    
    # Test case: Normal diffusivity
    sigma_d_factor = 1.0
    case_name = "Normal_Diffusivity_Improved"
    
    print(f"\nCase: {case_name} (σ_diseased = {sigma_d_factor}×σ_h)")
    print("-" * 50)
    
    # Step 1: Complete PINN training
    print("Step 1: Training Improved PINN...")
    model = ImprovedPINN(layers=[3, 64, 64, 64, 64, 1])
    
    # Train with reasonable epochs for demonstration
    trained_model, losses = train_improved_pinn(model, epochs=2000, lr=1e-3, sigma_factor=sigma_d_factor)
    
    print("\nFinal activation parameters:")
    for i, activation in enumerate(trained_model.activations):
        print(f"  Layer {i+1}: A={activation.A.item():.4f}, B={activation.B.item():.4f}")
    
    # Step 2: Evaluate PINN
    print("\nStep 2: Evaluating Improved PINN solution...")
    pinn_solutions, pinn_times = evaluate_improved_pinn(trained_model, nvx, nvy, int(T/dt)+1, T)
    
    print(f"PINN solution range: [{pinn_solutions.min():.3f}, {pinn_solutions.max():.3f}]")
    
    # Step 3: Run FEM simulation
    print("\nStep 3: Running FEM simulation...")
    fem_solutions, fem_times = run_fem_simulation(nvx, nvy, T, dt, sigma_h, sigma_d_factor, a, fr, ft, fd)
    
    print(f"FEM solution range: [{fem_solutions.min():.3f}, {fem_solutions.max():.3f}]")
    
    # Step 4: Create improved comparison
    print("\nStep 4: Creating improved comparison animation...")
    gif_file = create_improved_comparison_gif(fem_solutions, pinn_solutions, fem_times, case_name)
    
    # Step 5: Compute improved error metrics
    print("\nStep 5: Computing error metrics...")
    
    l2_error = np.sqrt(np.mean((fem_solutions - pinn_solutions)**2))
    max_error = np.max(np.abs(fem_solutions - pinn_solutions))
    rel_error = l2_error / np.sqrt(np.mean(fem_solutions**2)) * 100
    
    print(f"\nImproved PINN Error Analysis:")
    print(f"  L2 Error: {l2_error:.6f}")
    print(f"  Max Error: {max_error:.6f}")
    print(f"  Relative Error: {rel_error:.2f}%")
    
    # Compare with original PINN (if available)
    print(f"\nComparison completed!")
    print(f"  Improved Animation: {gif_file}")
    
    # Create a quick static comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    times_to_show = [0, len(fem_times)//2, -1]
    
    for i, t_idx in enumerate(times_to_show):
        # FEM
        im1 = axes[0, i].imshow(fem_solutions[t_idx], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'FEM t={fem_times[t_idx]:.1f}ms')
        
        # PINN
        im2 = axes[1, i].imshow(pinn_solutions[t_idx], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Improved PINN t={pinn_times[t_idx]:.1f}ms')
        
        for ax in [axes[0, i], axes[1, i]]:
            ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/improved_pinn_comparison_static.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Static comparison: improved_pinn_comparison_static.png")
    
    return trained_model, gif_file, l2_error, rel_error

if __name__ == "__main__":
    model, gif_file, l2_error, rel_error = main()

