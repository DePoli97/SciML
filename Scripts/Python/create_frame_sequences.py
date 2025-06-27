# Simplified Animation Creator - Focus on Frame Sequences
# Create frame sequences that can be easily converted to videos

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for high-quality output
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

def assembleMass(nvx, nvy, hx, hy):
    """Assemble mass matrix for bilinear finite elements on rectangular mesh."""
    Mref = np.array([[1/3, 1/6], [1/6, 1/3]])
    Mx = hx * Mref
    My = hy * Mref
    Aloc = np.kron(My, Mx)
    
    # Number of vertices and elements
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
    I, J, V = [], [], []
    for e in range(ne):
        for i in range(4):
            for j in range(4):
                I.append(conn[i, e])
                J.append(conn[j, e])
                V.append(Aloc[i, j])
    
    # Assemble the sparse matrix
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
    
    if len(sigma_elements) != ne:
        raise ValueError(f'sigma_elements must have length equal to number of elements ({ne})')
    
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
    """Cubic reaction term for cardiac ionic dynamics."""
    return a * (u - fr) * (u - ft) * (u - fd)

def setup_initial_condition(nvx, nvy):
    """Setup initial condition for cardiac stimulation."""
    x = np.linspace(0, 1, nvx)
    y = np.linspace(0, 1, nvy)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u0 = np.zeros((nvx, nvy))
    u0[X >= 0.9] = 1.0
    u0[Y >= 0.9] = 1.0
    u0 = u0.flatten(order='F')
    
    return u0

def setup_diffusivity(nvx, nvy, sigma_h, sigma_d_factor):
    """Setup element-wise diffusivity based on diseased regions."""
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

def create_single_frame(u_data, nvx, nvy, time_val, case_name, vmin=0, vmax=1):
    """Create a single animation frame."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create coordinate grids
    x = np.linspace(0, 1, nvx)
    y = np.linspace(0, 1, nvy)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Reshape solution data
    u_grid = u_data.reshape(nvx, nvy, order='F')
    
    # Create contour plot
    levels = np.linspace(vmin, vmax, 21)
    contour = ax.contourf(X, Y, u_grid, levels=levels, cmap='viridis', extend='both')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Transmembrane Potential (u)', fontsize=12)
    
    # Add diseased regions
    circle1 = patches.Circle((0.3, 0.7), 0.1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    circle2 = patches.Circle((0.7, 0.3), 0.15, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    circle3 = patches.Circle((0.5, 0.5), 0.1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # Add initial stimulus region
    rect = patches.Rectangle((0.9, 0.9), 0.1, 0.1, linewidth=2, edgecolor='white', facecolor='none', linestyle='-')
    ax.add_patch(rect)
    
    # Formatting
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(f'{case_name}\\nTime = {time_val:.2f} ms', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.text(0.02, 0.98, 'Diseased regions', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', color='red')
    ax.text(0.02, 0.90, 'Initial stimulus', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', color='white')
    
    plt.tight_layout()
    return fig

def run_simulation_and_create_frames(nvx, nvy, T, dt, sigma_h, sigma_d_factor, a, fr, ft, fd, case_name, output_dir='frames'):
    """Run simulation and create frame sequence directly."""
    
    print(f"Processing {case_name}...")
    
    # Setup mesh
    hx = 1.0 / (nvx - 1)
    hy = 1.0 / (nvy - 1)
    nv = nvx * nvy
    nt = int(T / dt)
    
    # Setup diffusivity
    sigma_elements = setup_diffusivity(nvx, nvy, sigma_h, sigma_d_factor)
    
    # Assemble matrices
    print("  Assembling matrices...")
    M = assembleMass(nvx, nvy, hx, hy)
    K = assembleDiffusion(nvx, nvy, hx, hy, sigma_elements)
    A = M + dt * K
    
    # Initial condition
    u = setup_initial_condition(nvx, nvy)
    
    # Create output directory
    frame_dir = os.path.join(output_dir, case_name)
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    # Store first few frames to determine global min/max
    print("  Running simulation...")
    u_samples = [u.copy()]
    times_samples = [0.0]
    
    # Run a few steps to get range
    for n in range(min(50, nt)):
        t = (n + 1) * dt
        f_reaction = cubic_reaction(u, a, fr, ft, fd)
        rhs = M.dot(u) - dt * M.dot(f_reaction)
        u_new = spsolve(A, rhs)
        u = u_new
        if n % 5 == 0:
            u_samples.append(u.copy())
            times_samples.append(t)
    
    # Determine global min/max for consistent color scale
    all_u = np.concatenate(u_samples)
    vmin, vmax = np.min(all_u), np.max(all_u)
    print(f"  Solution range: [{vmin:.3f}, {vmax:.3f}]")
    
    # Reset and run full simulation with frame generation
    u = setup_initial_condition(nvx, nvy)
    frame_count = 0
    save_every = max(1, nt // 100)  # Save ~100 frames total
    
    # Save initial frame
    fig = create_single_frame(u, nvx, nvy, 0.0, case_name, vmin, vmax)
    frame_filename = os.path.join(frame_dir, f'frame_{frame_count:04d}.png')
    plt.savefig(frame_filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    frame_count += 1
    
    print(f"  Generating frames (saving every {save_every} steps)...")
    
    # Time integration with frame generation
    for n in range(nt):
        t = (n + 1) * dt
        
        # Compute reaction term
        f_reaction = cubic_reaction(u, a, fr, ft, fd)
        
        # Right-hand side
        rhs = M.dot(u) - dt * M.dot(f_reaction)
        
        # Solve linear system
        u_new = spsolve(A, rhs)
        u = u_new
        
        # Save frame
        if n % save_every == 0 or n == nt - 1:
            fig = create_single_frame(u, nvx, nvy, t, case_name, vmin, vmax)
            frame_filename = os.path.join(frame_dir, f'frame_{frame_count:04d}.png')
            plt.savefig(frame_filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            frame_count += 1
        
        # Progress update
        if (n + 1) % (nt // 5) == 0:
            print(f"    Step {n+1}/{nt} (t={t:.1f}), frames: {frame_count}")
    
    print(f"  Completed: {frame_count} frames saved to {frame_dir}")
    
    # Create video compilation script
    script_filename = os.path.join(frame_dir, 'create_video.sh')
    with open(script_filename, 'w') as f:
        f.write(f"""#!/bin/bash
# Script to create video from frame sequence
# Requires ffmpeg to be installed

echo "Creating videos from {frame_count} frames..."

# Create MP4 video (high quality, 10 fps)
ffmpeg -y -r 10 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 {case_name}_wave_propagation.mp4

# Create WebM video (web-friendly, 10 fps)
ffmpeg -y -r 10 -i frame_%04d.png -c:v libvpx-vp9 -crf 30 -b:v 0 {case_name}_wave_propagation.webm

# Create animated GIF (lower quality, smaller file)
ffmpeg -y -r 5 -i frame_%04d.png -vf "scale=640:512" {case_name}_wave_propagation.gif

echo "Video compilation completed!"
echo "Generated files:"
echo "  {case_name}_wave_propagation.mp4 (high quality)"
echo "  {case_name}_wave_propagation.webm (web format)"
echo "  {case_name}_wave_propagation.gif (animated GIF)"
""")
    
    os.chmod(script_filename, 0o755)
    
    return frame_dir, frame_count, script_filename

# Problem parameters
sigma_h = 9.5298e-4
a = 18.515
fr = 0.2383
ft = 0.0
fd = 1.0
T = 15.0  # Shorter duration for manageable file sizes

# Animation parameters
nvx = nvy = 33  # 32x32 elements for good balance
dt = 0.15       # Larger time step for fewer frames

print("Creating Frame Sequences for Video Animation")
print("=" * 60)

# Define cases
diffusivity_cases = [
    {'factor': 10.0, 'name': 'High_Diffusivity_10x', 'title': 'High Diffusivity (10×σh)'},
    {'factor': 1.0, 'name': 'Normal_Diffusivity_1x', 'title': 'Normal Diffusivity (1×σh)'},
    {'factor': 0.1, 'name': 'Low_Diffusivity_01x', 'title': 'Low Diffusivity (0.1×σh)'}
]

# Create output directory
os.makedirs('frames', exist_ok=True)

results = []

for i, case in enumerate(diffusivity_cases):
    print(f"\nCase {i+1}: {case['title']}")
    print("-" * 50)
    
    start_time = time.time()
    
    frame_dir, frame_count, script = run_simulation_and_create_frames(
        nvx=nvx, nvy=nvy, T=T, dt=dt,
        sigma_h=sigma_h, sigma_d_factor=case['factor'],
        a=a, fr=fr, ft=ft, fd=fd,
        case_name=case['name']
    )
    
    elapsed = time.time() - start_time
    
    results.append({
        'case': case['name'],
        'title': case['title'],
        'frames': frame_count,
        'directory': frame_dir,
        'script': script,
        'time': elapsed
    })
    
    print(f"  Completed in {elapsed:.1f} seconds")

print(f"\n{'='*60}")
print("Frame Sequence Generation Completed!")
print(f"{'='*60}")

print("\n🎬 FRAME SEQUENCES CREATED:")
for result in results:
    print(f"  ✅ {result['title']}")
    print(f"     Frames: {result['frames']}")
    print(f"     Directory: {result['directory']}")
    print(f"     Script: {result['script']}")
    print(f"     Time: {result['time']:.1f}s")
    print()
