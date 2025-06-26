# PINN with Custom Initialization A=0.5, B=-2
# Testing different initialization values for the custom activation function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class CustomActivationCustomInit(nn.Module):
    """Custom activation function: A*(1-tanh(B*x)) with custom initialization"""
    def __init__(self, init_A=0.5, init_B=-2.0):
        super(CustomActivationCustomInit, self).__init__()
        # Initialize with custom values
        self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(init_B, dtype=torch.float32))
    
    def forward(self, x):
        return self.A * (1 - torch.tanh(self.B * x))

class CustomInitPINN(nn.Module):
    """PINN with custom initialization for activation parameters"""
    
    def __init__(self, layers=[3, 64, 64, 64, 64, 1], init_A=0.5, init_B=-2.0):
        super(CustomInitPINN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # Build network with custom initialization
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.activations.append(CustomActivationCustomInit(init_A, init_B))
        
        # Initialize weights
        self.init_weights()
        
        # Problem parameters
        self.sigma_h = 9.5298e-4
        self.a = 18.515
        self.fr = 0.2383
        self.ft = 0.0
        self.fd = 1.0
        
        # Normalization parameters
        self.x_mean = 0.5
        self.x_std = 0.3
        self.t_mean = 3.0
        self.t_std = 2.0
        
    def init_weights(self):
        """Initialize network weights"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def normalize_input(self, x, y, t):
        """Normalize inputs for better training stability"""
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.x_mean) / self.x_std
        t_norm = (t - self.t_mean) / self.t_std
        return x_norm, y_norm, t_norm
    
    def forward(self, inputs):
        """Forward pass with input normalization"""
        x, y, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        
        # Normalize inputs
        x_norm, y_norm, t_norm = self.normalize_input(x, y, t)
        normalized_inputs = torch.cat([x_norm, y_norm, t_norm], dim=1)
        
        # Forward pass
        out = normalized_inputs
        for i, layer in enumerate(self.layers[:-1]):
            out = layer(out)
            out = self.activations[i](out)
        
        # Final layer
        out = self.layers[-1](out)
        
        # Apply sigmoid to keep solution in reasonable range
        out = torch.sigmoid(out)
        
        return out
    
    def cubic_reaction(self, u):
        """Cubic reaction term"""
        return self.a * (u - self.fr) * (u - self.ft) * (u - self.fd)
    
    def get_diffusivity(self, x, y, factor=1.0):
        """Get diffusivity with diseased regions"""
        sigma = torch.full_like(x, self.sigma_h)
        
        # Diseased regions
        d1 = (x - 0.3)**2 + (y - 0.7)**2 < 0.1**2
        d2 = (x - 0.7)**2 + (y - 0.3)**2 < 0.15**2
        d3 = (x - 0.5)**2 + (y - 0.5)**2 < 0.1**2
        
        diseased_mask = d1 | d2 | d3
        sigma[diseased_mask] = factor * self.sigma_h
        
        return sigma

def compute_physics_loss_custom(model, x, y, t, sigma_factor=1.0):
    """Compute physics loss for custom init model"""
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    
    inputs = torch.cat([x, y, t], dim=1)
    u = model(inputs)
    
    # Compute derivatives
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True, retain_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True, retain_graph=True)[0]
    
    # Get diffusivity
    sigma = model.get_diffusivity(x, y, sigma_factor)
    
    # Reaction term
    f_reaction = model.cubic_reaction(u)
    
    # PDE residual
    laplacian = u_xx + u_yy
    pde_residual = u_t - sigma * laplacian - f_reaction
    
    return pde_residual, u

def generate_training_points_custom(n_interior=4000, n_boundary=800, n_initial=800, T=6.0):
    """Generate training points"""
    
    # Interior points
    x_int = torch.rand(n_interior, 1)
    y_int = torch.rand(n_interior, 1)
    t_int = torch.rand(n_interior, 1) * T
    
    # Boundary points
    n_side = n_boundary // 4
    
    x_left = torch.zeros(n_side, 1)
    y_left = torch.rand(n_side, 1)
    t_left = torch.rand(n_side, 1) * T
    
    x_right = torch.ones(n_side, 1)
    y_right = torch.rand(n_side, 1)
    t_right = torch.rand(n_side, 1) * T
    
    x_bottom = torch.rand(n_side, 1)
    y_bottom = torch.zeros(n_side, 1)
    t_bottom = torch.rand(n_side, 1) * T
    
    x_top = torch.rand(n_side, 1)
    y_top = torch.ones(n_side, 1)
    t_top = torch.rand(n_side, 1) * T
    
    x_bc = torch.cat([x_left, x_right, x_bottom, x_top])
    y_bc = torch.cat([y_left, y_right, y_bottom, y_top])
    t_bc = torch.cat([t_left, t_right, t_bottom, t_top])
    
    # Initial condition points
    x_ic = torch.rand(n_initial, 1)
    y_ic = torch.rand(n_initial, 1)
    t_ic = torch.zeros(n_initial, 1)
    
    u_ic_true = torch.zeros(n_initial, 1)
    stimulus_mask = (x_ic >= 0.9) | (y_ic >= 0.9)
    u_ic_true[stimulus_mask] = 1.0
    
    return (x_int, y_int, t_int), (x_bc, y_bc, t_bc), (x_ic, y_ic, t_ic, u_ic_true)

def train_custom_init_pinn(model, epochs=2000, lr=1e-3, sigma_factor=1.0):
    """Train PINN with custom initialization"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=150)
    
    model = model.to(device)
    
    # Generate training data
    (x_int, y_int, t_int), (x_bc, y_bc, t_bc), (x_ic, y_ic, t_ic, u_ic_true) = generate_training_points_custom()
    
    # Move to device
    x_int, y_int, t_int = x_int.to(device), y_int.to(device), t_int.to(device)
    x_bc, y_bc, t_bc = x_bc.to(device), y_bc.to(device), t_bc.to(device)
    x_ic, y_ic, t_ic = x_ic.to(device), y_ic.to(device), t_ic.to(device)
    u_ic_true = u_ic_true.to(device)
    
    losses = {'total': [], 'physics': [], 'boundary': [], 'initial': []}
    
    print(f"Training Custom Init PINN (A=0.5, B=-2.0, σ_factor={sigma_factor})...")
    print(f"Interior points: {len(x_int)}")
    print(f"Boundary points: {len(x_bc)}")
    print(f"Initial points: {len(x_ic)}")
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Physics loss
        pde_residual, _ = compute_physics_loss_custom(model, x_int, y_int, t_int, sigma_factor)
        physics_loss = torch.mean(pde_residual**2)
        
        # Boundary loss
        inputs_bc = torch.cat([x_bc, y_bc, t_bc], dim=1)
        u_bc = model(inputs_bc)
        
        x_bc_grad = x_bc.clone().detach().requires_grad_(True)
        y_bc_grad = y_bc.clone().detach().requires_grad_(True)
        inputs_bc_grad = torch.cat([x_bc_grad, y_bc_grad, t_bc], dim=1)
        u_bc_grad = model(inputs_bc_grad)
        
        u_x_bc = torch.autograd.grad(u_bc_grad.sum(), x_bc_grad, create_graph=True)[0]
        u_y_bc = torch.autograd.grad(u_bc_grad.sum(), y_bc_grad, create_graph=True)[0]
        
        boundary_loss = torch.mean(u_x_bc**2 + u_y_bc**2)
        
        # Initial condition loss
        inputs_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
        u_ic_pred = model(inputs_ic)
        initial_loss = torch.mean((u_ic_pred - u_ic_true)**2)
        
        # Total loss with adaptive weights
        if epoch < 800:
            total_loss = physics_loss + 10.0 * boundary_loss + 1000.0 * initial_loss
        else:
            total_loss = physics_loss + 10.0 * boundary_loss + 100.0 * initial_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        # Store losses
        losses['total'].append(total_loss.item())
        losses['physics'].append(physics_loss.item())
        losses['boundary'].append(boundary_loss.item())
        losses['initial'].append(initial_loss.item())
        
        # Track best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(model.state_dict(), '/tmp/best_custom_pinn_model.pth')
        
        # Print progress
        if epoch % 400 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:4d} | Loss: {total_loss:.2e} | Physics: {physics_loss:.2e} | "
                  f"BC: {boundary_loss:.2e} | IC: {initial_loss:.2e} | Time: {elapsed:.1f}s")
            
            # Print activation parameters
            for i, activation in enumerate(model.activations):
                print(f"    Layer {i+1}: A={activation.A.item():.4f}, B={activation.B.item():.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('/tmp/best_custom_pinn_model.pth'))
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best loss: {best_loss:.2e}")
    
    return model, losses

def evaluate_custom_pinn(model, nx=33, ny=33, nt=101, T=6.0):
    """Evaluate custom init PINN"""
    model.eval()
    
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    t_vals = torch.linspace(0, T, nt)
    
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    solutions = []
    
    print("Evaluating Custom Init PINN...")
    
    with torch.no_grad():
        for i, t_val in enumerate(t_vals):
            x_flat = X.flatten().unsqueeze(1).to(device)
            y_flat = Y.flatten().unsqueeze(1).to(device)
            t_flat = torch.full_like(x_flat, t_val).to(device)
            
            inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)
            u_pred = model(inputs).cpu()
            u_grid = u_pred.reshape(nx, ny)
            
            solutions.append(u_grid.numpy())
            
            if i % 20 == 0:
                print(f"  Time step {i}/{nt} (t={t_val:.2f})")
    
    return np.array(solutions), t_vals.numpy()

# Test the custom initialization
if __name__ == "__main__":
    print("Testing Custom Initialization PINN (A=0.5, B=-2.0)")
    print("=" * 50)
    
    # Create model with custom initialization
    model = CustomInitPINN(layers=[3, 64, 64, 64, 64, 1], init_A=0.5, init_B=-2.0)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}")
    
    print("\nInitial activation parameters:")
    for i, activation in enumerate(model.activations):
        print(f"  Layer {i+1}: A={activation.A.item():.4f}, B={activation.B.item():.4f}")
    
    # Train
    print(f"\nTraining with custom initialization...")
    trained_model, losses = train_custom_init_pinn(model, epochs=2000, lr=1e-3, sigma_factor=1.0)
    
    print("\nFinal activation parameters:")
    for i, activation in enumerate(trained_model.activations):
        print(f"  Layer {i+1}: A={activation.A.item():.4f}, B={activation.B.item():.4f}")
    
    # Evaluate
    print(f"\nEvaluating solution...")
    solutions, times = evaluate_custom_pinn(trained_model)
    
    print(f"Solution shape: {solutions.shape}")
    print(f"Solution range: [{solutions.min():.3f}, {solutions.max():.3f}]")
    
    # Quick visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    times_to_plot = [0, len(times)//2, -1]
    titles = ['t=0', f't={times[len(times)//2]:.1f}', f't={times[-1]:.1f}']
    
    for i, (t_idx, title) in enumerate(zip(times_to_plot, titles)):
        im = axes[i].imshow(solutions[t_idx], origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(title)
        axes[i].set_aspect('equal')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('custom_init_pinn_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Test visualization saved as 'custom_init_pinn_test.png'")

