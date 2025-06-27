# Improved PINN Implementation for Monodomain Equation
# Fixing the physics and training issues

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class CustomActivation(nn.Module):
    """Custom activation function: A*(1-tanh(B*x))"""
    def __init__(self):
        super(CustomActivation, self).__init__()
        # Initialize with better values
        self.A = nn.Parameter(torch.tensor(0.5))
        self.B = nn.Parameter(torch.tensor(-5.0))
    
    def forward(self, x):
        return self.A * (1 - torch.tanh(self.B * x))

class ImprovedPINN(nn.Module):
    """Improved PINN with better architecture and physics implementation"""
    
    def __init__(self, layers=[3, 64, 64, 64, 1]):
        super(ImprovedPINN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # Build deeper network
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.activations.append(CustomActivation())
        
        # Better weight initialization
        self.init_weights()
        
        # Problem parameters
        self.sigma_h = 9.5298e-4
        self.a = 18.515
        self.fr = 0.2383
        self.ft = 0.0
        self.fd = 1.0
        
        # Normalization parameters for better training
        self.x_mean = 0.5
        self.x_std = 0.3
        self.t_mean = 5.0
        self.t_std = 3.0
        
    def init_weights(self):
        """Better weight initialization"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def normalize_input(self, x, y, t):
        """Normalize inputs for better training stability"""
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.x_mean) / self.x_std  # Same as x
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
        """Cubic reaction term with proper scaling"""
        return self.a * (u - self.fr) * (u - self.ft) * (u - self.fd)
    
    def get_diffusivity(self, x, y, factor=1.0):
        """Get diffusivity with diseased regions"""
        sigma = torch.full_like(x, self.sigma_h)
        
        # Diseased regions (same as FEM)
        d1 = (x - 0.3)**2 + (y - 0.7)**2 < 0.1**2
        d2 = (x - 0.7)**2 + (y - 0.3)**2 < 0.15**2
        d3 = (x - 0.5)**2 + (y - 0.5)**2 < 0.1**2
        
        diseased_mask = d1 | d2 | d3
        sigma[diseased_mask] = factor * self.sigma_h
        
        return sigma

def compute_physics_loss(model, x, y, t, sigma_factor=1.0):
    """Improved physics loss computation"""
    # Ensure gradients
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    
    # Forward pass
    inputs = torch.cat([x, y, t], dim=1)
    u = model(inputs)
    
    # Compute first derivatives
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True, retain_graph=True)[0]
    
    # Compute second derivatives
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True, retain_graph=True)[0]
    
    # Get diffusivity
    sigma = model.get_diffusivity(x, y, sigma_factor)
    
    # Reaction term
    f_reaction = model.cubic_reaction(u)
    
    # PDE residual: ∂u/∂t - σ∇²u - f(u) = 0
    laplacian = u_xx + u_yy
    pde_residual = u_t - sigma * laplacian - f_reaction
    
    return pde_residual, u

def generate_training_points(n_interior=5000, n_boundary=1000, n_initial=1000, T=8.0):
    """Generate better distributed training points"""
    
    # Interior points with better sampling
    x_int = torch.rand(n_interior, 1)
    y_int = torch.rand(n_interior, 1)
    t_int = torch.rand(n_interior, 1) * T
    
    # Boundary points - more systematic
    n_side = n_boundary // 4
    
    # Left boundary (x=0)
    x_left = torch.zeros(n_side, 1)
    y_left = torch.rand(n_side, 1)
    t_left = torch.rand(n_side, 1) * T
    
    # Right boundary (x=1)
    x_right = torch.ones(n_side, 1)
    y_right = torch.rand(n_side, 1)
    t_right = torch.rand(n_side, 1) * T
    
    # Bottom boundary (y=0)
    x_bottom = torch.rand(n_side, 1)
    y_bottom = torch.zeros(n_side, 1)
    t_bottom = torch.rand(n_side, 1) * T
    
    # Top boundary (y=1)
    x_top = torch.rand(n_side, 1)
    y_top = torch.ones(n_side, 1)
    t_top = torch.rand(n_side, 1) * T
    
    # Combine boundary points
    x_bc = torch.cat([x_left, x_right, x_bottom, x_top])
    y_bc = torch.cat([y_left, y_right, y_bottom, y_top])
    t_bc = torch.cat([t_left, t_right, t_bottom, t_top])
    
    # Initial condition points
    x_ic = torch.rand(n_initial, 1)
    y_ic = torch.rand(n_initial, 1)
    t_ic = torch.zeros(n_initial, 1)
    
    # True initial condition
    u_ic_true = torch.zeros(n_initial, 1)
    stimulus_mask = (x_ic >= 0.9) | (y_ic >= 0.9)
    u_ic_true[stimulus_mask] = 1.0
    
    return (x_int, y_int, t_int), (x_bc, y_bc, t_bc), (x_ic, y_ic, t_ic, u_ic_true)

def train_improved_pinn(model, epochs=5000, lr=1e-3, sigma_factor=1.0):
    """Improved training with better strategy"""
    
    # Better optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=200)
    
    model = model.to(device)
    
    # Generate training data
    (x_int, y_int, t_int), (x_bc, y_bc, t_bc), (x_ic, y_ic, t_ic, u_ic_true) = generate_training_points()
    
    # Move to device
    x_int, y_int, t_int = x_int.to(device), y_int.to(device), t_int.to(device)
    x_bc, y_bc, t_bc = x_bc.to(device), y_bc.to(device), t_bc.to(device)
    x_ic, y_ic, t_ic = x_ic.to(device), y_ic.to(device), t_ic.to(device)
    u_ic_true = u_ic_true.to(device)
    
    losses = {'total': [], 'physics': [], 'boundary': [], 'initial': []}
    
    print(f"Training Improved PINN (σ_factor={sigma_factor})...")
    print(f"Interior points: {len(x_int)}")
    print(f"Boundary points: {len(x_bc)}")
    print(f"Initial points: {len(x_ic)}")
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Physics loss
        pde_residual, _ = compute_physics_loss(model, x_int, y_int, t_int, sigma_factor)
        physics_loss = torch.mean(pde_residual**2)
        
        # Boundary loss (Neumann BC: ∂u/∂n = 0)
        inputs_bc = torch.cat([x_bc, y_bc, t_bc], dim=1)
        u_bc = model(inputs_bc)
        
        # Compute boundary derivatives (simplified)
        x_bc_grad = x_bc.clone().detach().requires_grad_(True)
        y_bc_grad = y_bc.clone().detach().requires_grad_(True)
        inputs_bc_grad = torch.cat([x_bc_grad, y_bc_grad, t_bc], dim=1)
        u_bc_grad = model(inputs_bc_grad)
        
        u_x_bc = torch.autograd.grad(u_bc_grad.sum(), x_bc_grad, create_graph=True)[0]
        u_y_bc = torch.autograd.grad(u_bc_grad.sum(), y_bc_grad, create_graph=True)[0]
        
        # Boundary loss (zero normal derivative)
        boundary_loss = torch.mean(u_x_bc**2 + u_y_bc**2)
        
        # Initial condition loss
        inputs_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
        u_ic_pred = model(inputs_ic)
        initial_loss = torch.mean((u_ic_pred - u_ic_true)**2)
        
        # Total loss with adaptive weights
        if epoch < 1000:
            # Focus on initial conditions first
            total_loss = physics_loss + 10.0 * boundary_loss + 1000.0 * initial_loss
        else:
            # Then focus on physics
            total_loss = physics_loss + 10.0 * boundary_loss + 100.0 * initial_loss
        
        total_loss.backward()
        
        # Gradient clipping for stability
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
            # Save in a temporary location during training
            torch.save(model.state_dict(), '/tmp/best_pinn_model.pth')
        
        # Print progress
        if epoch % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:4d} | Loss: {total_loss:.2e} | Physics: {physics_loss:.2e} | "
                  f"BC: {boundary_loss:.2e} | IC: {initial_loss:.2e} | Time: {elapsed:.1f}s")
            
            # Print activation parameters
            for i, activation in enumerate(model.activations):
                print(f"    Layer {i+1}: A={activation.A.item():.4f}, B={activation.B.item():.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('/tmp/best_pinn_model.pth'))
    
    # Save the model permanently
    model_save_path = f'improved_pinn_sigma{sigma_factor}.pth'
    torch.save(model.state_dict(), model_save_path)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best loss: {best_loss:.2e}")
    print(f"Model saved to {model_save_path}")
    
    return model, losses

def evaluate_improved_pinn(model, nx=33, ny=33, nt=101, T=8.0):
    """Evaluate improved PINN"""
    model.eval()
    
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    t_vals = torch.linspace(0, T, nt)
    
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    solutions = []
    
    print("Evaluating Improved PINN...")
    
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

def load_trained_model(model_path, device='cpu'):
    """Load a previously trained PINN model
    
    Args:
        model_path (str): Path to the saved model file (.pth)
        device (str): Device to load the model on ('cpu' or 'cuda' or 'mps')
        
    Returns:
        ImprovedPINN: Loaded model
    """
    # Create a new model instance
    model = ImprovedPINN(layers=[3, 64, 64, 64, 64, 1])
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    print(f"Model loaded from {model_path}")
    print("Activation parameters:")
    for i, activation in enumerate(model.activations):
        print(f"  Layer {i+1}: A={activation.A.item():.4f}, B={activation.B.item():.4f}")
    
    return model

# Esempio di utilizzo di un modello salvato
def use_saved_model_example():
    """Example of how to load and use a saved model"""
    # Path to the saved model
    model_path = 'improved_pinn_sigma1.0.pth'
    
    # Check if the file exists
    import os
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Load the model
    model = load_trained_model(model_path, device)
    
    # Evaluate the model
    solutions, times = evaluate_improved_pinn(model)
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    times_to_plot = [0, len(times)//2, -1]
    titles = ['t=0', f't={times[len(times)//2]:.1f}', f't={times[-1]:.1f}']
    
    for i, (t_idx, title) in enumerate(zip(times_to_plot, titles)):
        im = axes[i].imshow(solutions[t_idx], origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(title)
        axes[i].set_aspect('equal')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('loaded_model_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Results from loaded model saved as 'loaded_model_results.png'")

# Test the improved implementation
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or use Improved PINN')
    parser.add_argument('--load', help='Path to saved model to load', type=str, default=None)
    parser.add_argument('--epochs', help='Number of epochs to train', type=int, default=3000)
    parser.add_argument('--sigma', help='Diffusivity factor', type=float, default=1.0)
    args = parser.parse_args()
    
    print("Improved PINN Implementation")
    print("=" * 40)
    
    if args.load:
        # Load and evaluate existing model
        print(f"Loading model from {args.load}...")
        model = load_trained_model(args.load, device)
        trained_model = model  # for consistency with code below
    else:
        # Create and train new model
        print("Creating new model...")
        model = ImprovedPINN(layers=[3, 64, 64, 64, 64, 1])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params}")
        
        print("\nInitial activation parameters:")
        for i, activation in enumerate(model.activations):
            print(f"  Layer {i+1}: A={activation.A.item():.4f}, B={activation.B.item():.4f}")
        
        # Train for normal diffusivity case
        print(f"\nTraining for diffusivity case (sigma_factor={args.sigma})...")
        trained_model, losses = train_improved_pinn(model, epochs=args.epochs, lr=1e-3, sigma_factor=args.sigma)
    
    print("\nFinal activation parameters:")
    for i, activation in enumerate(trained_model.activations):
        print(f"  Layer {i+1}: A={activation.A.item():.4f}, B={activation.B.item():.4f}")
    
    # Evaluate
    print(f"\nEvaluating solution...")
    solutions, times = evaluate_improved_pinn(trained_model)
    
    print(f"Solution shape: {solutions.shape}")
    print(f"Solution range: [{solutions.min():.3f}, {solutions.max():.3f}]")
    
    # Quick visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    times_to_plot = [0, len(times)//2, -1]
    titles = ['t=0', f't={times[len(times)//2]:.1f}', f't={times[-1]:.1f}']
    
    for i, (t_idx, title) in enumerate(zip(times_to_plot, titles)):
        im = axes[i].imshow(solutions[t_idx], origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[i].set_title(title)
        axes[i].set_aspect('equal')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('improved_pinn_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Test visualization saved as 'improved_pinn_test.png'")

