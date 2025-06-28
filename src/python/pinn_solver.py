# src/pinn_solver.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time

class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()
        self.A = nn.Parameter(torch.tensor(0.5))
        self.B = nn.Parameter(torch.tensor(-5.0))
    
    def forward(self, x):
        return self.A * (1 - torch.tanh(self.B * x))

class PINNSolver(nn.Module):
    """Risolve l'equazione del monodominio usando una PINN."""
    def __init__(self, device, sigma_h, a, fr, ft, fd, layers=[3, 64, 64, 64, 1]):
        super(PINNSolver, self).__init__()
        
        self.device = device if device is not None else torch.device('cpu')
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.activations.append(CustomActivation())
        
        self.init_weights()
        
        # Parametri fisici (possono essere aggiornati)
        self.sigma_h = sigma_h
        self.a = a
        self.fr = fr
        self.ft = ft
        self.fd = fd

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            inputs = self.activations[i](layer(inputs))
        output = self.layers[-1](inputs)
        return output

    def get_physics_loss(self, x, y, t):
        u = self(x, y, t)
        
        # Calcolo delle derivate con autograd
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        laplacian = u_xx + u_yy
        reaction = self.a * (u - self.fr) * (u - self.ft) * (u - self.fd)
        
        # Residuo della PDE
        pde_residual = u_t - self.sigma_h * laplacian + reaction
        return torch.mean(pde_residual**2)

    def generate_frames(self, T, nvx, nvy, case_name, output_dir, num_frames=100):
        """Genera i frame delle predizioni dal modello PINN."""
        from .plotting import create_single_frame
        
        print(f"Generazione frame dalla PINN per il caso: {case_name}")
        
        # Crea la directory per i frame se non esiste
        os.makedirs(output_dir, exist_ok=True)
        
        x = np.linspace(0, 1, nvx)
        y = np.linspace(0, 1, nvy)
        X, Y = np.meshgrid(x, y, indexing='ij')
        x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(self.device)
        y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(self.device)
        
        times = np.linspace(0, T, num_frames)
        
        for i, t_val in enumerate(times):
            t_tensor = torch.full_like(x_flat, t_val)
            with torch.no_grad():
                u_pred = self(x_flat, y_flat, t_tensor).cpu().numpy()
            
            fig = create_single_frame(u_pred, nvx, nvy, t_val, case_name)
            plt.savefig(os.path.join(output_dir, f'frame_{i:04d}.png'))
            plt.close(fig)
            if (i + 1) % 10 == 0:
                print(f"  Frame {i+1}/{num_frames} generato.")
                
        print(f"  {num_frames} frame salvati in: {output_dir}")
        return output_dir

class PINNTrainer:
    def __init__(self, model, learning_rate=1e-3, device=None, T=35.0):
        self.device = device if device is not None else torch.device('cpu')
        self.model = model.to(self.device)
        self.T = T
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=500, factor=0.5)

    def train(self, n_epochs, n_points_pde, n_points_ic):
        print("Avvio training della PINN...")
        start_time = time.time()

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            # 1. Loss fisica (punti nel dominio)
            x_pde = torch.rand(n_points_pde, 1, device=self.device, requires_grad=True)
            y_pde = torch.rand(n_points_pde, 1, device=self.device, requires_grad=True)
            t_pde = torch.rand(n_points_pde, 1, device=self.device, requires_grad=True) * self.T
            loss_pde = self.model.get_physics_loss(x_pde, y_pde, t_pde)
            
            # 2. Loss condizione iniziale
            x_ic = torch.rand(n_points_ic, 1, device=self.device)
            y_ic = torch.rand(n_points_ic, 1, device=self.device)
            t_ic = torch.zeros(n_points_ic, 1, device=self.device)
            u_ic_pred = self.model(x_ic, y_ic, t_ic)
            
            u_ic_true = torch.zeros_like(u_ic_pred)
            u_ic_true[(x_ic >= 0.9) & (y_ic >= 0.9)] = 1.0
            loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)
            
            # Loss totale
            total_loss = loss_pde + 100 * loss_ic # Peso maggiore per la IC
            
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss.item():.4e}, "
                      f"Loss PDE: {loss_pde.item():.4e}, Loss IC: {loss_ic.item():.4e}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.4e}")

        end_time = time.time()
        print(f"Training completato in {end_time - start_time:.2f} secondi.")
