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
        self.B = nn.Parameter(torch.tensor(-2.0))
    
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

    def compute_solution(self, T, nvx, nvy, num_frames=100):
        """
        Calcola la soluzione numerica per diversi istanti temporali.
        
        Args:
            T (float): Tempo finale della simulazione.
            nvx (int): Numero di punti nella direzione x.
            nvy (int): Numero di punti nella direzione y.
            num_frames (int, optional): Numero di frame temporali da calcolare. Default 100.
            
        Returns:
            dict: Un dizionario contenente:
                - 'x': coordinate x della griglia.
                - 'y': coordinate y della griglia.
                - 'times': array dei tempi simulati.
                - 'solutions': lista di soluzioni per ogni istante temporale.
        """
        print("Calcolo della soluzione PINN...")
        
        # Prepara la griglia
        x = np.linspace(0, 1, nvx)
        y = np.linspace(0, 1, nvy)
        X, Y = np.meshgrid(x, y, indexing='ij')
        x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(self.device)
        y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Calcola la soluzione per diversi istanti di tempo
        times = np.linspace(0, T, num_frames)
        solutions = []
        
        for i, t_val in enumerate(times):
            t_tensor = torch.full_like(x_flat, t_val)
            with torch.no_grad():
                u_pred = self(x_flat, y_flat, t_tensor).cpu().numpy()
                # Reshaping per ottenere una griglia 2D
                u_grid = u_pred.reshape(nvx, nvy, order='F')
                solutions.append(u_grid)
            
            if (i + 1) % 10 == 0:
                print(f"  Istante {i+1}/{num_frames} calcolato (t={t_val:.2f}).")
        
        return {
            'x': x,
            'y': y,
            'times': times,
            'solutions': solutions
        }

class PINNTrainer:
    def __init__(self, model, learning_rate=1e-3, device=None, T=35.0):
        self.device = device if device is not None else torch.device('cpu')
        self.model = model.to(self.device)
        self.T = T
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=500, factor=0.5, min_lr=1e-7
        )

    def train(self, n_epochs, n_points_pde, n_points_ic):
        print("Avvio training della PINN...")
        start_time = time.time()
        
        # Liste per memorizzare l'andamento delle loss
        history = {
            'epochs': [],
            'total_loss': [],
            'pde_loss': [],
            'ic_loss': [],
            'learning_rate': []
        }

        # Warm-up iniziale focalizzato sulla IC
        print("Warm-up sulla condizione iniziale...")
        for warmup_epoch in range(500):
            self.optimizer.zero_grad()
            x_ic = torch.rand(n_points_ic, 1, device=self.device)
            y_ic = torch.rand(n_points_ic, 1, device=self.device)
            t_ic = torch.zeros(n_points_ic, 1, device=self.device)
            u_ic_pred = self.model(x_ic, y_ic, t_ic)
            u_ic_true = torch.zeros_like(u_ic_pred)
            u_ic_true[(x_ic >= 0.9) & (y_ic >= 0.9)] = 1.0
            loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)
            loss_ic.backward()
            self.optimizer.step()
        print("Fine warm-up.")

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
            total_loss = loss_pde + 20 * loss_ic  # Ridotto ulteriormente il peso della IC
            
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Salva i valori delle loss ogni 100 epoche
            if (epoch + 1) % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                history['epochs'].append(epoch + 1)
                history['total_loss'].append(total_loss.item())
                history['pde_loss'].append(loss_pde.item())
                history['ic_loss'].append(loss_ic.item())
                history['learning_rate'].append(lr)
                
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss.item():.4e}, "
                      f"Loss PDE: {loss_pde.item():.4e}, Loss IC: {loss_ic.item():.4e}, "
                      f"LR: {lr:.4e}")

        end_time = time.time()
        print(f"Training completato in {end_time - start_time:.2f} secondi.")
        
        return history
