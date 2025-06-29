# src/deepritz_solver.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time

class CustomActivation(nn.Module):
    def __init__(self, sharpness=1.0, shift=0.0):
        super(CustomActivation, self).__init__()
        self.sharpness = nn.Parameter(torch.tensor(sharpness))
        self.shift = nn.Parameter(torch.tensor(shift))

    def forward(self, x):
        return 0.5 * (1.0 - torch.tanh(x))

class DeepRitzSolver(nn.Module):
    """
    Risolve l'equazione del monodominio usando il metodo DeepRitz.
    Utilizza la formulazione variazionale (forma debole) invece della forma forte della PDE.
    """
    def __init__(self, device, sigma_h, a, fr, ft, fd, layers=[3, 64, 64, 64, 1]):
        super(DeepRitzSolver, self).__init__()
        
        self.device = device if device is not None else torch.device('cpu')
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.activations.append(CustomActivation())
        
        # Aggiunta opzioni per normalizzazione dell'input
        self.input_normalization = True
        self.x_mean = 0.5
        self.x_std = 0.5
        self.y_mean = 0.5
        self.y_std = 0.5
        self.t_mean = 17.5  # T/2
        self.t_std = 17.5   # T/2
        
        self.init_weights()
        
        # Parametri fisici (possono essere aggiornati)
        self.sigma_h = sigma_h
        self.a = a
        self.fr = fr
        self.ft = ft
        self.fd = fd
        
        # Pesi adattivi per la loss
        self.pde_weight = 1.0      # Peso per il residuo PDE
        self.ic_weight = 100.0     # Dare più importanza alla condizione iniziale
        self.bc_weight = 10.0      # Peso per le condizioni al contorno

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, y, t):
        x, y, t = self.normalize_input(x, y, t)
        inputs = torch.cat([x, y, t], dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            inputs = self.activations[i](layer(inputs))
        output = self.layers[-1](inputs)
        return output

    def normalize_input(self, x, y, t):
        if self.input_normalization:
            x = (x - self.x_mean) / self.x_std
            y = (y - self.y_mean) / self.y_std
            t = (t - self.t_mean) / self.t_std
        return x, y, t

    def get_energy_functional_loss(self, x, y, t):
        """
        Calcola la loss basata sul residuo della PDE usando forma debole.
        Per il monodomain: ∂u/∂t - ∇·(σ∇u) - a*u*(u-fr)*(u-ft)*(u-fd) = 0
        """
        u = self(x, y, t)
        
        # Calcolo delle derivate con autograd
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Seconda derivata spaziale (Laplaciano)
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        # Termine di diffusione
        diffusion = self.sigma_h * (u_xx + u_yy)
        
        # Termine di reazione (FitzHugh-Nagumo)
        reaction = self.a * u * (u - self.fr) * (u - self.ft) * (u - self.fd)
        
        # Residuo della PDE: ∂u/∂t - diffusion - reaction = 0
        pde_residual = u_t - diffusion - reaction
        
        # Loss come MSE del residuo
        return torch.mean(pde_residual**2)

    def get_initial_condition_loss(self, x, y, t0=0.0):
        """
        Loss per la condizione iniziale: u(x,y,0) = u0(x,y)
        """
        t_ic = torch.full_like(x, t0)
        u_pred = self(x, y, t_ic)
        
        # Condizione iniziale: impulso nel corner (0.9, 0.9)
        u_true = torch.zeros_like(u_pred)
        mask = (x >= 0.9) & (y >= 0.9)
        u_true[mask] = 1.0
        
        return torch.mean((u_pred - u_true)**2)

    def get_boundary_condition_loss(self, x_bc, y_bc, t_bc):
        """
        Loss per le condizioni al contorno (Neumann omogenee: ∂u/∂n = 0)
        """
        u = self(x_bc, y_bc, t_bc)
        
        # Calcolo delle derivate normali al contorno
        u_x = torch.autograd.grad(u, x_bc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_bc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Per bordi verticali (x=0 e x=1): ∂u/∂x = 0
        # Per bordi orizzontali (y=0 e y=1): ∂u/∂y = 0
        bc_loss = torch.mean(u_x**2 + u_y**2)
        
        return bc_loss

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
        print("Calcolo della soluzione DeepRitz...")
        
        # Griglia spaziale
        x = np.linspace(0, 1, nvx)
        y = np.linspace(0, 1, nvy)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Tempi di output
        times = np.linspace(0, T, num_frames)
        solutions = []
        
        self.eval()
        with torch.no_grad():
            for i, t in enumerate(times):
                print(f"Frame {i+1}/{num_frames} (t={t:.2f})")
                
                # Converte in tensori
                X_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=self.device, requires_grad=False)
                Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=self.device, requires_grad=False)
                T_flat = torch.full_like(X_flat, t)
                
                # Predizione
                u_pred = self(X_flat.unsqueeze(1), Y_flat.unsqueeze(1), T_flat.unsqueeze(1))
                u_solution = u_pred.squeeze().cpu().numpy().reshape(nvx, nvy)
                
                solutions.append(u_solution)
        
        return {
            'x': x,
            'y': y, 
            'times': times,
            'solutions': solutions
        }

class DeepRitzTrainer:
    """Classe per l'addestramento del modello DeepRitz."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.loss_history = []
        
    def generate_training_data(self, n_domain=2000, n_boundary=400, n_initial=400, T=35.0):
        """
        Genera i dati per l'addestramento.
        
        Args:
            n_domain (int): Numero di punti nel dominio interno.
            n_boundary (int): Numero di punti sui bordi.
            n_initial (int): Numero di punti per la condizione iniziale.
            T (float): Tempo finale.
            
        Returns:
            dict: Dizionario con i dati di addestramento.
        """
        # Punti nel dominio interno
        x_domain = torch.rand(n_domain, 1, device=self.device, requires_grad=True)
        y_domain = torch.rand(n_domain, 1, device=self.device, requires_grad=True)
        t_domain = torch.rand(n_domain, 1, device=self.device, requires_grad=True) * T
        
        # Punti sui bordi
        # Bordo sinistro (x=0)
        n_side = n_boundary // 4
        x_left = torch.zeros(n_side, 1, device=self.device, requires_grad=True)
        y_left = torch.rand(n_side, 1, device=self.device, requires_grad=True)
        t_left = torch.rand(n_side, 1, device=self.device, requires_grad=True) * T
        
        # Bordo destro (x=1)
        x_right = torch.ones(n_side, 1, device=self.device, requires_grad=True)
        y_right = torch.rand(n_side, 1, device=self.device, requires_grad=True)
        t_right = torch.rand(n_side, 1, device=self.device, requires_grad=True) * T
        
        # Bordo inferiore (y=0)
        x_bottom = torch.rand(n_side, 1, device=self.device, requires_grad=True)
        y_bottom = torch.zeros(n_side, 1, device=self.device, requires_grad=True)
        t_bottom = torch.rand(n_side, 1, device=self.device, requires_grad=True) * T
        
        # Bordo superiore (y=1)
        x_top = torch.rand(n_side, 1, device=self.device, requires_grad=True)
        y_top = torch.ones(n_side, 1, device=self.device, requires_grad=True)
        t_top = torch.rand(n_side, 1, device=self.device, requires_grad=True) * T
        
        # Combina i punti sui bordi
        x_boundary = torch.cat([x_left, x_right, x_bottom, x_top])
        y_boundary = torch.cat([y_left, y_right, y_bottom, y_top])
        t_boundary = torch.cat([t_left, t_right, t_bottom, t_top])
        
        # Punti per la condizione iniziale
        x_initial = torch.rand(n_initial, 1, device=self.device, requires_grad=True)
        y_initial = torch.rand(n_initial, 1, device=self.device, requires_grad=True)
        
        return {
            'domain': (x_domain, y_domain, t_domain),
            'boundary': (x_boundary, y_boundary, t_boundary),
            'initial': (x_initial, y_initial)
        }
    
    def train(self, epochs=10000, lr=1e-3, n_domain=2000, n_boundary=400, n_initial=400, T=35.0):
        """
        Addestra il modello DeepRitz.
        
        Args:
            epochs (int): Numero di epoche di addestramento.
            lr (float): Learning rate.
            n_domain (int): Numero di punti nel dominio.
            n_boundary (int): Numero di punti sui bordi.
            n_initial (int): Numero di punti per la condizione iniziale.
            T (float): Tempo finale.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
        
        print(f"Inizio addestramento DeepRitz per {epochs} epoche...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Genera nuovi dati ad ogni epoca per migliorare la generalizzazione
            data = self.generate_training_data(n_domain, n_boundary, n_initial, T)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Calcola le loss
            pde_loss = self.model.get_energy_functional_loss(*data['domain'])
            ic_loss = self.model.get_initial_condition_loss(*data['initial'])
            bc_loss = self.model.get_boundary_condition_loss(*data['boundary'])
            
            # Loss totale
            total_loss = (self.model.pde_weight * pde_loss + 
                         self.model.ic_weight * ic_loss + 
                         self.model.bc_weight * bc_loss)
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Salva la loss
            self.loss_history.append(total_loss.item())
            
            # Print progress
            if epoch % 500 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoca {epoch}/{epochs} - "
                      f"Loss totale: {total_loss.item():.6f} - "
                      f"PDE: {pde_loss.item():.6f} - "
                      f"IC: {ic_loss.item():.6f} - "
                      f"BC: {bc_loss.item():.6f} - "
                      f"LR: {current_lr:.2e}")
        
        end_time = time.time()
        print(f"Addestramento completato in {end_time - start_time:.2f} secondi")
        
    def save_model(self, model_dir):
        """Salva il modello e la storia dell'addestramento."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Salva i pesi del modello
        model_path = os.path.join(model_dir, 'model_weights.pth')
        torch.save(self.model.state_dict(), model_path)
        
        # Salva la storia della loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('Epoca')
        plt.ylabel('Loss')
        plt.title('Storia dell\'addestramento DeepRitz')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Salva info di addestramento
        with open(os.path.join(model_dir, 'training_info.txt'), 'w') as f:
            f.write(f"DeepRitz Model Training Info\n")
            f.write(f"============================\n")
            f.write(f"Total epochs: {len(self.loss_history)}\n")
            f.write(f"Final loss: {self.loss_history[-1]:.6f}\n")
            f.write(f"Min loss: {min(self.loss_history):.6f}\n")
            f.write(f"Model architecture: {[layer.in_features if hasattr(layer, 'in_features') else 'Activation' for layer in self.model.layers]}\n")
            f.write(f"Physics parameters:\n")
            f.write(f"  sigma_h: {self.model.sigma_h}\n")
            f.write(f"  a: {self.model.a}\n")
            f.write(f"  fr: {self.model.fr}\n")
            f.write(f"  ft: {self.model.ft}\n")
            f.write(f"  fd: {self.model.fd}\n")
        
        print(f"Modello salvato in {model_dir}")
