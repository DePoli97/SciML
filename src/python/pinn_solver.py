# src/pinn_solver.py
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
        self.pde_weight = 1.0
        self.ic_weight = 10.0  # Dare più importanza alla condizione iniziale

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

    def adapt_weights(self, epoch, pde_loss, ic_loss):
        """Adatta i pesi delle perdite in base all'andamento del training"""
        if epoch == 0:
            return
        
        # Strategia adattiva: se una componente della loss è molto più grande dell'altra, bilanciale
        ratio = pde_loss / (ic_loss + 1e-8)
        
        if ratio > 10.0:
            self.pde_weight *= 0.95
            self.ic_weight *= 1.05
        elif ratio < 0.1:
            self.pde_weight *= 1.05
            self.ic_weight *= 0.95
        
        # Limiti per evitare valori estremi
        self.pde_weight = max(min(self.pde_weight, 100.0), 0.01)
        self.ic_weight = max(min(self.ic_weight, 100.0), 0.01)

class PINNTrainer:
    def __init__(self, model, learning_rate=1e-3, device=None, T=35.0):
        self.device = device if device is not None else torch.device('cpu')
        self.model = model.to(self.device)
        self.T = T
        
        # Utilizzo di un ottimizzatore più sofisticato: Adam con parametri ottimizzati
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            betas=(0.9, 0.999),  # Valori predefiniti ma esplicitati per chiarezza
            eps=1e-8,
            weight_decay=1e-6    # Regolarizzazione L2 per evitare overfitting
        )
        
        # Scheduler più aggressivo per ridurre il learning rate quando la loss si stabilizza
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=300, factor=0.2, min_lr=1e-7, verbose=True
        )
        
        # Flag per il curriculum learning
        self.use_curriculum = True

    def train(self, n_epochs, n_points_pde, n_points_ic):
        print("Avvio training della PINN...")
        start_time = time.time()
        
        # Liste per memorizzare l'andamento delle loss
        history = {
            'epochs': [],
            'total_loss': [],
            'pde_loss': [],
            'ic_loss': [],
            'learning_rate': [],
            'pde_weight': [],
            'ic_weight': []
        }

        # Warm-up iniziale focalizzato sulla IC
        print("Warm-up sulla condizione iniziale...")
        n_warmup = 1000  # Aumentato per una migliore convergenza iniziale
        
        # Usa punti stratificati con più densità nelle regioni di interesse
        x_ic_corner = torch.cat([
            torch.rand(n_points_ic // 2, 1, device=self.device) * 0.1 + 0.9,  # Punti nell'angolo (0.9-1.0, 0.9-1.0)
            torch.rand(n_points_ic // 2, 1, device=self.device)               # Punti nel resto del dominio
        ], dim=0)
        
        y_ic_corner = torch.cat([
            torch.rand(n_points_ic // 2, 1, device=self.device) * 0.1 + 0.9,  # Punti nell'angolo (0.9-1.0, 0.9-1.0)
            torch.rand(n_points_ic // 2, 1, device=self.device)               # Punti nel resto del dominio
        ], dim=0)
        
        t_ic = torch.zeros(n_points_ic, 1, device=self.device)
        u_ic_true = torch.zeros(n_points_ic, 1, device=self.device)
        u_ic_true[(x_ic_corner >= 0.9) & (y_ic_corner >= 0.9)] = 1.0
        
        # Warm-up con scheduler di learning rate
        warmup_optimizer = optim.Adam(self.model.parameters(), lr=5e-3)
        for warmup_epoch in range(n_warmup):
            warmup_optimizer.zero_grad()
            u_ic_pred = self.model(x_ic_corner, y_ic_corner, t_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)
            loss_ic.backward()
            warmup_optimizer.step()
            
            if (warmup_epoch + 1) % 200 == 0:
                print(f"  Warm-up epoch {warmup_epoch+1}/{n_warmup}, IC Loss: {loss_ic.item():.6f}")
        
        print("Fine warm-up.")

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            # 1. Campionamento più intelligente per i punti PDE
            # Utilizzo punti casuali con maggiore densità sul fronte di propagazione
            if self.use_curriculum and epoch < n_epochs // 3:
                # Fase iniziale: concentrarsi su tempi brevi per catturare meglio la dinamica iniziale
                t_max = self.T * 0.3
            elif self.use_curriculum and epoch < 2 * n_epochs // 3:
                # Fase intermedia: espandersi gradualmente
                t_max = self.T * 0.7
            else:
                # Fase finale: intero dominio temporale
                t_max = self.T
            
            # Campionamento stratificato per x, y
            x_pde = torch.rand(n_points_pde, 1, device=self.device, requires_grad=True)
            y_pde = torch.rand(n_points_pde, 1, device=self.device, requires_grad=True)
            
            # Campionamento temporale con più punti nelle fasi iniziali
            t_distribution = torch.rand(n_points_pde, 1, device=self.device) ** 1.5  # Esponente < 1 aumenta la densità vicino a t=0
            t_pde = t_distribution * t_max
            t_pde.requires_grad = True
            
            # Calcolo della loss fisica
            loss_pde = self.model.get_physics_loss(x_pde, y_pde, t_pde)
            
            # 2. Campionamento stratificato per la condizione iniziale (come nel warm-up)
            x_ic = torch.cat([
                torch.rand(n_points_ic // 2, 1, device=self.device) * 0.1 + 0.9,  # Angolo (0.9-1.0)
                torch.rand(n_points_ic // 2, 1, device=self.device)               # Resto del dominio
            ], dim=0)
            
            y_ic = torch.cat([
                torch.rand(n_points_ic // 2, 1, device=self.device) * 0.1 + 0.9,  # Angolo (0.9-1.0)
                torch.rand(n_points_ic // 2, 1, device=self.device)               # Resto del dominio
            ], dim=0)
            
            t_ic = torch.zeros(n_points_ic, 1, device=self.device)
            u_ic_pred = self.model(x_ic, y_ic, t_ic)
            
            u_ic_true = torch.zeros_like(u_ic_pred)
            u_ic_true[(x_ic >= 0.9) & (y_ic >= 0.9)] = 1.0
            loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)
            
            # 3. Utilizzo di pesi adattivi per bilanciare le componenti della loss
            total_loss = self.model.pde_weight * loss_pde + self.model.ic_weight * loss_ic
            
            # Backpropagation e ottimizzazione
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Clipping del gradiente per stabilità
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # Adatta i pesi delle loss
            self.model.adapt_weights(epoch, loss_pde.item(), loss_ic.item())
            
            # Salva i valori delle loss ogni 100 epoche
            if (epoch + 1) % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                history['epochs'].append(epoch + 1)
                history['total_loss'].append(total_loss.item())
                history['pde_loss'].append(loss_pde.item())
                history['ic_loss'].append(loss_ic.item())
                history['learning_rate'].append(lr)
                history['pde_weight'].append(self.model.pde_weight)
                history['ic_weight'].append(self.model.ic_weight)
                
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss.item():.4e}, "
                      f"Loss PDE: {loss_pde.item():.4e}, Loss IC: {loss_ic.item():.4e}, "
                      f"LR: {lr:.4e}")
                
            # Early stopping basato sulla loss
            if epoch > 1000 and epoch % 500 == 0:
                recent_losses = history['total_loss'][-5:]
                if len(recent_losses) >= 5:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    std_loss = np.std(recent_losses)
                    if std_loss / avg_loss < 0.001:  # Variazione relativa molto piccola
                        print(f"Early stopping attivato all'epoca {epoch+1}: la loss si è stabilizzata.")
                        break

        end_time = time.time()
        print(f"Training completato in {end_time - start_time:.2f} secondi.")
        
        return history

    def plot_loss(self, history, output_dir=None):
        """
        Crea un plot più dettagliato dell'andamento delle loss durante il training
        """
        plt.figure(figsize=(18, 10))
        
        # Subplot per le loss
        plt.subplot(2, 2, 1)
        plt.semilogy(history['epochs'], history['total_loss'], 'b-', label='Loss Totale')
        plt.semilogy(history['epochs'], history['pde_loss'], 'r-', label='Loss PDE')
        plt.semilogy(history['epochs'], history['ic_loss'], 'g-', label='Loss IC')
        plt.xlabel('Epoche')
        plt.ylabel('Loss (log scale)')
        plt.legend()
        plt.title('Andamento delle Loss')
        plt.grid(True, which="both", ls="--")
        
        # Subplot per il learning rate
        plt.subplot(2, 2, 2)
        plt.semilogy(history['epochs'], history['learning_rate'], 'k-')
        plt.xlabel('Epoche')
        plt.ylabel('Learning Rate (log scale)')
        plt.title('Learning Rate')
        plt.grid(True, which="both", ls="--")
        
        # Subplot per i pesi delle loss
        if 'pde_weight' in history and 'ic_weight' in history:
            plt.subplot(2, 2, 3)
            plt.plot(history['epochs'], history['pde_weight'], 'r-', label='Peso PDE')
            plt.plot(history['epochs'], history['ic_weight'], 'g-', label='Peso IC')
            plt.xlabel('Epoche')
            plt.ylabel('Valore del peso')
            plt.title('Pesi delle Loss')
            plt.legend()
            plt.grid(True, ls="--")
        
        # Subplot per il rapporto tra le loss
        plt.subplot(2, 2, 4)
        loss_ratio = [pde / ic if ic > 0 else pde for pde, ic in zip(history['pde_loss'], history['ic_loss'])]
        plt.semilogy(history['epochs'], loss_ratio, 'm-')
        plt.xlabel('Epoche')
        plt.ylabel('Rapporto PDE/IC (log scale)')
        plt.title('Rapporto tra Loss PDE e IC')
        plt.grid(True, which="both", ls="--")
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300)
            print(f"Plot della loss salvato in {os.path.join(output_dir, 'training_loss.png')}")
            
        plt.show()
