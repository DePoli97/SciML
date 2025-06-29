import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class UNetBlock(nn.Module):
    """Blocco base per l'architettura U-Net."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class CNNSolver(nn.Module):
    """
    Risolve l'equazione del monodominio usando un modello CNN.
    Questo modello predice la soluzione al tempo t+dt dato lo stato al tempo t.
    """
    def __init__(self, device=None, prediction_steps=1, latent_dim=32, nvx=101, nvy=101):
        super(CNNSolver, self).__init__()
        
        self.device = device if device is not None else torch.device('cpu')
        self.prediction_steps = prediction_steps  # Quanti passi di tempo predire in avanti
        self.latent_dim = latent_dim
        self.nvx = nvx  # Numero di punti nella direzione x
        self.nvy = nvy  # Numero di punti nella direzione y
        
        # Encoder (estrae features dalle soluzioni)
        self.encoder = nn.Sequential(
            UNetBlock(1, 16),
            nn.MaxPool2d(2),  # Riduzione dimensionalità 2x
            UNetBlock(16, 32),
            nn.MaxPool2d(2),  # Riduzione dimensionalità 4x
            UNetBlock(32, latent_dim),
        )
        
        # Decoder (ricostruisce la soluzione dallo spazio latente con dimensioni esatte)
        self.decoder = nn.Sequential(
            UNetBlock(latent_dim, 32),
            nn.Upsample(size=(26, 26), mode='bilinear', align_corners=True),
            UNetBlock(32, 16),
            nn.Upsample(size=(nvx//2, nvy//2), mode='bilinear', align_corners=True),
            UNetBlock(16, 8),
            nn.Upsample(size=(nvx, nvy), mode='bilinear', align_corners=True),  # Dimensioni esatte dell'output
            nn.Conv2d(8, 1, kernel_size=1)  # Mappa finale a 1 canale
        )
        
        # Proiezione dei parametri fisici nello spazio latente
        self.params_projection = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        
        # Modulo di previsione temporale (evolve lo stato latente nel tempo)
        self.time_evolution = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def _spatial_params_integration(self, latent, sigma_h):
        """Integra i parametri fisici nello spazio latente."""
        # sigma_h può essere diverso per ogni simulazione (low, normal, high)
        batch_size = latent.shape[0]
        
        # Gestisci sia il caso in cui sigma_h è un tensore che il caso in cui è uno scalare
        if isinstance(sigma_h, torch.Tensor):
            # Se sigma_h è già un tensore (durante il training), usalo direttamente
            # Assicurati che abbia la forma corretta (batch_size, 1)
            if len(sigma_h.shape) == 1:
                sigma_h_tensor = sigma_h.view(-1, 1).to(self.device)
            else:
                sigma_h_tensor = sigma_h.to(self.device)
        else:
            # Se sigma_h è uno scalare (durante l'inferenza), creane un tensore
            sigma_h_tensor = torch.full((batch_size, 1), sigma_h, dtype=torch.float32).to(self.device)
        
        # Parametri aggiuntivi che potrebbero essere inclusi in futuro
        zeros = torch.zeros_like(sigma_h_tensor).to(self.device)
        params = torch.cat([sigma_h_tensor, zeros, zeros, zeros], dim=1)
        
        # Proiezione e reshape per essere compatibile con il tensore latente
        params_projected = self.params_projection(params)
        params_spatial = params_projected.view(batch_size, self.latent_dim, 1, 1)
        
        return latent + params_spatial
        
    def forward(self, x, sigma_h, steps=None):
        """
        Predice l'evoluzione della soluzione per un numero specificato di passi.
        
        Args:
            x: Tensore di input [batch, 1, height, width] che rappresenta lo stato attuale
            sigma_h: Coefficiente di diffusione
            steps: Numero di passi temporali da predire (se None, usa prediction_steps)
            
        Returns:
            Lista di tensori, ciascuno rappresentante la soluzione a t+i*dt
        """
        if steps is None:
            steps = self.prediction_steps
            
        batch_size = x.shape[0]
        
        # Estrae lo spazio latente dalla soluzione attuale
        latent = self.encoder(x)
        
        # Integra i parametri fisici
        latent = self._spatial_params_integration(latent, sigma_h)
        
        predictions = []
        current_state = latent
        
        # Evolve lo stato nello spazio latente per il numero di passi richiesto
        for _ in range(steps):
            # Applica l'evoluzione temporale
            next_state = self.time_evolution(current_state)
            
            # Decodifica il nuovo stato
            prediction = self.decoder(next_state)
            
            predictions.append(prediction)
            current_state = next_state
            
        return predictions
    
    def compute_solution(self, T, nvx=None, nvy=None, sigma_h=None, num_frames=100, ic=None):
        """
        Calcola la soluzione numerica per diversi istanti temporali.
        
        Args:
            T (float): Tempo finale della simulazione.
            nvx (int, optional): Numero di punti nella direzione x. Se None, usa il valore dell'istanza.
            nvy (int, optional): Numero di punti nella direzione y. Se None, usa il valore dell'istanza.
            sigma_h (float, optional): Coefficiente di diffusione. Se None, usa un valore predefinito.
            num_frames (int, optional): Numero di frame temporali da calcolare. Default 100.
            ic (ndarray, optional): Condizione iniziale. Se None, viene usata quella standard.
            
        Returns:
            dict: Un dizionario contenente:
                - 'x': coordinate x della griglia.
                - 'y': coordinate y della griglia.
                - 'times': array dei tempi simulati.
                - 'solutions': lista di soluzioni per ogni istante temporale.
        """
        print("Calcolo della soluzione CNN...")
        
        # Usa i valori dell'istanza se non specificati
        nvx = nvx if nvx is not None else self.nvx
        nvy = nvy if nvy is not None else self.nvy
        sigma_h = sigma_h if sigma_h is not None else 9.5298e-4  # Valore predefinito di sigma_h
        
        # Prepara la griglia
        x = np.linspace(0, 1, nvx)
        y = np.linspace(0, 1, nvy)
        
        # Se non è fornita la condizione iniziale, usiamo quella standard
        if ic is None:
            X, Y = np.meshgrid(x, y, indexing='ij')
            ic = np.zeros((nvx, nvy))
            ic[(X >= 0.9) & (Y >= 0.9)] = 1.0
        
        # Prepara la condizione iniziale come tensore PyTorch
        u_current = torch.tensor(ic, dtype=torch.float32).view(1, 1, nvx, nvy).to(self.device)
        
        # Calcola la soluzione per diversi istanti di tempo
        times = np.linspace(0, T, num_frames)
        solutions = [ic.copy()]  # Inizia con la condizione iniziale
        
        # Predice la soluzione avanzando di prediction_steps passi alla volta
        with torch.no_grad():
            for i in range(1, num_frames):
                # Predice il prossimo stato
                predictions = self(u_current, sigma_h, steps=1)
                
                # Aggiorna lo stato corrente con l'ultima predizione
                u_current = predictions[-1]
                
                # Salva la soluzione
                u_grid = u_current.squeeze().cpu().numpy()
                solutions.append(u_grid)
                
                if (i + 1) % 10 == 0:
                    print(f"  Istante {i+1}/{num_frames} calcolato (t={times[i]:.2f}).")
        
        return {
            'x': x,
            'y': y,
            'times': times,
            'solutions': solutions
        }


class CNNTrainer:
    """Trainer per il modello CNN."""
    def __init__(self, model, learning_rate=1e-4, device=None):
        self.device = device if device is not None else torch.device('cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5, min_lr=1e-6, verbose=True
        )
    
    def generate_training_data_from_fem(self, fem_solver, sigmas, T, dt, num_samples=100, seed=42):
        """
        Genera dati di training dal solver FEM.
        
        Args:
            fem_solver: Istanza di FEMSolver
            sigmas: Lista di coefficienti di diffusione da usare
            T: Tempo finale della simulazione
            dt: Passo temporale
            num_samples: Numero di coppie (stato, evoluzione) da generare
            
        Returns:
            inputs: Tensore degli stati iniziali [num_samples, channels, height, width]
            targets: Tensore degli stati futuri [num_samples, channels, height, width]
            sigma_values: Tensore dei coefficienti di diffusione [num_samples]
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        nvx, nvy = fem_solver.nvx, fem_solver.nvy
        inputs = []
        targets = []
        sigma_values = []
        
        print("Generazione dati di training dal solver FEM...")
        for sigma in sigmas:
            # Configura il solver FEM con questo sigma
            fem_solver.sigma_h = sigma
            
            # Simula con FEM
            solution_data = fem_solver.compute_solution(T=T, dt=dt, sigma_d_factor=1.0, num_frames=50)
            solutions = solution_data['solutions']
            
            # Campiona coppie di stati (t, t+1) dalla simulazione
            n_frames = len(solutions)
            samples_per_sigma = num_samples // len(sigmas)
            
            for _ in range(samples_per_sigma):
                # Scegli un frame casuale (escludendo l'ultimo)
                idx = np.random.randint(0, n_frames - self.model.prediction_steps)
                
                # Usa questo frame come input e quello successivo come target
                input_state = solutions[idx]
                target_states = [solutions[idx + i + 1] for i in range(self.model.prediction_steps)]
                
                # Aggiungi ai dataset
                inputs.append(torch.tensor(input_state, dtype=torch.float32).view(1, nvx, nvy))
                targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                sigma_values.append(sigma)
        
        return inputs, targets, sigma_values
    
    def train(self, train_loader, valid_loader=None, num_epochs=100):
        """
        Addestra il modello CNN sui dati forniti.
        
        Args:
            train_loader: DataLoader con i dati di training
            valid_loader: DataLoader con i dati di validazione
            num_epochs: Numero di epoche di training
            
        Returns:
            dict: Storia del training
        """
        print("Avvio training del modello CNN...")
        start_time = time.time()
        
        history = {
            'train_loss': [],
            'valid_loss': []
        }
        
        best_valid_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for inputs, targets_list, sigmas in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs = inputs.to(self.device)
                sigmas = sigmas.to(self.device)
                
                # Forward pass
                predictions = self.model(inputs, sigmas)
                
                # Calcola la loss
                loss = 0.0
                for i, pred in enumerate(predictions):
                    # Assicurati che i target abbiano la stessa dimensione del batch delle predizioni
                    if i < len(targets_list):
                        # Prendi il target corrispondente al passo temporale i
                        target = targets_list[i].to(self.device)
                        
                        # Assicurati che target abbia la stessa dimensione di batch delle predizioni
                        if target.shape[0] != pred.shape[0]:
                            # Se il batch size è diverso, prendi solo i primi elementi del target
                            # che corrispondono alla dimensione del batch delle predizioni
                            target = target[:pred.shape[0]]
                        
                        loss += self.criterion(pred, target)
                
                # Backward pass e ottimizzazione
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            history['train_loss'].append(train_loss)
            
            # Validazione
            if valid_loader is not None:
                valid_loss = self.evaluate(valid_loader)
                history['valid_loss'].append(valid_loss)
                self.scheduler.step(valid_loss)
                
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Valid Loss: {valid_loss:.6f}")
                
                # Salva il modello migliore
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), 'best_cnn_model.pth')
            else:
                self.scheduler.step(train_loss)
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")
        
        end_time = time.time()
        print(f"Training completato in {end_time - start_time:.2f} secondi.")
        
        return history
    
    def evaluate(self, data_loader):
        """Valuta il modello sul dataset fornito."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets_list, sigmas in data_loader:
                inputs = inputs.to(self.device)
                sigmas = sigmas.to(self.device)
                
                # Forward pass
                predictions = self.model(inputs, sigmas)
                
                # Calcola la loss
                loss = 0.0
                for i, pred in enumerate(predictions):
                    # Assicurati che i target abbiano la stessa dimensione del batch delle predizioni
                    if i < len(targets_list):
                        # Prendi il target corrispondente al passo temporale i
                        target = targets_list[i].to(self.device)
                        
                        # Assicurati che target abbia la stessa dimensione di batch delle predizioni
                        if target.shape[0] != pred.shape[0]:
                            # Se il batch size è diverso, prendi solo i primi elementi del target
                            # che corrispondono alla dimensione del batch delle predizioni
                            target = target[:pred.shape[0]]
                        
                        loss += self.criterion(pred, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def plot_loss(self, history, output_dir=None):
        """
        Crea un grafico dell'andamento della loss durante il training
        """
        plt.figure(figsize=(12, 8))
        
        plt.plot(history['train_loss'], 'b-', label='Training Loss')
        if 'valid_loss' in history:
            plt.plot(history['valid_loss'], 'r--', label='Validation Loss')
        
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Andamento della Loss Durante il Training')
        
        if output_dir:
            output_path = os.path.join(output_dir, 'training_loss.png')
            plt.savefig(output_path)
            print(f"Grafico delle loss salvato in: {output_path}")
        
        plt.show()
