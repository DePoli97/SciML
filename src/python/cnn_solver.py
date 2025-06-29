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
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x

class ResUNetBlock(nn.Module):
    """Blocco avanzato con connessione residuale per l'architettura U-Net."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1):
        super(ResUNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Connessione residuale (skip connection)
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        
        # Somma con la connessione residuale
        x += residual
        x = self.relu(x)
        
        return x

class CNNSolver(nn.Module):
    """
    Risolve l'equazione del monodominio usando un modello CNN avanzato e profondo.
    Questo modello predice la soluzione al tempo t+dt dato lo stato al tempo t.
    
    Architettura migliorata con connessioni residuali, aumentata profondità, incrementato spazio latente,
    e regolarizzazione tramite dropout per prevenire l'overfitting.
    """
    def __init__(self, device=None, prediction_steps=1, latent_dim=128, nvx=101, nvy=101, dropout_rate=0.2):
        super(CNNSolver, self).__init__()
        
        self.device = device if device is not None else torch.device('cpu')
        self.prediction_steps = prediction_steps  # Quanti passi di tempo predire in avanti
        self.latent_dim = latent_dim
        self.nvx = nvx  # Numero di punti nella direzione x
        self.nvy = nvy  # Numero di punti nella direzione y
        
        # Encoder molto più profondo con più livelli di estrazione caratteristiche
        self.encoder = nn.Sequential(
            # Primo livello - estrazione caratteristiche iniziali
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResUNetBlock(16, 32, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),  # Riduzione dimensionalità 2x
            
            # Secondo livello - media risoluzione
            ResUNetBlock(32, 64, dropout_rate=dropout_rate),
            ResUNetBlock(64, 64, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),  # Riduzione dimensionalità 4x
            
            # Terzo livello - bassa risoluzione, alta semantica
            ResUNetBlock(64, 128, dropout_rate=dropout_rate),
            ResUNetBlock(128, 128, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),  # Riduzione dimensionalità 8x
            
            # Quarto livello - ulteriore compressione
            ResUNetBlock(128, 256, dropout_rate=dropout_rate),
            nn.MaxPool2d(2),  # Riduzione dimensionalità 16x
            
            # Bottleneck con attenzione alla preservazione dell'informazione
            ResUNetBlock(256, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate),
        )
        
        # Decoder migliorato con percorso più graduale e più livelli di trasformazione
        self.decoder = nn.Sequential(
            # Espansione iniziale dal bottleneck profondo
            ResUNetBlock(latent_dim, 256, dropout_rate=dropout_rate),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Primo livello di upsampling - molto bassa risoluzione
            ResUNetBlock(256, 128, dropout_rate=dropout_rate),
            ResUNetBlock(128, 128, dropout_rate=dropout_rate),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Secondo livello di upsampling - bassa risoluzione
            ResUNetBlock(128, 64, dropout_rate=dropout_rate),
            ResUNetBlock(64, 64, dropout_rate=dropout_rate),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Terzo livello di upsampling - media risoluzione
            ResUNetBlock(64, 32, dropout_rate=dropout_rate),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Livello di output - alta risoluzione
            ResUNetBlock(32, 16, dropout_rate=dropout_rate),
            
            # Adatta le dimensioni esattamente all'output desiderato
            nn.Upsample(size=(nvx, nvy), mode='bilinear', align_corners=True),
            
            # Proiezione finale a 1 canale con attenzione alla stabilità numerica
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()  # Limita l'output tra 0 e 1 per stabilità
        )
        
        # Proiezione dei parametri fisici nello spazio latente (architettura più complessa)
        self.params_projection = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, latent_dim)
        )
        
        # Modulo di previsione temporale molto più profondo con molti blocchi residuali
        # per catturare meglio l'evoluzione temporale
        self.time_evolution = nn.Sequential(
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate)
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
    """
    Trainer avanzato per il modello CNN con strategie di ottimizzazione migliorate.
    
    Caratteristiche:
    - Ottimizzatore AdamW con weight decay adattivo
    - Loss ibrida MSE+MAE con pesi configurabili
    - Learning rate scheduler multi-stadio (ReduceLROnPlateau + CosineAnnealingWarmRestarts)
    - Gradient clipping per prevenire esplosione dei gradienti
    - Supporto per early stopping
    - Aumento/diminuzione progressivo del learning rate (warmup/cooldown)
    """
    def __init__(self, model, learning_rate=5e-4, device=None, weight_decay=2e-5):
        self.device = device if device is not None else torch.device('cpu')
        self.model = model.to(self.device)
        
        # Ottimizzatore AdamW più aggressivo con weight decay ottimizzato
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,  # Regolarizzazione L2 migliorata
            betas=(0.9, 0.999),  # Valori ottimizzati per convergenza stabile
            eps=1e-8
        )
        
        # Funzioni di loss combinate per migliore stabilità e convergenza
        self.criterion = nn.MSELoss()  # Sensibilità ai grandi errori
        self.mae_criterion = nn.L1Loss()  # Sensibilità agli errori piccoli
        
        # Learning rate scheduler principale - riduce il LR quando la loss si stabilizza
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=15, factor=0.5, min_lr=1e-7, verbose=True
        )
        
        # Scheduler secondario - introduce oscillazioni nel LR per uscire da minimi locali
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=30, T_mult=2, eta_min=1e-7
        )
        
        # Tracciamento del best model per early stopping
        self.best_loss = float('inf')
        self.best_model_state = None
    
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
    
    def train(self, train_loader, valid_loader=None, num_epochs=300, mse_weight=0.6, mae_weight=0.4, early_stop_patience=40):
        """
        Addestra il modello CNN sui dati forniti con strategia di training avanzata.
        
        Args:
            train_loader: DataLoader con i dati di training
            valid_loader: DataLoader con i dati di validazione
            num_epochs: Numero di epoche di training (default 300 per convergenza profonda)
            mse_weight: Peso della MSE loss nella combinazione
            mae_weight: Peso della MAE loss nella combinazione
            early_stop_patience: Numero di epoche di attesa prima di early stopping
            
        Returns:
            dict: Storia del training
        """
        print(f"Avvio training avanzato del modello CNN con {num_epochs} epoche...")
        start_time = time.time()
        
        history = {
            'train_loss': [],
            'valid_loss': [],
            'learning_rate': []
        }
        
        best_valid_loss = float('inf')
        early_stop_counter = 0
        
        # Learning rate warmup - inizia con un LR basso e aumenta gradualmente
        initial_lr = self.optimizer.param_groups[0]['lr']
        warmup_epochs = min(10, num_epochs // 10)  # 10 epoche o 10% del totale
        
        for epoch in range(num_epochs):
            # Learning rate warmup
            if epoch < warmup_epochs:
                # Aumenta linearmente il LR da 10% a 100%
                lr_scale = 0.1 + 0.9 * epoch / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = initial_lr * lr_scale
            
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            # Progress bar con tqdm
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for inputs, targets_list, sigmas in progress_bar:
                inputs = inputs.to(self.device)
                sigmas = sigmas.to(self.device)
                
                # Forward pass
                predictions = self.model(inputs, sigmas)
                
                # Calcola la loss ibrida (MSE + MAE) con ponderazione temporale
                batch_loss = 0.0
                predictions_count = 0
                
                for i, pred in enumerate(predictions):
                    if i < len(targets_list):
                        # Prendi il target corrispondente al passo temporale i
                        target = targets_list[i].to(self.device)
                        
                        # Gestione batch size mismatch
                        if target.shape[0] != pred.shape[0]:
                            target = target[:pred.shape[0]]
                        
                        # Loss ibrida MSE + MAE
                        mse_loss = self.criterion(pred, target)
                        mae_loss = self.mae_criterion(pred, target)
                        
                        # Schema di ponderazione temporale migliorato
                        # Dà più peso alle previsioni a breve termine (cruciali per la stabilità)
                        # ma mantiene significativa anche la previsione a lungo termine
                        step_weight = 1.0 / (1.0 + i * 0.15)
                        combined_loss = (mse_weight * mse_loss + mae_weight * mae_loss) * step_weight
                        batch_loss += combined_loss
                        predictions_count += 1
                
                # Normalizza per il numero effettivo di predizioni
                if predictions_count > 0:
                    batch_loss /= predictions_count
                
                # Backward pass e ottimizzazione
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping per stabilità numerica (previene esplosione di gradienti)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += batch_loss.item()
                num_batches += 1
                
                # Aggiorna la barra di progresso con la loss corrente
                progress_bar.set_postfix(loss=f"{batch_loss.item():.6f}")
            
            # Applicazione alternata degli scheduler
            if epoch % 5 == 0:  # Cosine annealing ogni 5 epoche
                self.cosine_scheduler.step()
            
            # Salva il learning rate corrente
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            # Normalizza la loss di training
            train_loss /= num_batches
            history['train_loss'].append(train_loss)
            
            # Validazione
            if valid_loader is not None:
                # Valutazione sul validation set
                valid_loss = self.evaluate(valid_loader, mse_weight, mae_weight)
                history['valid_loss'].append(valid_loss)
                
                # Aggiorna lo scheduler principale basato sulla loss di validazione
                self.scheduler.step(valid_loss)
                
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Valid Loss: {valid_loss:.6f}, "
                      f"LR: {current_lr:.8f}")
                
                # Tracciamento del modello migliore e early stopping
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    
                    # Salva il modello migliore sia su file che in memoria
                    torch.save(self.model.state_dict(), 'best_cnn_model.pth')
                    self.best_model_state = self.model.state_dict().copy()
                    
                    early_stop_counter = 0
                    print(f"  Nuovo miglior modello salvato (loss: {best_valid_loss:.6f})")
                else:
                    early_stop_counter += 1
                
                # Early stopping con messaggio dettagliato
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping attivato dopo {epoch+1} epoche. "
                          f"Nessun miglioramento per {early_stop_patience} epoche consecutive.")
                    
                    # Ripristina il miglior modello per l'output finale
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state, weights_only=True)
                        print(f"Ripristinato il miglior modello (loss: {best_valid_loss:.6f})")
                    
                    break
            else:
                # Se non c'è validation set, aggiorna lo scheduler con la loss di training
                self.scheduler.step(train_loss)
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"LR: {current_lr:.8f}")
        
        # Statistiche finali del training
        end_time = time.time()
        training_minutes = (end_time - start_time) / 60
        training_hours = training_minutes / 60
        
        if training_hours >= 1:
            print(f"Training completato in {training_hours:.2f} ore ({training_minutes:.2f} minuti)")
        else:
            print(f"Training completato in {training_minutes:.2f} minuti")
        
        return history
    
    def evaluate(self, data_loader, mse_weight=0.6, mae_weight=0.4):
        """
        Valuta il modello sul dataset fornito usando la stessa funzione di loss del training.
        
        Args:
            data_loader: DataLoader con i dati di validazione
            mse_weight: Peso della MSE loss nella combinazione
            mae_weight: Peso della MAE loss nella combinazione
            
        Returns:
            float: Loss media di validazione
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():  # Disattiva il calcolo dei gradienti per la validazione
            for inputs, targets_list, sigmas in data_loader:
                inputs = inputs.to(self.device)
                sigmas = sigmas.to(self.device)
                
                # Forward pass
                predictions = self.model(inputs, sigmas)
                
                # Calcola la loss con lo stesso schema del training per consistenza
                batch_loss = 0.0
                predictions_count = 0
                
                for i, pred in enumerate(predictions):
                    if i < len(targets_list):
                        target = targets_list[i].to(self.device)
                        
                        # Gestione batch size mismatch
                        if target.shape[0] != pred.shape[0]:
                            target = target[:pred.shape[0]]
                        
                        # Loss ibrida MSE + MAE (identica al training)
                        mse_loss = self.criterion(pred, target)
                        mae_loss = self.mae_criterion(pred, target)
                        
                        # Schema di ponderazione temporale identico al training
                        step_weight = 1.0 / (1.0 + i * 0.15)
                        combined_loss = (mse_weight * mse_loss + mae_weight * mae_loss) * step_weight
                        batch_loss += combined_loss
                        predictions_count += 1
                
                # Normalizza per il numero effettivo di predizioni
                if predictions_count > 0:
                    batch_loss /= predictions_count
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def plot_loss(self, history, output_dir=None):
        """
        Crea un grafico dettagliato dell'andamento della loss e learning rate durante il training.
        
        Args:
            history: Dizionario con i dati di training
            output_dir: Directory in cui salvare il grafico (opzionale)
        """
        # Crea una figura con due subplot: loss e learning rate
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Subplot 1: Loss di training e validation
        epochs = range(1, len(history['train_loss'])+1)
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        
        if 'valid_loss' in history and history['valid_loss']:
            ax1.plot(epochs, history['valid_loss'], 'r--', linewidth=2, label='Validation Loss')
            
            # Individua l'epoca con la loss minima
            min_idx = history['valid_loss'].index(min(history['valid_loss']))
            min_epoch = min_idx + 1  # +1 perché le epoche partono da 1
            min_loss = history['valid_loss'][min_idx]
            
            # Evidenzia il punto di minimo
            ax1.scatter([min_epoch], [min_loss], color='green', s=100, zorder=5, 
                       label=f'Best Model (Epoch {min_epoch}, Loss: {min_loss:.6f})')
        
        ax1.set_title('Andamento della Loss Durante il Training', fontsize=16)
        ax1.set_xlabel('Epoche', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, which="both", ls="--", alpha=0.7)
        ax1.legend(fontsize=12)
        
        # Utilizza scala logaritmica se la differenza tra massimo e minimo è grande
        max_loss = max(history['train_loss'])
        min_loss = min(history['train_loss'])
        if max_loss / (min_loss + 1e-10) > 100:  # Se il rapporto è grande, usa scala log
            ax1.set_yscale('log')
            ax1.set_ylabel('Loss (scala log)', fontsize=12)
        
        # Subplot 2: Learning rate
        if 'learning_rate' in history:
            ax2.semilogy(epochs, history['learning_rate'], 'g-', linewidth=2)
            ax2.set_title('Andamento del Learning Rate', fontsize=14)
            ax2.set_xlabel('Epoche', fontsize=12)
            ax2.set_ylabel('Learning Rate (scala log)', fontsize=12)
            ax2.grid(True, which="both", ls="--", alpha=0.7)
        
        # Migliora lo spacing tra i subplot
        plt.tight_layout()
        
        # Salva l'immagine se specificato
        if output_dir:
            output_path = os.path.join(output_dir, 'training_loss.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Grafico dettagliato delle loss salvato in: {output_path}")
        
        plt.show()
