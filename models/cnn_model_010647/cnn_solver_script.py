import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import random
import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR  # Importiamo SWA
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

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

class StdConv2d(nn.Conv2d):
    """Convolution con Weight Standardization per una migliore generalizzazione"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(StdConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride, 
            padding=padding, bias=bias
        )
        
    def forward(self, x):
        # Weight standardization
        weight = self.weight
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=[1, 2, 3], keepdim=True) + 1e-5
        weight = weight / std
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class WeightStandardizedConv2d(nn.Conv2d):
    """Implementazione alternativa di Weight Standardization più efficiente"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WeightStandardizedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SpectralConv2d(nn.Module):
    """Convolution 2D con regolarizzazione spettrale per controllo della Lipschitzianità."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(SpectralConv2d, self).__init__()
        self.conv = nn.utils.spectral_norm(
            WeightStandardizedConv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, stride=stride)
        )
    
    def forward(self, x):
        return self.conv(x)

class StochasticDepth(nn.Module):
    """Implementazione di Stochastic Depth: salta alcuni layer casualmente durante il training."""
    def __init__(self, drop_prob=0.3):  # Aumentiamo la drop probability a 0.3
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.keep_prob
        return x * binary_tensor / self.keep_prob

class ResUNetBlock(nn.Module):
    """Blocco avanzato con connessione residuale per l'architettura U-Net."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1, groups=8):
        super(ResUNetBlock, self).__init__()
        # Usa SpectralConv2d per regolarizzazione spettrale
        self.conv1 = SpectralConv2d(in_channels, out_channels, kernel_size, padding=padding)
        # GroupNorm invece di BatchNorm per stabilità indipendente dalla batch size
        self.gn1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = SpectralConv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.gn2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)  # LeakyReLU per prevenire dying ReLU
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.stochastic_depth = StochasticDepth(drop_prob=0.2)  # 20% di probabilità di saltare il blocco
        
        # Connessione residuale (skip connection)
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                WeightStandardizedConv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(groups, out_channels), out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.gn2(self.conv2(x))
        
        # Somma con la connessione residuale attraverso stochastic depth
        x = self.stochastic_depth(x) + residual
        x = self.relu(x)
        
        return x

class CNNSolver(nn.Module):
    """
    Risolve l'equazione del monodominio usando un modello CNN avanzato e profondo.
    Questo modello predice la soluzione al tempo t+dt dato lo stato al tempo t.
    
    Architettura migliorata con connessioni residuali, aumentata profondità, incrementato spazio latente,
    e regolarizzazione tramite dropout per prevenire l'overfitting.
    Include ora anche regolarizzazione spettrale, stochastic depth, GroupNorm e weight standardization
    per combattere l'overfitting e migliorare la generalizzazione.
    """
    def __init__(self, device=None, prediction_steps=1, latent_dim=128, nvx=101, nvy=101, dropout_rate=0.5):
        super(CNNSolver, self).__init__()
        
        self.device = device if device is not None else torch.device('cpu')
        self.prediction_steps = prediction_steps  # Quanti passi di tempo predire in avanti
        self.latent_dim = latent_dim
        self.nvx = nvx  # Numero di punti nella direzione x
        self.nvy = nvy  # Numero di punti nella direzione y
        
        # Encoder molto più profondo con più livelli di estrazione caratteristiche
        self.encoder = nn.Sequential(
            # Primo livello - estrazione caratteristiche iniziali
            WeightStandardizedConv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.1, inplace=True),
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
            nn.Dropout2d(0.25),  # Aggiunto dropout addizionale tra livelli
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
            nn.Dropout2d(0.2),  # Aggiunto dropout addizionale
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Terzo livello di upsampling - media risoluzione
            ResUNetBlock(64, 32, dropout_rate=dropout_rate),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Livello di output - alta risoluzione
            ResUNetBlock(32, 16, dropout_rate=dropout_rate),
            
            # Adatta le dimensioni esattamente all'output desiderato
            nn.Upsample(size=(nvx, nvy), mode='bilinear', align_corners=True),
            WeightStandardizedConv2d(16, 8, kernel_size=3, padding=1),
            nn.GroupNorm(2, 8),
            nn.LeakyReLU(0.1, inplace=True),
            WeightStandardizedConv2d(8, 1, kernel_size=1),
            nn.Sigmoid()  # Garantisce output in [0,1]
        )
        
        # Proiezione dei parametri fisici nello spazio latente con architettura più complessa
        # e ulteriormente regolarizzata per ridurre l'overfitting
        self.params_projection = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate + 0.1),  # Dropout aumentato
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate + 0.1),  # Dropout aumentato
            nn.Linear(128, 256),
            nn.LayerNorm(256),  # Aggiunta normalizzazione
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate + 0.05),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)  # Normalizzazione finale
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
    def __init__(self, model, learning_rate=5e-4, device=None, weight_decay=1e-3):
        self.device = device if device is not None else torch.device('cpu')
        self.model = model.to(self.device)
        
        # Configurazione base dell'ottimizzatore per SAM
        base_optimizer = lambda params, **kwargs: optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,  # Peso della regolarizzazione L2 ulteriormente aumentato
            betas=(0.9, 0.999), 
            eps=1e-8
        )
        
        # Sharpness-Aware Minimization per trovare minimi più piatti della loss
        self.optimizer = SAM(
            self.model.parameters(),
            base_optimizer,
            rho=0.05,  # Parametro di perturbazione
            adaptive=True  # Perturbazione adattiva per parametro
        )
        
        # Stochastic Weight Averaging per migliorare la generalizzazione
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            self.optimizer.base_optimizer,  # Nota: ora usiamo base_optimizer dentro SAM
            swa_lr=learning_rate * 0.5,
            anneal_epochs=5,
            anneal_strategy='cos'
        )
        self.swa_start = 100  # Inizia SWA dopo 100 epoche
        
        # Loss functions più avanzate con focus su diverse caratteristiche dell'errore
        self.criterion = nn.MSELoss()  # Errori grandi
        self.mae_criterion = nn.L1Loss()  # Robustezza agli outlier
        
        # Learning rate scheduler - riduce il LR quando la loss si stabilizza
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.base_optimizer,  # Nota: ora usiamo base_optimizer dentro SAM
            'min', patience=20, factor=0.5, min_lr=5e-8, verbose=True
        )
        
        # Scheduler secondario per oscillazioni LR
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer.base_optimizer,  # Nota: ora usiamo base_optimizer dentro SAM
            T_0=20, T_mult=2, eta_min=1e-7
        )
        
        # Tracciamento del best model per early stopping
        self.best_loss = float('inf')
        self.best_model_state = None
    
    def generate_training_data_from_fem(self, fem_solver, sigmas, T, dt, num_samples=100, seed=42):
        """
        Genera dati di training dal solver FEM con opzionale data augmentation.
        
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
        random.seed(seed)
        
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
                
                # Aggiungi l'esempio originale al dataset
                inputs.append(torch.tensor(input_state, dtype=torch.float32).view(1, nvx, nvy))
                targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                sigma_values.append(sigma)
                
                # Data Augmentation 1: Aggiungi rumore gaussiano all'input (aumenta la robustezza)
                if np.random.rand() < 0.7:  # 70% di probabilità
                    noise_level = np.random.uniform(0.001, 0.015)  # Rumore maggiore
                    noisy_input = input_state + np.random.normal(0, noise_level, input_state.shape)
                    # Mantieni i valori nell'intervallo [0, 1]
                    noisy_input = np.clip(noisy_input, 0, 1)
                    inputs.append(torch.tensor(noisy_input, dtype=torch.float32).view(1, nvx, nvy))
                    targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                    sigma_values.append(sigma)
                
                # Data Augmentation 2: Jitter più aggressivo nei valori di sigma
                if np.random.rand() < 0.7:  # 70% di probabilità
                    jitter_factor = np.random.uniform(0.8, 1.2)  # Range più ampio
                    jittered_sigma = sigma * jitter_factor
                    inputs.append(torch.tensor(input_state, dtype=torch.float32).view(1, nvx, nvy))
                    targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                    sigma_values.append(jittered_sigma)
                    
                # Data Augmentation 3: Cutout - applica maschere casuali all'input
                if np.random.rand() < 0.5:  # 50% di probabilità
                    mask_size = np.random.randint(5, 15)  # Dimensione della maschera
                    cutout_input = input_state.copy()
                    h_start = np.random.randint(0, nvx - mask_size)
                    w_start = np.random.randint(0, nvy - mask_size)
                    cutout_input[h_start:h_start+mask_size, w_start:w_start+mask_size] = 0
                    inputs.append(torch.tensor(cutout_input, dtype=torch.float32).view(1, nvx, nvy))
                    targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                    sigma_values.append(sigma)
        
        return inputs, targets, sigma_values
    
    def _apply_mixup(self, inputs, targets_list, sigmas, alpha=0.2):
        """
        Applica tecniche avanzate di mixup e cutmix per ridurre l'overfitting.
        
        Args:
            inputs: Tensore di input [batch, channels, height, width]
            targets_list: Lista di tensori target
            sigmas: Valori di sigma
            alpha: Parametro di mixup per il campionamento Beta
            
        Returns:
            mixed_inputs: Input mixati
            mixed_targets_list: Target mixati
            mixed_sigmas: Sigma mixati
        """
        # Caso di batch singolo: non applicare mixup
        if inputs.size(0) <= 1:
            return inputs, targets_list, sigmas
        
        # Scegli tra mixup (70%) e cutmix (30%)
        use_cutmix = np.random.rand() < 0.3
            
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        if use_cutmix and inputs.size(2) > 10 and inputs.size(3) > 10:  # Solo se l'immagine è abbastanza grande
            # Applica cutmix: scambia una regione rettangolare tra due immagini
            mixed_inputs = inputs.clone()
            _, _, H, W = inputs.shape
            # Genera le dimensioni e posizione del rettangolo
            cut_rat = np.sqrt(1.0 - lam)  # Rapporto di taglio
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            # Assicurati che il taglio sia almeno di dimensione 1
            cut_w = max(1, cut_w)
            cut_h = max(1, cut_h)
            
            cx = np.random.randint(W)  # centro x
            cy = np.random.randint(H)  # centro y
            
            # Definisce i limiti del rettangolo
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            # Sostituisci la regione rettangolare con i dati da altre immagini
            if bbx2 > bbx1 and bby2 > bby1:  # Verifica che il rettangolo abbia area positiva
                mixed_inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
                
                # Ricalcola lambda in base all'area effettivamente tagliata
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (W * H)
            else:
                # Fallback a mixup standard
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        else:
            # Mixup standard
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        
        # Mix sigma values
        mixed_sigmas = lam * sigmas + (1 - lam) * sigmas[index]
        
        # Mix targets (per ciascun timestep)
        mixed_targets_list = []
        
        for targets in targets_list:
            # Garantiamo che targets abbia la stessa dimensione del batch di inputs
            if targets.size(0) != batch_size:
                # Se il batch target è più grande, troncalo
                if targets.size(0) > batch_size:
                    targets = targets[:batch_size]
                # Se il batch target è più piccolo, non possiamo applicare il mixup in modo sicuro
                else:
                    # Ripeti l'ultimo elemento per arrivare alla stessa dimensione
                    pad_size = batch_size - targets.size(0)
                    padding = targets[-1:].repeat(pad_size, 1, 1, 1)
                    targets = torch.cat([targets, padding], dim=0)
                
            # Verifichiamo che l'indice di permutazione sia compatibile
            permuted_index = index
            if permuted_index.size(0) > targets.size(0):
                permuted_index = permuted_index[:targets.size(0)]
                
            # Applichiamo il mixup
            mixed_targets = lam * targets + (1 - lam) * targets[permuted_index, :]
            mixed_targets_list.append(mixed_targets)
            
        return mixed_inputs, mixed_targets_list, mixed_sigmas
    
    def train(self, train_loader, valid_loader=None, num_epochs=300, mse_weight=0.6, mae_weight=0.4, early_stop_patience=100):
        """
        Addestra il modello CNN sui dati forniti con strategia di training avanzata.
        Utilizza SAM (Sharpness-Aware Minimization) con doppio backward pass,
        data augmentation avanzata, SWA e altri meccanismi anti-overfitting.
        
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
            'learning_rate': [],
            'train_val_gap': []  # Tracciamento del gap train/validation per monitorare l'overfitting
        }
        
        best_valid_loss = float('inf')
        early_stop_counter = 0
        
        # Learning rate warmup - inizia con un LR basso e aumenta gradualmente
        initial_lr = self.optimizer.base_optimizer.param_groups[0]['lr']
        warmup_epochs = min(10, num_epochs // 10)  # 10 epoche o 10% del totale
        
        for epoch in range(num_epochs):
            # Learning rate warmup
            if epoch < warmup_epochs:
                # Aumenta linearmente il LR da 10% a 100%
                lr_scale = 0.1 + 0.9 * epoch / warmup_epochs
                for param_group in self.optimizer.base_optimizer.param_groups:
                    param_group['lr'] = initial_lr * lr_scale
            
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            # Print epoch information
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            for inputs, targets_list, sigmas in train_loader:
                inputs = inputs.to(self.device)
                sigmas = sigmas.to(self.device)
                
                # Converti la lista di tensori target in lista di tensori su device
                targets_device = []
                for t in targets_list:
                    targets_device.append(t.to(self.device))
                
                # Applica data augmentation avanzata con probabilità crescente
                aug_prob = min(0.8, 0.2 + epoch / (num_epochs * 0.8))  # Aumenta gradualmente fino all'80%
                if np.random.rand() < aug_prob:
                    inputs = advanced_data_augmentation(
                        inputs, 
                        sigma=min(0.02, 0.005 + 0.015 * epoch / num_epochs),  # Rumore crescente
                        cutout_prob=min(0.6, 0.2 + 0.4 * epoch / num_epochs),  # Probabilità cutout crescente
                        noise_prob=min(0.8, 0.5 + 0.3 * epoch / num_epochs)    # Probabilità noise crescente
                    )
                
                # Applica mixup o cutmix con probabilità crescente dopo le prime 30 epoche
                use_mixup = epoch >= 30 and np.random.rand() < min(0.5, 0.2 + 0.3 * epoch / num_epochs) and inputs.size(0) > 1
                if use_mixup:
                    mixup_alpha = min(0.4, 0.1 + 0.3 * epoch / num_epochs)  # Intensità di mixup crescente
                    inputs, targets_device, sigmas = self._apply_mixup(inputs, targets_device, sigmas, alpha=mixup_alpha)
                
                # Forward pass
                predictions = self.model(inputs, sigmas)
                
                # Calcola la loss ibrida (MSE + MAE) con ponderazione temporale
                batch_loss = 0.0
                predictions_count = 0
                
                for i, pred in enumerate(predictions):
                    if i < len(targets_device):
                        # Prendi il target corrispondente al passo temporale i
                        target = targets_device[i]
                        
                        # Gestione più robusta del batch size mismatch
                        if target.shape[0] != pred.shape[0]:
                            # Se il target è più grande del pred, troncalo
                            if target.shape[0] > pred.shape[0]:
                                target = target[:pred.shape[0]]
                            # Se il target è più piccolo del pred, dobbiamo adattare pred
                            else:
                                pred = pred[:target.shape[0]]
                        
                        # Controllo di sicurezza che i tensor abbiano la stessa dimensione
                        assert target.shape == pred.shape, f"Shape mismatch: target {target.shape}, pred {pred.shape}"
                        
                        # Label smoothing: aggiunge un leggero rumore ai target per ridurre l'overfitting
                        # Basato sul concetto che predizioni troppo "sicure" possono portare a overfitting
                        if epoch > warmup_epochs:
                            # Applica label smoothing solo dopo il warmup, con intensità crescente
                            eps_min = 0.05  # Valore minimo di smoothing
                            eps_max = 0.15  # Valore massimo di smoothing
                            progress = min(1.0, (epoch - warmup_epochs) / (num_epochs - warmup_epochs))  # Progresso 0-1
                            eps = eps_min + progress * (eps_max - eps_min)  # Aumenta progressivamente
                            
                            # Usa una distribuzione più avanzata per il label smoothing
                            if np.random.rand() < 0.3:  # 30% probabilità di usare smoothing non uniforme
                                # Aggiungi un po' di rumore casuale ai target (simula incertezze)
                                noise = torch.randn_like(target) * 0.02
                                smooth_target = target * (1 - eps) + eps/2 + noise * eps
                            else:
                                # Smoothing standard
                                smooth_target = target * (1 - eps) + eps/2  # Shifta leggermente dal target perfetto
                                
                            # Garantisci che i valori rimangano in [0,1] per stabilità
                            smooth_target = torch.clamp(smooth_target, 0.0, 1.0)
                        else:
                            smooth_target = target
                            
                        # Loss ibrida MSE + MAE
                        mse_loss = self.criterion(pred, smooth_target)
                        mae_loss = self.mae_criterion(pred, smooth_target)
                        
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
                
                # SAM richiede due backward/step separati
                # Prima fase: calcola e applica la perturbazione
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.first_step(zero_grad=True)
                
                # Seconda fase: calcola gradiente sui pesi perturbati
                # Forward pass sul modello con pesi perturbati
                predictions_perturbed = self.model(inputs, sigmas)
                
                # Ricalcola la loss sui pesi perturbati
                batch_loss_perturbed = 0.0
                predictions_count_perturbed = 0
                
                for i, pred in enumerate(predictions_perturbed):
                    if i < len(targets_device):
                        target = targets_device[i]
                        if target.shape[0] != pred.shape[0]:
                            if target.shape[0] > pred.shape[0]:
                                target = target[:pred.shape[0]]
                            else:
                                pred = pred[:target.shape[0]]
                        
                        # Usa target non-smoothed per la seconda fase SAM
                        mse_loss = self.criterion(pred, target)
                        mae_loss = self.mae_criterion(pred, target)
                        step_weight = 1.0 / (1.0 + i * 0.15)
                        combined_loss = (mse_weight * mse_loss + mae_weight * mae_loss) * step_weight
                        batch_loss_perturbed += combined_loss
                        predictions_count_perturbed += 1
                
                if predictions_count_perturbed > 0:
                    batch_loss_perturbed /= predictions_count_perturbed
                
                # Backward sui pesi perturbati
                batch_loss_perturbed.backward()
                
                # Gradient clipping per stabilità numerica
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Ripristina i pesi originali e applica l'aggiornamento
                self.optimizer.second_step(zero_grad=True)
                
                train_loss += batch_loss.item()
                num_batches += 1
            
            # Gestione degli scheduler
            if epoch >= self.swa_start:
                # Dopo l'inizio di SWA, usa lo scheduler SWA
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            elif epoch % 5 == 0:
                # Prima dell'inizio di SWA, usa cosine annealing ogni 5 epoche
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
                        self.model.load_state_dict(self.best_model_state)
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
        
        # Close the figure to release memory
        plt.close(fig)

class SAM(torch.optim.Optimizer):
    """
    Implementazione di Sharpness-Aware Minimization (SAM)
    che cerca minimi più ampi e piatti della funzione di loss.
    
    Basato su: Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    https://arxiv.org/abs/2010.01412
    
    Questa tecnica offre un'ottima generalizzazione per modelli profondi.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: iterabile con i parametri del modello
            base_optimizer: ottimizzatore base (es. torch.optim.SGD)
            rho: parametro di perturbazione per la norma del gradiente
            adaptive: se True, usa una perturbazione specifica per ogni parametro
            **kwargs: parametri aggiuntivi per l'ottimizzatore base
        """
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Calcola e applica la perturbazione ai pesi.
        Deve essere chiamato dopo loss.backward() ma prima di optimizer.step()
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                
                # Calcola perturbazione
                if group["adaptive"]:
                    perturb = p.grad * scale.to(p) * (p.abs() + 1e-12)
                else:
                    perturb = p.grad * scale.to(p)
                
                # Perturba i pesi
                self.state[p]["old_p"] = p.data.clone()
                p.add_(perturb)
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Ripristina i pesi originali e applica l'aggiornamento normale.
        Deve essere chiamato dopo la seconda loss.backward()
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]: continue
                p.data = self.state[p]["old_p"]  # Ripristina i pesi originali
        
        self.base_optimizer.step()  # Applica l'ottimizzazione normale
        
        if zero_grad: self.zero_grad()
    
    def _grad_norm(self):
        """
        Calcola la norma L2 dei gradienti, usata per la perturbazione
        """
        norm = torch.norm(
            torch.stack([
                ((p.grad.detach().abs() ** 2).sum() if p.grad is not None else 0.0)
                for group in self.param_groups for p in group["params"]
            ]).float().sqrt()
        )
        return norm
    
    def step(self, closure=None):
        """
        Esegue un passo di ottimizzazione (prima e seconda fase)
        """
        if closure is not None:
            raise RuntimeError("SAM richiede due backward/step separati - usa first_step e second_step")
        
        raise RuntimeError("SAM richiede due backward/step separati - usa first_step e second_step")

class ShakeShake(nn.Module):
    """Implementazione di Shake-Shake regularization (Gastaldi, 2017)
    Una tecnica per regolarizzare modelli con connessioni parallele."""
    def __init__(self):
        super(ShakeShake, self).__init__()

    def forward(self, x1, x2):
        if self.training:
            # Coefficienti casuali per il forward pass
            alpha = torch.rand(x1.size(0), 1, 1, 1).to(x1.device)
            # Forward pass: shake
            y = alpha * x1 + (1 - alpha) * x2
            # Genera nuovi coefficienti casuali per il backward pass
            self.beta = torch.rand(x1.size(0), 1, 1, 1).to(x1.device)
            return y
        else:
            # In fase di validazione, media semplice (0.5/0.5)
            return 0.5 * x1 + 0.5 * x2
    
    def backward(self, grad_output):
        # Backward pass utilizza i coefficienti beta invece di alpha
        return self.beta * grad_output, (1 - self.beta) * grad_output

def advanced_data_augmentation(inputs, sigma=0.02, cutout_prob=0.5, noise_prob=0.8):
    """
    Applica tecniche di data augmentation avanzate alle immagini di input.
    
    Args:
        inputs: Tensore di input [batch_size, channels, height, width]
        sigma: Forza del rumore gaussiano
        cutout_prob: Probabilità di applicare cutout
        noise_prob: Probabilità di applicare rumore gaussiano
        
    Returns:
        Tensore augmentato della stessa dimensione
    """
    batch_size, channels, height, width = inputs.shape
    augmented = inputs.clone()
    
    # Rumore gaussiano con probabilità noise_prob
    if random.random() < noise_prob:
        noise = torch.randn_like(augmented) * sigma * random.uniform(0.5, 1.5)
        augmented = augmented + noise
        augmented = torch.clamp(augmented, 0.0, 1.0)
    
    # Cutout con probabilità cutout_prob
    if random.random() < cutout_prob:
        # Definisci dimensioni del cutout (10-20% dell'immagine)
        cutout_size_h = int(height * random.uniform(0.1, 0.2))
        cutout_size_w = int(width * random.uniform(0.1, 0.2))
        
        # Per ogni immagine nel batch
        for i in range(batch_size):
            # Posizione casuale del cutout
            top = random.randint(0, height - cutout_size_h - 1)
            left = random.randint(0, width - cutout_size_w - 1)
            
            # Applica cutout (metti a zero)
            augmented[i, :, top:top+cutout_size_h, left:left+cutout_size_w] = 0.0
    
    # Shift dei valori con probabilità 0.3
    if random.random() < 0.3:
        shift = random.uniform(-0.05, 0.05)
        augmented = augmented + shift
        augmented = torch.clamp(augmented, 0.0, 1.0)
    
    # Jitter di contrasto con probabilità 0.3
    if random.random() < 0.3:
        factor = random.uniform(0.8, 1.2)
        augmented = (augmented - 0.5) * factor + 0.5
        augmented = torch.clamp(augmented, 0.0, 1.0)
    
    return augmented
