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
    
    Nuove funzionalità:
    - Deep Supervision con output intermedi
    - Feature extraction per Self-Distillation e Feature Alignment
    - Architettura a multi-scala con skip connections migliorate
    - Incorporazione più complessa dei parametri fisici
    """
    def __init__(self, device=None, prediction_steps=1, latent_dim=128, nvx=101, nvy=101, dropout_rate=0.5):
        super(CNNSolver, self).__init__()
        
        self.device = device if device is not None else torch.device('cpu')
        self.prediction_steps = prediction_steps  # Quanti passi di tempo predire in avanti
        self.latent_dim = latent_dim
        self.nvx = nvx  # Numero di punti nella direzione x
        self.nvy = nvy  # Numero di punti nella direzione y
        
        # ----- ENCODER PATH -----
        self.enc_stage1 = nn.Sequential(
            WeightStandardizedConv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.1, inplace=True),
            ResUNetBlock(16, 32, dropout_rate=dropout_rate)
        )
        self.enc_pool1 = nn.MaxPool2d(2)  # Riduzione a 1/2
        
        self.enc_stage2 = nn.Sequential(
            ResUNetBlock(32, 64, dropout_rate=dropout_rate),
            ResUNetBlock(64, 64, dropout_rate=dropout_rate)
        )
        self.enc_pool2 = nn.MaxPool2d(2)  # Riduzione a 1/4
        
        self.enc_stage3 = nn.Sequential(
            ResUNetBlock(64, 128, dropout_rate=dropout_rate),
            ResUNetBlock(128, 128, dropout_rate=dropout_rate)
        )
        self.enc_pool3 = nn.MaxPool2d(2)  # Riduzione a 1/8
        
        self.enc_stage4 = nn.Sequential(
            ResUNetBlock(128, 256, dropout_rate=dropout_rate),
            nn.Dropout2d(0.25)
        )
        self.enc_pool4 = nn.MaxPool2d(2)  # Riduzione a 1/16
        
        # ----- BOTTLENECK -----
        self.bottleneck = nn.Sequential(
            ResUNetBlock(256, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate)
        )
        
        # ----- DECODER PATH -----
        self.dec_up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_stage4 = nn.Sequential(
            ResUNetBlock(latent_dim + 256, 256, dropout_rate=dropout_rate),
            ResUNetBlock(256, 256, dropout_rate=dropout_rate)
        )
        # Output intermedio per deep supervision
        self.aux_out4 = nn.Sequential(
            WeightStandardizedConv2d(256, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(size=(nvx, nvy), mode='bilinear', align_corners=True),
            WeightStandardizedConv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dec_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_stage3 = nn.Sequential(
            ResUNetBlock(256 + 128, 128, dropout_rate=dropout_rate),
            ResUNetBlock(128, 128, dropout_rate=dropout_rate),
            nn.Dropout2d(0.2)
        )
        # Output intermedio per deep supervision
        self.aux_out3 = nn.Sequential(
            WeightStandardizedConv2d(128, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(size=(nvx, nvy), mode='bilinear', align_corners=True),
            WeightStandardizedConv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dec_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_stage2 = nn.Sequential(
            ResUNetBlock(128 + 64, 64, dropout_rate=dropout_rate),
            ResUNetBlock(64, 64, dropout_rate=dropout_rate)
        )
        # Output intermedio per deep supervision
        self.aux_out2 = nn.Sequential(
            WeightStandardizedConv2d(64, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(size=(nvx, nvy), mode='bilinear', align_corners=True),
            WeightStandardizedConv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dec_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_stage1 = nn.Sequential(
            ResUNetBlock(64 + 32, 32, dropout_rate=dropout_rate),
            ResUNetBlock(32, 32, dropout_rate=dropout_rate)
        )
        
        # Output finale
        self.final = nn.Sequential(
            nn.Upsample(size=(nvx, nvy), mode='bilinear', align_corners=True),
            WeightStandardizedConv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(0.1, inplace=True),
            WeightStandardizedConv2d(16, 8, kernel_size=3, padding=1),
            nn.GroupNorm(2, 8),
            nn.LeakyReLU(0.1, inplace=True),
            WeightStandardizedConv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ----- INTEGRAZIONE PARAMETRI FISICI -----
        # Proiezione potenziata dei parametri fisici
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
        
        # Modulo FiLM per migliore condizionamento dei parametri fisici
        self.film_bottleneck = nn.Linear(latent_dim, latent_dim*2)  # Scale & shift per ogni canale
        
        # ----- EVOLUZIONE TEMPORALE -----
        self.time_evolution = nn.Sequential(
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate),
            ResUNetBlock(latent_dim, latent_dim, dropout_rate=dropout_rate)
        )
        
    def _apply_film(self, features, params):
        """Applica condizionamento FiLM (Feature-wise Linear Modulation)"""
        batch_size = features.size(0)
        film_params = self.film_bottleneck(params)
        gamma, beta = film_params.chunk(2, dim=1)
        
        # Reshape per broadcasting
        gamma = gamma.view(batch_size, -1, 1, 1)
        beta = beta.view(batch_size, -1, 1, 1)
        
        # Applica trasformazione affine per canale
        return features * (1 + gamma) + beta
        
    def _spatial_params_integration(self, latent, sigma_h):
        """Integra i parametri fisici nello spazio latente usando FiLM."""
        batch_size = latent.size(0)
        
        # Gestisci sigma_h in modo più robusto
        if isinstance(sigma_h, torch.Tensor):
            if sigma_h.dim() == 0:  # Scalare
                sigma_h_batch = sigma_h.expand(batch_size)
            elif sigma_h.dim() == 1:  # Vettore
                if sigma_h.size(0) == 1:
                    # Singolo valore per tutti i batch
                    sigma_h_batch = sigma_h.expand(batch_size)
                elif sigma_h.size(0) == batch_size:
                    # Un valore per ogni elemento del batch
                    sigma_h_batch = sigma_h
                else:
                    # Dimensione non compatibile, resize
                    sigma_h_batch = sigma_h[:batch_size] if sigma_h.size(0) > batch_size else torch.cat([sigma_h, sigma_h.repeat(batch_size - sigma_h.size(0))])
            elif sigma_h.dim() == 2:  # Matrice [batch_size, 1]
                sigma_h_batch = sigma_h.squeeze(1)
            else:
                raise ValueError(f"Formato non supportato per sigma_h: {sigma_h.size()}")
        else:
            # Se è uno scalare Python, convertilo in tensore
            sigma_h_batch = torch.full((batch_size,), float(sigma_h), device=self.device)
        
        # Costanti fisiche
        a = 18.515  # Costante di diffusione
        I_ext = 0.0  # Corrente esterna
        
        # Crea un tensore [batch_size, 4] con [sigma_h, sigma_t, a, I_ext]
        physical_params = torch.zeros((batch_size, 4), device=self.device)
        physical_params[:, 0] = sigma_h_batch  # Diffusività longitudinale
        physical_params[:, 1] = sigma_h_batch * 0.25  # Diffusività trasversale
        physical_params[:, 2] = a
        physical_params[:, 3] = I_ext
        
        # Proietta i parametri fisici nello spazio latente
        params_embedding = self.params_projection(physical_params)
        
        # Applica condizionamento FiLM
        modulated_latent = self._apply_film(latent, params_embedding)
        
        return modulated_latent
        
    def forward(self, x, sigma_h, steps=None, return_features=False):
        """
        Forward pass con supporto per:
        - Predizione multi-step 
        - Feature extraction per self-distillation
        - Deep supervision con output intermedi
        
        Args:
            x: Tensore di input [batch_size, 1, height, width]
            sigma_h: Tensore o scalare con i coefficienti di diffusione
            steps: Numero di passi di predizione (default: self.prediction_steps)
            return_features: Se True, ritorna anche le feature intermedie
            
        Returns:
            Tuple di tensori delle predizioni temporali e, opzionalmente, feature intermedie
        """
        if steps is None:
            steps = self.prediction_steps
            
        # Tensor encoding con connessioni skip
        enc1 = self.enc_stage1(x)
        enc2 = self.enc_stage2(self.enc_pool1(enc1))
        enc3 = self.enc_stage3(self.enc_pool2(enc2))
        enc4 = self.enc_stage4(self.enc_pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.enc_pool4(enc4))
        
        # Integrazione parametri fisici nel bottleneck
        modulated_latent = self._spatial_params_integration(bottleneck, sigma_h)
        
        # Lista per memorizzare le predizioni
        predictions = []
        
        # Lista per memorizzare le feature per self-distillation
        features = [enc1, enc2, enc3, enc4, bottleneck] if return_features else None
        
        # Stato corrente da far evolvere nel tempo
        current_state = modulated_latent
        
        # Evoluzione temporale e predizione per ciascun passo richiesto
        for _ in range(steps):
            # Evoluzione temporale
            evolved_state = self.time_evolution(current_state)
            current_state = evolved_state
            
            # Decoding path con connessioni skip - con controllo dimensioni
            # Upsampling e concatenazione con controllo dimensioni
            dec_up4_out = self.dec_up4(evolved_state)
            # Verifichiamo che le dimensioni siano compatibili
            if dec_up4_out.shape[2:] != enc4.shape[2:]:
                dec_up4_out = F.interpolate(dec_up4_out, size=enc4.shape[2:], mode='bilinear', align_corners=True)
            d4 = self.dec_stage4(torch.cat([dec_up4_out, enc4], dim=1))
            aux4 = self.aux_out4(d4)  # Output intermedio
            
            # Stage 3
            dec_up3_out = self.dec_up3(d4)
            if dec_up3_out.shape[2:] != enc3.shape[2:]:
                dec_up3_out = F.interpolate(dec_up3_out, size=enc3.shape[2:], mode='bilinear', align_corners=True)
            d3 = self.dec_stage3(torch.cat([dec_up3_out, enc3], dim=1))
            aux3 = self.aux_out3(d3)  # Output intermedio
            
            # Stage 2
            dec_up2_out = self.dec_up2(d3)
            if dec_up2_out.shape[2:] != enc2.shape[2:]:
                dec_up2_out = F.interpolate(dec_up2_out, size=enc2.shape[2:], mode='bilinear', align_corners=True)
            d2 = self.dec_stage2(torch.cat([dec_up2_out, enc2], dim=1))
            aux2 = self.aux_out2(d2)  # Output intermedio
            
            # Stage 1
            dec_up1_out = self.dec_up1(d2)
            if dec_up1_out.shape[2:] != enc1.shape[2:]:
                dec_up1_out = F.interpolate(dec_up1_out, size=enc1.shape[2:], mode='bilinear', align_corners=True)
            d1 = self.dec_stage1(torch.cat([dec_up1_out, enc1], dim=1))
            
            # Output finale
            output = self.final(d1)
            
            if self.training:
                # Durante il training, aggiungi alla lista sia l'output finale che gli intermedi
                # per la deep supervision
                predictions.append((output, aux2, aux3, aux4))
            else:
                # Durante inference usa solo l'output finale
                predictions.append(output)
                
        if return_features:
            return predictions, features
        else:
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
    Trainer avanzato per il modello CNN con strategie contro l'overfitting estremamente potenziate.
    
    Caratteristiche:
    - Ottimizzatore SAM (Sharpness-Aware Minimization)
    - Deep Supervision con skip loss
    - Self-Distillation con EMA
    - Curriculum Learning con data augmentation progressiva
    - Learning Rate scheduling avanzato e adattivo
    - Gradient clipping per stabilità numerica
    - Feature alignment loss per migliore generalizzazione
    - Dynamic loss weighting e R-Drop
    """
    def __init__(self, model, learning_rate=3e-4, device=None, weight_decay=1e-3):
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
        
        # Exponential Moving Average (EMA) per stabilizzare le predizioni
        self.ema = EMA(model, decay=0.9995)
        
        # Stochastic Weight Averaging
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            self.optimizer.base_optimizer,  # Nota: ora usiamo base_optimizer dentro SAM
            swa_lr=learning_rate * 0.5,
            anneal_epochs=5,
            anneal_strategy='cos'
        )
        self.swa_start = 100  # Inizia SWA dopo 100 epoche
        
        # Loss functions avanzate
        self.criterion = nn.MSELoss()  # Errori grandi (MSE)
        self.mae_criterion = nn.L1Loss()  # Robustezza agli outlier (MAE)
        self.distillation_loss = SelfDistillationLoss(T=3.0)  # Self-distillation
        self.feature_alignment_loss = FeatureAlignmentLoss()  # Allineamento features
        
        # Learning rate scheduler - riduce il LR quando la loss si stabilizza
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.base_optimizer,  # Nota: ora usiamo base_optimizer dentro SAM
            'min', patience=20, factor=0.5, min_lr=1e-7, verbose=True
        )
        
        # Scheduler secondario per oscillazioni LR
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer.base_optimizer,
            T_0=20, T_mult=2, eta_min=1e-7
        )
        
        # Scheduler per aumentare gradualmente l'intensità della data augmentation
        self.augmentation_scheduler = {
            'noise_sigma': lambda epoch, total: min(0.03, 0.005 + 0.025 * epoch / total),
            'cutout_prob': lambda epoch, total: min(0.7, 0.3 + 0.4 * epoch / total),
            'mixup_alpha': lambda epoch, total: min(0.8, 0.2 + 0.6 * epoch / total),
        }
        
        # Tracciamento del best model per early stopping
        self.best_loss = float('inf')
        self.best_model_state = None
        self.best_ema_state = None
    
    def generate_training_data_from_fem(self, fem_solver, sigmas, T, dt, num_samples=100, seed=42):
        """
        Genera dati di training dal solver FEM con data augmentation estrema.
        
        Args:
            fem_solver: Istanza di FEMSolver
            sigmas: Lista di coefficienti di diffusione da usare
            T: Tempo finale della simulazione
            dt: Passo temporale
            num_samples: Numero di coppie (stato, evoluzione) da generare
            
        Returns:
            inputs: Tensore degli stati iniziali
            targets: Tensore degli stati futuri 
            sigma_values: Tensore dei coefficienti di diffusione
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        nvx, nvy = fem_solver.nvx, fem_solver.nvy
        inputs = []
        targets = []
        sigma_values = []
        
        print("Generazione dati di training dal solver FEM con augmentation...")
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
                
                # ----- TECNICHE DI AUGMENTATION ANCORA PIÙ AGGRESSIVE -----
                
                # 1. Rumore gaussiano variabile (70% probabilità)
                if np.random.rand() < 0.7:
                    for noise_level in [0.005, 0.01, 0.02]:  # Multiple noise levels
                        noisy_input = input_state + np.random.normal(0, noise_level, input_state.shape)
                        noisy_input = np.clip(noisy_input, 0, 1)
                        inputs.append(torch.tensor(noisy_input, dtype=torch.float32).view(1, nvx, nvy))
                        targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                        sigma_values.append(sigma)
                
                # 2. Jitter sigma con più valori (80% probabilità)
                if np.random.rand() < 0.8:
                    for jitter_factor in [0.7, 0.85, 1.15, 1.3]:  # Multiple jitter factors
                        jittered_sigma = sigma * jitter_factor
                        inputs.append(torch.tensor(input_state, dtype=torch.float32).view(1, nvx, nvy))
                        targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                        sigma_values.append(jittered_sigma)
                
                # 3. Cutouts multipli (50% probabilità)
                if np.random.rand() < 0.5:
                    for _ in range(3):  # Try multiple cutout patterns
                        cutout_input = input_state.copy()
                        # Applica 1-3 cutouts casuali
                        num_cutouts = np.random.randint(1, 4)
                        for _ in range(num_cutouts):
                            mask_size = np.random.randint(5, 20)
                            h_start = np.random.randint(0, nvx - mask_size)
                            w_start = np.random.randint(0, nvy - mask_size)
                            cutout_input[h_start:h_start+mask_size, w_start:w_start+mask_size] = 0
                        inputs.append(torch.tensor(cutout_input, dtype=torch.float32).view(1, nvx, nvy))
                        targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                        sigma_values.append(sigma)
                
                # 4. Shift spaziale (40% probabilità)
                if np.random.rand() < 0.4:
                    for shift_amount in [1, 2, 3]:  # Shift di 1-3 pixel
                        shift_dir = np.random.randint(0, 4)  # 0=up, 1=right, 2=down, 3=left
                        shifted_input = np.zeros_like(input_state)
                        
                        if shift_dir == 0:  # Shift up
                            shifted_input[:-shift_amount, :] = input_state[shift_amount:, :]
                        elif shift_dir == 1:  # Shift right
                            shifted_input[:, shift_amount:] = input_state[:, :-shift_amount]
                        elif shift_dir == 2:  # Shift down
                            shifted_input[shift_amount:, :] = input_state[:-shift_amount, :]
                        else:  # Shift left
                            shifted_input[:, :-shift_amount] = input_state[:, shift_amount:]
                        
                        inputs.append(torch.tensor(shifted_input, dtype=torch.float32).view(1, nvx, nvy))
                        targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                        sigma_values.append(sigma)
                
                # 5. Contrast jitter (40% probabilità)
                if np.random.rand() < 0.4:
                    for contrast_factor in [0.8, 1.2]:
                        contrast_input = np.clip((input_state - 0.5) * contrast_factor + 0.5, 0, 1)
                        inputs.append(torch.tensor(contrast_input, dtype=torch.float32).view(1, nvx, nvy))
                        targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                        sigma_values.append(sigma)
                        
                # 6. GridMask (30% probabilità)
                if np.random.rand() < 0.3:
                    grid_input = input_state.copy()
                    grid_size = np.random.randint(5, 15)
                    for i in range(0, nvx, grid_size * 2):
                        for j in range(0, nvy, grid_size * 2):
                            if i + grid_size < nvx and j + grid_size < nvy:
                                grid_input[i:i+grid_size, j:j+grid_size] = 0
                    
                    inputs.append(torch.tensor(grid_input, dtype=torch.float32).view(1, nvx, nvy))
                    targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                    sigma_values.append(sigma)
                
                # 7. Elastic distortion (20% probabilità)
                if np.random.rand() < 0.2:
                    # Crea un campo di displacement
                    displacement = np.random.randn(2, nvx, nvy) * 3.0
                    from scipy.ndimage import gaussian_filter
                    displacement[0] = gaussian_filter(displacement[0], sigma=10)
                    displacement[1] = gaussian_filter(displacement[1], sigma=10)
                    
                    # Normalizza per limitare lo spostamento massimo
                    max_displacement = 5.0
                    displacement = displacement / np.max(np.abs(displacement)) * max_displacement
                    
                    # Crea una griglia di coordinate base
                    y, x = np.meshgrid(np.arange(nvy), np.arange(nvx))
                    
                    # Applica il displacement
                    x_displaced = x + displacement[0]
                    y_displaced = y + displacement[1]
                    
                    # Clip ai bordi
                    x_displaced = np.clip(x_displaced, 0, nvx - 1)
                    y_displaced = np.clip(y_displaced, 0, nvy - 1)
                    
                    # Interpolazione bilineare (implementazione semplificata)
                    x0 = np.floor(x_displaced).astype(int)
                    y0 = np.floor(y_displaced).astype(int)
                    x1 = np.minimum(x0 + 1, nvx - 1)
                    y1 = np.minimum(y0 + 1, nvy - 1)
                    
                    x_weight = x_displaced - x0
                    y_weight = y_displaced - y0
                    
                    distorted_input = (input_state[x0, y0] * (1 - x_weight) * (1 - y_weight) + 
                                      input_state[x1, y0] * x_weight * (1 - y_weight) + 
                                      input_state[x0, y1] * (1 - x_weight) * y_weight + 
                                      input_state[x1, y1] * x_weight * y_weight)
                    
                    inputs.append(torch.tensor(distorted_input, dtype=torch.float32).view(1, nvx, nvy))
                    targets.append([torch.tensor(state, dtype=torch.float32).view(1, nvx, nvy) for state in target_states])
                    sigma_values.append(sigma)
        
        print(f"Dataset generato: {len(inputs)} esempi ({len(inputs) // (num_samples // len(sigmas) * len(sigmas))}x augmentation)")
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
        
        # Scegli tra mixup (50%), cutmix (30%), e manifold mixup (20%)
        rand_val = np.random.rand()
        use_cutmix = rand_val < 0.3
        use_manifold = 0.3 <= rand_val < 0.5
            
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
            
            # CutMix avanzato: multiple regioni
            num_regions = np.random.randint(1, 4)  # 1-3 regioni di cutmix
            
            effective_lam = 1.0  # Lambda effettivo (area residua)
            
            for _ in range(num_regions):
                # Genera le dimensioni e posizione del rettangolo
                cut_rat = np.sqrt(1.0 - lam) / np.sqrt(num_regions)  # Rapporto di taglio
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
                    
                    # Aggiorna lambda in base all'area effettivamente tagliata
                    region_area = ((bbx2 - bbx1) * (bby2 - bby1)) / (W * H)
                    effective_lam -= region_area
            
            # Assicurati che lambda sia non-negativo
            effective_lam = max(0, effective_lam)
            lam = effective_lam
            
        elif use_manifold:
            # Per manifold mixup lasciamo gli input originali intatti
            # Il mixup verrà fatto a livello di feature latenti durante il forward pass
            # Qui settiamo solo il flag per indicare che vogliamo usare manifold mixup
            mixed_inputs = inputs
            # La lambda verrà usata durante il forward pass
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
    
    def _compute_loss_with_deep_supervision(self, predictions, targets, epoch, num_epochs):
        """
        Calcola la loss con deep supervision (output intermedi).
        
        Args:
            predictions: Lista di tuple (output, aux2, aux3, aux4)
            targets: Lista di tensori target
            epoch: Epoca corrente per ponderazione dinamica
            num_epochs: Numero totale di epoche
            
        Returns:
            Loss totale
        """
        batch_loss = 0.0
        predictions_count = 0
        
        # Coefficiente per la deep supervision (aumenta col training)
        aux_weight = min(0.3, 0.05 + 0.25 * epoch / num_epochs)
        
        for i, pred_tuple in enumerate(predictions):
            if i < len(targets):
                target = targets[i]
                
                # Unpack della tupla di predizione
                if isinstance(pred_tuple, tuple) and len(pred_tuple) > 1:
                    # Durante training: output principale + outputs ausiliari
                    main_pred, aux2, aux3, aux4 = pred_tuple
                else:
                    # Durante val: solo output principale
                    main_pred = pred_tuple
                
                # Gestione più robusta del batch size mismatch
                if target.shape[0] != main_pred.shape[0]:
                    if target.shape[0] > main_pred.shape[0]:
                        target = target[:main_pred.shape[0]]
                    else:
                        main_pred = main_pred[:target.shape[0]]
                
                # Controllo di sicurezza che i tensor abbiano la stessa dimensione
                assert target.shape == main_pred.shape, f"Shape mismatch: target {target.shape}, pred {main_pred.shape}"
                
                # Loss principale (MSE + MAE)
                mse_loss = self.criterion(main_pred, target)
                mae_loss = self.mae_criterion(main_pred, target)
                main_loss = 0.7 * mse_loss + 0.3 * mae_loss
                
                # Step weight: ponderazione temporale
                step_weight = 1.0 / (1.0 + i * 0.15)
                combined_loss = main_loss * step_weight
                
                # Aggiungi loss di deep supervision per output intermedi
                if isinstance(pred_tuple, tuple) and len(pred_tuple) > 1:
                    aux2_loss = 0.7 * self.criterion(aux2, target) + 0.3 * self.mae_criterion(aux2, target)
                    aux3_loss = 0.7 * self.criterion(aux3, target) + 0.3 * self.mae_criterion(aux3, target)
                    aux4_loss = 0.7 * self.criterion(aux4, target) + 0.3 * self.mae_criterion(aux4, target)
                    
                    # Aggiungi le loss ausiliarie con peso crescente durante il training
                    combined_loss += aux_weight * (aux2_loss + aux3_loss + aux4_loss) * step_weight
                
                batch_loss += combined_loss
                predictions_count += 1
        
        # Normalizza per il numero effettivo di predizioni
        if predictions_count > 0:
            batch_loss /= predictions_count
            
        return batch_loss
    
    def train(self, train_loader, valid_loader=None, num_epochs=3000, mse_weight=0.6, mae_weight=0.4, early_stop_patience=200):
        """
        Addestra il modello CNN sui dati forniti con strategie anti-overfitting estremamente potenziate.
        
        Args:
            train_loader: DataLoader con i dati di training
            valid_loader: DataLoader con i dati di validazione
            num_epochs: Numero di epoche di training (default 3000 per convergenza molto profonda)
            mse_weight: Peso della MSE loss nella combinazione
            mae_weight: Peso della MAE loss nella combinazione
            early_stop_patience: Numero di epoche di attesa prima di early stopping
            
        Returns:
            dict: Storia del training
        """
        print(f"Avvio training ultra-avanzato del modello CNN con {num_epochs} epoche...")
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
        warmup_epochs = min(50, num_epochs // 20)  # 50 epoche o 5% del totale
        
        # Flag per tracking di training phases
        using_ema = False
        using_swa = False
        
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
            
            # Parametri di augmentation dinamici in base alla fase di training
            aug_progress = epoch / num_epochs
            noise_sigma = self.augmentation_scheduler['noise_sigma'](epoch, num_epochs)
            cutout_prob = self.augmentation_scheduler['cutout_prob'](epoch, num_epochs)
            mixup_alpha = self.augmentation_scheduler['mixup_alpha'](epoch, num_epochs)
            
            # Attiva EMA dopo alcune epoche
            use_ema_epoch = num_epochs // 10  # 10% delle epoche
            if epoch >= use_ema_epoch and not using_ema:
                print(f"Epoch {epoch+1}: Attivazione EMA (Exponential Moving Average)")
                using_ema = True
            
            for inputs, targets_list, sigmas in train_loader:
                inputs = inputs.to(self.device)
                sigmas = sigmas.to(self.device)
                
                # Converti la lista di tensori target in lista di tensori su device
                targets_device = []
                for t in targets_list:
                    targets_device.append(t.to(self.device))
                
                # Applica data augmentation avanzata con parametri dinamici
                inputs = advanced_data_augmentation(
                    inputs, 
                    sigma=noise_sigma,
                    cutout_prob=cutout_prob,
                    noise_prob=min(0.9, 0.5 + 0.4 * aug_progress)
                )
                
                # Applica mixup/cutmix con probabilità e intensità crescenti
                use_mixup = np.random.rand() < min(0.7, 0.3 + 0.4 * aug_progress)
                if use_mixup and inputs.size(0) > 1:
                    inputs, targets_device, sigmas = self._apply_mixup(
                        inputs, targets_device, sigmas, alpha=mixup_alpha
                    )
                
                # ----- PRIMO PASSAGGIO SAM: CALCOLO GRADIENTE ORIGINALE -----
                
                # Forward pass con flag per features extraction (per self-distillation)
                predictions, features = self.model(inputs, sigmas, return_features=True)
                
                # Calcola la loss con deep supervision
                batch_loss = self._compute_loss_with_deep_supervision(
                    predictions, targets_device, epoch, num_epochs
                )
                
                # R-Drop: consistenza tra dropout diversi (attivo dopo 20% delle epoche)
                if epoch > num_epochs * 0.2 and np.random.rand() < 0.3:
                    # Secondo forward pass con dropout diverso
                    predictions2, features2 = self.model(inputs, sigmas, return_features=True)
                    
                    # Loss di consistenza tra le due forward pass
                    rdrop_weight = min(0.2, 0.05 + 0.15 * (epoch - num_epochs * 0.2) / (num_epochs * 0.8))
                    
                    for i, (pred1, pred2) in enumerate(zip(predictions, predictions2)):
                        # Prendiamo solo l'output principale, non gli ausiliari
                        if isinstance(pred1, tuple):
                            pred1 = pred1[0]  # main output
                        if isinstance(pred2, tuple):
                            pred2 = pred2[0]  # main output
                            
                        # MSE tra le due predizioni come loss di consistenza
                        rdrop_loss = self.criterion(pred1, pred2)
                        batch_loss += rdrop_weight * rdrop_loss
                
                # Self-Distillation con EMA teacher (attivo se EMA è attivo)
                if using_ema and epoch >= use_ema_epoch:
                    # Salva temporaneamente i pesi originali
                    self.ema.apply_shadow()
                    
                    # Forward pass con il teacher (EMA model)
                    with torch.no_grad():
                        teacher_predictions, teacher_features = self.model(inputs, sigmas, return_features=True)
                    
                    # Ripristina i pesi originali
                    self.ema.restore()
                    
                    # Feature alignment loss tra student e teacher
                    align_loss = self.feature_alignment_loss(features, teacher_features)
                    
                    # Peso crescente per la feature alignment
                    align_weight = min(0.2, 0.05 + 0.15 * (epoch - use_ema_epoch) / (num_epochs - use_ema_epoch))
                    batch_loss += align_weight * align_loss
                    
                    # Self-distillation tra output di student e teacher
                    for i, (student_pred, teacher_pred) in enumerate(zip(predictions, teacher_predictions)):
                        # Prendiamo solo l'output principale
                        if isinstance(student_pred, tuple):
                            student_pred = student_pred[0]
                        if isinstance(teacher_pred, tuple):
                            teacher_pred = teacher_pred[0]
                            
                        # Peso della distillation aumenta gradualmente
                        distill_weight = min(0.25, 0.05 + 0.2 * (epoch - use_ema_epoch) / (num_epochs - use_ema_epoch))
                        
                        # Peso temporale per dare più importanza ai primi timestep
                        time_weight = 1.0 / (1.0 + i * 0.2)
                        
                        # MSE tra student e teacher
                        distill_loss = self.criterion(student_pred, teacher_pred.detach())
                        batch_loss += distill_weight * distill_loss * time_weight
                
                # Prima fase SAM: calcola e applica la perturbazione
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.first_step(zero_grad=True)
                
                # ----- SECONDO PASSAGGIO SAM: CALCOLO GRADIENTE SUI PESI PERTURBATI -----
                
                # Forward pass sul modello con pesi perturbati
                perturbed_predictions = self.model(inputs, sigmas)
                
                # Ricalcola la loss sui pesi perturbati (più semplice, solo loss principale)
                perturbed_loss = self._compute_loss_with_deep_supervision(
                    perturbed_predictions, targets_device, epoch, num_epochs
                )
                
                # Backward sui pesi perturbati
                perturbed_loss.backward()
                
                # Gradient clipping per stabilità numerica
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Ripristina i pesi originali e applica l'aggiornamento
                self.optimizer.second_step(zero_grad=True)
                
                # Aggiorna EMA se attivo
                if using_ema:
                    self.ema.update()
                
                train_loss += batch_loss.item()
                num_batches += 1
            
            # ----- GESTIONE SCHEDULER E TRACKING -----
            
            # Attiva SWA dopo la fase iniziale
            if epoch >= self.swa_start and not using_swa:
                print(f"Epoch {epoch+1}: Attivazione SWA (Stochastic Weight Averaging)")
                using_swa = True
            
            # Gestione degli scheduler
            if using_swa:
                # Dopo l'inizio di SWA, usa lo scheduler SWA
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            elif epoch % 5 == 0:
                # Prima dell'inizio di SWA, usa cosine annealing ogni 5 epoche
                self.cosine_scheduler.step()
            
            # Salva il learning rate corrente
            current_lr = self.optimizer.base_optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            # Normalizza la loss di training
            train_loss /= num_batches
            history['train_loss'].append(train_loss)
            
            # ----- VALIDAZIONE -----
            
            if valid_loader is not None:
                # Valutazione sul validation set
                valid_loss = self.evaluate(valid_loader, mse_weight, mae_weight)
                history['valid_loss'].append(valid_loss)
                
                # Calcola e traccia il gap train/val come metrica di overfitting
                train_val_gap = valid_loss - train_loss
                history['train_val_gap'].append(train_val_gap)
                
                # Aggiorna lo scheduler principale basato sulla loss di validazione
                self.scheduler.step(valid_loss)
                
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Valid Loss: {valid_loss:.6f}, "
                      f"Gap: {train_val_gap:.6f}, "
                      f"LR: {current_lr:.8f}")
                
                # Tracciamento del modello migliore e early stopping
                improved = False
                
                # Usa EMA per validazione se attivo
                if using_ema:
                    # Salva pesi originali
                    self.ema.apply_shadow()
                    
                    # Valuta con EMA
                    ema_valid_loss = self.evaluate(valid_loader, mse_weight, mae_weight)
                    
                    # Ripristina pesi originali
                    self.ema.restore()
                    
                    print(f"  EMA Valid Loss: {ema_valid_loss:.6f} ({'migliore' if ema_valid_loss < valid_loss else 'peggiore'} del modello standard)")
                    
                    # Confronta con il miglior modello finora
                    if ema_valid_loss < best_valid_loss:
                        best_valid_loss = ema_valid_loss
                        
                        # Salva sia il modello EMA che quello standard
                        torch.save(self.model.state_dict(), 'best_cnn_model.pth')
                        self.best_model_state = self.model.state_dict().copy()
                        
                        # Salva anche i pesi EMA
                        self.ema.apply_shadow()
                        torch.save(self.model.state_dict(), 'best_cnn_model_ema.pth')
                        self.best_ema_state = self.model.state_dict().copy()
                        self.ema.restore()
                        
                        improved = True
                        early_stop_counter = 0
                        print(f"  Nuovo miglior modello EMA salvato (loss: {best_valid_loss:.6f})")
                else:
                    # Senza EMA, usa la validazione standard
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        
                        # Salva il modello migliore
                        torch.save(self.model.state_dict(), 'best_cnn_model.pth')
                        self.best_model_state = self.model.state_dict().copy()
                        
                        improved = True
                        early_stop_counter = 0
                        print(f"  Nuovo miglior modello salvato (loss: {best_valid_loss:.6f})")
                
                if not improved:
                    early_stop_counter += 1
                    print(f"  Nessun miglioramento per {early_stop_counter}/{early_stop_patience} epoche")
                
                # Early stopping con messaggio dettagliato
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping attivato dopo {epoch+1} epoche. "
                          f"Nessun miglioramento per {early_stop_patience} epoche consecutive.")
                    
                    # Ripristina il miglior modello per l'output finale
                    if using_ema and self.best_ema_state is not None:
                        self.model.load_state_dict(self.best_ema_state)
                        print(f"Ripristinato il miglior modello EMA (loss: {best_valid_loss:.6f})")
                    elif self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                        print(f"Ripristinato il miglior modello (loss: {best_valid_loss:.6f})")
                    
                    break
            else:
                # Se non c'è validation set, aggiorna lo scheduler con la loss di training
                self.scheduler.step(train_loss)
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"LR: {current_lr:.8f}")
            
            # Salva periodicamente il modello
            if (epoch + 1) % 100 == 0 or epoch + 1 == num_epochs:
                torch.save(self.model.state_dict(), f'cnn_model_epoch_{epoch+1}.pth')
                print(f"  Checkpoint salvato all'epoca {epoch+1}")
        
        # Statistiche finali del training
        end_time = time.time()
        training_minutes = (end_time - start_time) / 60
        training_hours = training_minutes / 60
        
        if training_hours >= 1:
            print(f"Training completato in {training_hours:.2f} ore ({training_minutes:.2f} minuti)")
        else:
            print(f"Training completato in {training_minutes:.2f} minuti")
        
        # Se è stato attivato SWA, aggiorna le statistiche di normalizzazione
        if using_swa:
            print("Aggiornamento delle statistiche BatchNorm per il modello SWA...")
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model)
            
            # Valuta il modello SWA
            self.model = self.swa_model
            if valid_loader is not None:
                swa_valid_loss = self.evaluate(valid_loader, mse_weight, mae_weight)
                print(f"SWA model validation loss: {swa_valid_loss:.6f}")
                
                # Se SWA è migliore del miglior modello finora, salvalo
                if swa_valid_loss < best_valid_loss:
                    torch.save(self.swa_model.state_dict(), 'best_cnn_model_swa.pth')
                    print(f"SWA model è il migliore, salvato (loss: {swa_valid_loss:.6f})")
        
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
    Applica tecniche di data augmentation estremamente avanzate e aggressive alle immagini di input.
    
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
    
    # --- GRUPPO 1: RUMORE E DISTURBI ---
    
    # Rumore gaussiano con probabilità noise_prob e sigma variabile
    if random.random() < noise_prob:
        # Intensità variabile del rumore
        noise_factor = random.uniform(0.5, 2.0)
        noise = torch.randn_like(augmented) * sigma * noise_factor
        augmented = augmented + noise
        augmented = torch.clamp(augmented, 0.0, 1.0)
    
    # Rumore spazialmente correlato (più realistico) con probabilità 0.3
    if random.random() < 0.3:
        # Generiamo un rumore di base a bassa risoluzione
        noise_base = torch.randn(batch_size, channels, 
                               max(height // 8, 4), 
                               max(width // 8, 4), 
                               device=augmented.device) * sigma * 2
        
        # Upsampling per creare correlazione spaziale
        noise_upsampled = torch.nn.functional.interpolate(
            noise_base, size=(height, width), mode='bilinear', align_corners=False)
        
        augmented = augmented + noise_upsampled
        augmented = torch.clamp(augmented, 0.0, 1.0)
    
    # Disturbi strutturati (linee, punti) con probabilità 0.2
    if random.random() < 0.2:
        for i in range(batch_size):
            # Decide il tipo di disturbo
            if random.random() < 0.5:  # Linee
                num_lines = random.randint(1, 3)
                for _ in range(num_lines):
                    thickness = random.randint(1, 3)
                    if random.random() < 0.5:  # Linea orizzontale
                        y_pos = random.randint(0, height - 1)
                        augmented[i, :, max(0, y_pos-thickness):min(height, y_pos+thickness), :] *= random.uniform(0.7, 1.3)
                    else:  # Linea verticale
                        x_pos = random.randint(0, width - 1)
                        augmented[i, :, :, max(0, x_pos-thickness):min(width, x_pos+thickness)] *= random.uniform(0.7, 1.3)
            else:  # Punti/macchie
                num_spots = random.randint(2, 5)
                for _ in range(num_spots):
                    spot_size = random.randint(3, 10)
                    y_pos = random.randint(0, height - spot_size)
                    x_pos = random.randint(0, width - spot_size)
                    
                    # Intensità del disturbo (amplifica o diminuisce)
                    factor = random.uniform(0.6, 1.4)
                    augmented[i, :, y_pos:y_pos+spot_size, x_pos:x_pos+spot_size] *= factor
    
    # --- GRUPPO 2: MASCHERE E OCCLUSIONI ---
    
    # Cutout multiplo con probabilità variabile
    if random.random() < cutout_prob:
        # Numero di cutout proporzionale alla dimensione dell'immagine
        num_cutouts = random.randint(1, 3)
        for _ in range(num_cutouts):
            # Dimensione variabile dei cutout (5-25% dell'immagine)
            cutout_size_h = int(height * random.uniform(0.05, 0.25))
            cutout_size_w = int(width * random.uniform(0.05, 0.25))
            
            # Per ogni immagine nel batch
            for i in range(batch_size):
                # Posizione casuale del cutout
                top = random.randint(0, height - cutout_size_h)
                left = random.randint(0, width - cutout_size_w)
                
                # Applica cutout con valore random invece di zero
                cutout_value = random.uniform(0, 0.3) if random.random() < 0.3 else 0.0
                augmented[i, :, top:top+cutout_size_h, left:left+cutout_size_w] = cutout_value
    
    # GridMask (maschere a griglia) con probabilità 0.2
    if random.random() < 0.2:
        grid_size = random.randint(8, max(16, min(height, width) // 8))
        for i in range(batch_size):
            mask = torch.ones_like(augmented[i])
            for y in range(0, height, grid_size * 2):
                for x in range(0, width, grid_size * 2):
                    h_end = min(y + grid_size, height)
                    w_end = min(x + grid_size, width)
                    if h_end > y and w_end > x:  # Verifica che il blocco sia valido
                        mask[:, y:h_end, x:w_end] = 0
            
            # Applica la griglia con valore random
            grid_value = random.uniform(0, 0.2) if random.random() < 0.5 else 0.0
            augmented[i] = augmented[i] * mask + grid_value * (1 - mask)
    
    # --- GRUPPO 3: TRASFORMAZIONI DI VALORI ---
    
    # Shift/bias globale dei valori con probabilità 0.4
    if random.random() < 0.4:
        shift = random.uniform(-0.1, 0.1)  # Shift più aggressivo
        augmented = augmented + shift
        augmented = torch.clamp(augmented, 0.0, 1.0)
    
    # Jitter di contrasto con probabilità 0.4
    if random.random() < 0.4:
        factor = random.uniform(0.7, 1.3)  # Fattore più aggressivo
        mean = augmented.mean(dim=[2, 3], keepdim=True)
        augmented = (augmented - mean) * factor + mean
        augmented = torch.clamp(augmented, 0.0, 1.0)
    
    # Inversione locale con probabilità 0.15
    if random.random() < 0.15:
        for i in range(batch_size):
            if random.random() < 0.5:  # Inversione regione
                # Seleziona una regione casuale
                region_h = random.randint(height // 4, height // 2)
                region_w = random.randint(width // 4, width // 2)
                top = random.randint(0, height - region_h)
                left = random.randint(0, width - region_w)
                
                # Inverti i valori nella regione (1-x)
                augmented[i, :, top:top+region_h, left:left+region_w] = 1.0 - augmented[i, :, top:top+region_h, left:left+region_w]
            else:  # Inversione a scacchiera
                checker_size = random.randint(5, 20)
                for y in range(0, height, checker_size * 2):
                    for x in range(0, width, checker_size * 2):
                        y_end = min(y + checker_size, height)
                        x_end = min(x + checker_size, width)
                        if y_end > y and x_end > x:
                            augmented[i, :, y:y_end, x:x_end] = 1.0 - augmented[i, :, y:y_end, x:x_end]
    
    # --- GRUPPO 4: TRASFORMAZIONI SPAZIALI ---
    
    # Traslazione locale con probabilità 0.2
    if random.random() < 0.2:
        # Seleziona una regione casuale da traslare
        region_size = int(min(height, width) * random.uniform(0.2, 0.4))
        top = random.randint(0, height - region_size)
        left = random.randint(0, width - region_size)
        
        # Direzione e intensità della traslazione
        shift_y = random.randint(-region_size//4, region_size//4)
        shift_x = random.randint(-region_size//4, region_size//4)
        
        # Applica la traslazione solo se non esce dai bordi
        if (top + shift_y >= 0 and top + region_size + shift_y <= height and
            left + shift_x >= 0 and left + region_size + shift_x <= width):
            for i in range(batch_size):
                # Salva la regione originale
                original = augmented[i, :, top:top+region_size, left:left+region_size].clone()
                
                # Cancella la regione originale
                augmented[i, :, top:top+region_size, left:left+region_size] = 0
                
                # Inserisci la regione nella nuova posizione
                augmented[i, :, top+shift_y:top+region_size+shift_y, 
                           left+shift_x:left+region_size+shift_x] = original
    
    # Distorsione elastica con probabilità 0.15
    if random.random() < 0.15:
        # Crea griglie di displacement con correlazione spaziale
        displacement_x = torch.randn(batch_size, 1, max(height//8, 4), max(width//8, 4), device=augmented.device) * 2
        displacement_y = torch.randn(batch_size, 1, max(height//8, 4), max(width//8, 4), device=augmented.device) * 2
        
        # Upsampling per ottenere displacement field smooth
        displacement_x = torch.nn.functional.interpolate(displacement_x, size=(height, width), mode='bilinear', align_corners=False)
        displacement_y = torch.nn.functional.interpolate(displacement_y, size=(height, width), mode='bilinear', align_corners=False)
        
        # Crea griglia di coordinate base
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=augmented.device),
            torch.linspace(-1, 1, width, device=augmented.device)
        )
        
        # Espandi le dimensioni per il batch
        grid_x = grid_x.unsqueeze(0).repeat(batch_size, 1, 1)
        grid_y = grid_y.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Aggiungi displacement e normalizza
        scale = 0.05  # Intensità della distorsione
        grid_x = grid_x + displacement_x.squeeze(1) * scale
        grid_y = grid_y + displacement_y.squeeze(1) * scale
        
        # Assicurati che i valori della griglia siano nel range [-1, 1]
        grid_x = torch.clamp(grid_x, -1, 1)
        grid_y = torch.clamp(grid_y, -1, 1)
        
        # Combina in griglia di sampling finale
        grid = torch.stack([grid_x, grid_y], dim=3)
        
        # Applica la griglia con grid_sample
        augmented = torch.nn.functional.grid_sample(
            augmented, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )
    
    return augmented

def manifold_mixup(inputs1, inputs2, features1=None, features2=None, alpha=0.2):
    """
    Implementa Manifold Mixup per la regolarizzazione nello spazio latente
    
    Args:
        inputs1, inputs2: Tensori di input da mixare
        features1, features2: Features latenti opzionali
        alpha: Parametro per la distribuzione beta
        
    Returns:
        Mixed inputs e/o features con lambda
    """
    device = inputs1.device
    
    # Genera lambda dalla distribuzione beta
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
    batch_size = inputs1.size(0)
    
    # Mixup degli input
    mixed_inputs = lam * inputs1 + (1 - lam) * inputs2
    
    # Mixup delle features se fornite
    mixed_features = None
    if features1 is not None and features2 is not None:
        mixed_features = lam * features1 + (1 - lam) * features2
    
    return mixed_inputs, mixed_features, lam

class EMA:
    """Exponential Moving Average per parametri del modello.
    
    Mantiene una versione "media" dei pesi del modello che tende a produrre
    predizioni più stabili e generalizzabili rispetto ai pesi istantanei.
    """
    def __init__(self, model, decay=0.999):
        """
        Args:
            model: Il modello di cui mantenere l'EMA
            decay: Fattore di decadimento (più alto = aggiornamenti più lenti)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Registra i parametri iniziali
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Aggiorna l'EMA con i pesi correnti del modello."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Sostituisce i parametri del modello con quelli dell'EMA."""
        self.backup = {}  # Inizializza il backup
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()  # Salva i parametri originali
                param.data = self.shadow[name]
    
    def restore(self):
        """Ripristina i parametri originali del modello."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class SelfDistillationLoss(nn.Module):
    """Loss di Self-Distillation per il training con consistency regularization.
    
    Combina la loss standard con una loss KL-divergence tra le predizioni
    del modello e le predizioni del modello EMA, incentivando la consistenza.
    """
    def __init__(self, T=2.0):
        """
        Args:
            T: Temperatura per la distillation
        """
        super(SelfDistillationLoss, self).__init__()
        self.T = T  # Temperatura (più alta = probabilità più smooth)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, student_preds, teacher_preds, targets, weights=None):
        """
        Args:
            student_preds: Predizioni del modello normale
            teacher_preds: Predizioni del modello EMA (teacher)
            targets: Target reali
            weights: Pesi opzionali per le diverse loss
        
        Returns:
            Loss combinata
        """
        if weights is None:
            # Pesi di default: 60% MSE, 20% MAE, 20% distillation
            weights = {'mse': 0.6, 'mae': 0.2, 'distill': 0.2}
        
        # Calcola loss primaria rispetto ai target reali
        task_loss = weights['mse'] * self.mse(student_preds, targets) + \
                    weights['mae'] * self.mae(student_preds, targets)
        
        # Calcola loss di distillation rispetto al teacher
        # Per l'equazione del calore usiamo direttamente MSE invece di KL-div
        distill_loss = weights['distill'] * self.mse(
            student_preds / self.T, 
            teacher_preds.detach() / self.T
        ) * (self.T ** 2)  # Ri-scala per bilanciare l'effetto della temperatura
        
        return task_loss + distill_loss

class FeatureAlignmentLoss(nn.Module):
    """Regolarizzazione tramite allineamento delle feature tra teacher e student model."""
    def __init__(self):
        super(FeatureAlignmentLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: Feature del modello normale
            teacher_features: Feature del modello EMA
            
        Returns:
            Loss di allineamento delle feature
        """
        total_loss = 0.0
        n_features = len(student_features)
        
        # Calcola la loss per ogni livello di feature
        for sf, tf in zip(student_features, teacher_features):
            # Normalizzazione L2 per confrontare solo i pattern, non le magnitudini
            sf_norm = F.normalize(sf, dim=1)
            tf_norm = F.normalize(tf.detach(), dim=1)
            total_loss += self.mse(sf_norm, tf_norm)
        
        return total_loss / n_features
