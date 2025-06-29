# src/main.py
import argparse
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import inspect
import os.path
import shutil
from datetime import datetime

from .fem_solver import FEMSolver
from .pinn_solver import PINNSolver, PINNTrainer
from .cnn_solver import CNNSolver, CNNTrainer
from .deepritz_solver import DeepRitzSolver, DeepRitzTrainer
from .plotting import create_single_frame, create_video_from_frames

# --- Parametri Globali ---
# Parametri fisici
SIGMA_H = 9.5298e-4 # Diffusività
A = 18.515 # Costante di diffusione
FR = 0.0 # Forza di stimolo
FT = 0.2383 # Forza di stimolo temporale
FD = 1.0 # Forza di stimolo spaziale

# Parametri di simulazione
T = 35.0
DT = 0.1
NVX = NVY = 101

# Casi predefiniti
CASE_NORMAL = "Normal_Diffusivity_1x"
CASE_LOW = "Low_Diffusivity_01x"
CASE_HIGH = "High_Diffusivity_10x"



# --- Configurazione dispositivo ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Usando dispositivo: {DEVICE}")

# For DeepRitz, use CPU if MPS due to autograd limitations
DEEPRITZ_DEVICE = 'cpu' if DEVICE == 'mps' else DEVICE
if DEVICE == 'mps':
    print("Nota: DeepRitz utilizzerà CPU invece di MPS per evitare limitazioni di autograd.")

# --- Funzioni di Esecuzione ---

def run_fem_simulation(case_name, sigma_d_factor):
    """Esegue una singola simulazione FEM e genera i frame/video."""
    output_dir = os.path.join('assets', 'fem')
    frame_dir = os.path.join(output_dir, case_name, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    
    # Crea il solver e calcola la soluzione
    solver = FEMSolver(nvx=NVX, nvy=NVY, sigma_h=SIGMA_H, a=A, fr=FR, ft=FT, fd=FD)
    
    print(f"Esecuzione simulazione FEM per il caso: {case_name}")
    solution_data = solver.compute_solution(
        T=T, 
        dt=DT, 
        sigma_d_factor=sigma_d_factor
    )
    
    # Genera i frame dalle soluzioni calcolate
    print(f"Generazione dei frame per il caso {case_name}...")
    times = solution_data['times']
    solutions = solution_data['solutions']
    
    from .plotting import create_single_frame
    for i, (t, solution) in enumerate(zip(times, solutions)):
        # Appiattire la soluzione per create_single_frame
        u_flat = solution.flatten(order='F')
        
        fig = create_single_frame(u_flat, NVX, NVY, t, case_name)
        plt.savefig(os.path.join(frame_dir, f'frame_{i:04d}.png'))
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{len(times)} generato.")
    
    print(f"  {len(times)} frame salvati in: {frame_dir}")
    
    # Crea il video dai frame
    create_video_from_frames(frame_dir, case_name, fps=10)

def train_pinn_model(model_name=None):
    """
    Addestra il modello PINN e salva modello, grafico delle loss e script in una cartella dedicata.
    
    Args:
        model_name (str, optional): Nome del modello da usare. Se None, verrà generato automaticamente.
    """
    # Crea il nome del modello se non specificato
    if model_name is None:
        timestamp = time.strftime("%H%M%S")
        model_name = f"pinn_model_{timestamp}"
    
    # Crea directory per il modello specifico
    models_base_dir = 'models'
    model_dir = os.path.join(models_base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Definisci percorsi dei file
    model_path = os.path.join(model_dir, 'model_weights.pth')
    loss_plot_path = os.path.join(model_dir, 'training_loss.png')
    script_copy_path = os.path.join(model_dir, 'pinn_solver_script.py')
    
    # Otteniamo il percorso del modulo corrente
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Costruiamo il percorso relativo al file pinn_solver.py
    pinn_script_path = os.path.join(current_dir, 'pinn_solver.py')
    
    import shutil
    shutil.copy2(pinn_script_path, script_copy_path)
    print(f"Script PINN salvato in: {script_copy_path}")
    
    # Crea e addestra il modello
    pinn = PINNSolver(
        device=DEVICE,
        sigma_h=SIGMA_H,
        a=A,
        fr=FR,
        ft=FT,
        fd=FD
    ).to(DEVICE)
    
    trainer = PINNTrainer(pinn, device=DEVICE, T=T)
    
    # Otteniamo l'history del training
    history = trainer.train(n_epochs=10000, n_points_pde=4096, n_points_ic=1024)
    
    # Salva il modello addestrato
    torch.save(pinn.state_dict(), model_path)
    print(f"Modello PINN salvato in: {model_path}")
    
    # Crea e salva il grafico delle loss
    plt.figure(figsize=(12, 8))
    
    # Subplot per le loss
    plt.subplot(2, 1, 1)
    plt.semilogy(history['epochs'], history['total_loss'], 'b-', label='Loss Totale')
    plt.semilogy(history['epochs'], history['pde_loss'], 'r--', label='Loss PDE')
    plt.semilogy(history['epochs'], history['ic_loss'], 'g-.', label='Loss IC')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Epoche')
    plt.ylabel('Loss (scala log)')
    plt.legend()
    plt.title('Andamento delle Loss Durante il Training')
    
    # Subplot per il learning rate
    plt.subplot(2, 1, 2)
    plt.semilogy(history['epochs'], history['learning_rate'], 'k-')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Epoche')
    plt.ylabel('Learning Rate (scala log)')
    plt.title('Andamento del Learning Rate')
    
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"Grafico delle loss salvato in: {loss_plot_path}")
    plt.close()
    
    # Salviamo solo i file essenziali: modello, grafico delle loss e script
    print(f"Training completato con loss finale: {history['total_loss'][-1]:.4e}")
    
    return model_name

def generate_pinn_frames(case='normal', model_name=None):
    """
    Carica un modello PINN addestrato e genera i frame/video.
    
    Args:
        case (str): Caso di diffusività da simulare ('high', 'normal', 'low').
        model_name (str, optional): Nome del modello da caricare. Se None, usa l'ultimo modello disponibile.
    """
    # Individua il percorso del modello da caricare
    if model_name is None:
        # Se non specificato, usa il modello predefinito o cerca l'ultimo
        models_dir = 'models'
        available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        if available_models:
            # Ordina per timestamp (assumendo il formato nome_YYYYMMDD_HHMMSS)
            available_models.sort(reverse=True)
            model_name = available_models[0]
            print(f"Modello non specificato. Uso il più recente: {model_name}")
        else:
            model_path = os.path.join(models_dir, 'best_pinn_model.pth')
            if os.path.exists(model_path):
                print("Usando il modello predefinito 'best_pinn_model.pth'")
            else:
                print(f"Errore: Nessun modello trovato. Esegui prima il training.")
                return
    
    # Costruisci il percorso completo del modello
    if model_name:
        model_dir = os.path.join('models', model_name)
        model_path = os.path.join(model_dir, 'model_weights.pth')
    else:
        model_path = os.path.join('models', 'best_pinn_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato in {model_path}. Esegui prima il training o verifica il nome del modello.")
        return

    # Calcoliamo sigma_h in base al caso
    if case == 'high':
        sigma_h_value = SIGMA_H * 10.0
        case_display = 'High_Diffusivity_10x'
    elif case == 'low':
        sigma_h_value = SIGMA_H * 0.1
        case_display = 'Low_Diffusivity_01x'
    else:  # normal
        sigma_h_value = SIGMA_H
        case_display = 'Normal_Diffusivity_1x'
    
    # Inizializza il modello PINN
    model = PINNSolver(
        device=DEVICE,
        sigma_h=sigma_h_value,
        a=A, 
        fr=FR, 
        ft=FT, 
        fd=FD
    ).to(DEVICE)
    
    # Carica i pesi pre-addestrati
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Organizza i risultati per modello e caso di diffusività
    # Crea una directory principale per il modello
    model_name_safe = model_name if model_name else "default_model"
    output_dir = os.path.join('assets', 'pinn', model_name_safe)
    
    # Sotto-directory per il caso specifico di diffusività
    case_name = f"PINN_Prediction_{case_display}"
    frames_dir = os.path.join(output_dir, case_name, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Risultati saranno salvati in: {os.path.dirname(frames_dir)}")

    # Calcola la soluzione utilizzando il metodo compute_solution
    print(f"Calcolo della soluzione PINN per il caso: {case_name}")
    solution_data = model.compute_solution(
        T=T,
        nvx=NVX,
        nvy=NVY,
        num_frames=100
    )
    
    # Genera i frame dalle soluzioni calcolate
    print(f"Generazione dei frame per il caso {case_name}...")
    times = solution_data['times']
    solutions = solution_data['solutions']
    
    from .plotting import create_single_frame
    for i, (t, solution) in enumerate(zip(times, solutions)):
        # Appiattire la soluzione per create_single_frame
        u_flat = solution.flatten(order='F')
        
        fig = create_single_frame(u_flat, NVX, NVY, t, case_name)
        plt.savefig(os.path.join(frames_dir, f'frame_{i:04d}.png'))
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{len(times)} generato.")
    
    print(f"  {len(times)} frame salvati in: {frames_dir}")
    
    # Crea il video dai frame
    create_video_from_frames(frames_dir, case_name, fps=10)
            
    return frames_dir

def train_cnn_model(model_name=None):
    """
    Addestra il modello CNN avanzato usando i dati generati dal FEM solver e salva il modello addestrato.
    
    Args:
        model_name (str, optional): Nome del modello da usare. Se None, verrà generato automaticamente.
    """
    # Crea il nome del modello se non specificato
    if model_name is None:
        timestamp = datetime.now().strftime('%H%M%S')
        model_name = f"cnn_model_{timestamp}"
    
    # Crea directory per il modello specifico
    models_base_dir = 'models'
    model_dir = os.path.join(models_base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Definisci percorsi dei file
    model_path = os.path.join(model_dir, 'model_weights.pth')
    loss_plot_path = os.path.join(model_dir, 'training_loss.png')
    script_copy_path = os.path.join(model_dir, 'cnn_solver_script.py')
    
    # Otteniamo il percorso del modulo corrente
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Costruiamo il percorso relativo al file cnn_solver.py
    cnn_script_path = os.path.join(current_dir, 'cnn_solver.py')
    shutil.copy2(cnn_script_path, script_copy_path)
    print(f"Script CNN salvato in: {script_copy_path}")
    
    # Crea il modello con latent_dim aumentato per maggiore capacità
    print("Inizializzazione del modello CNN avanzato...")
    cnn_model = CNNSolver(
        device=DEVICE,
        prediction_steps=5,  # Predizione multistep
        latent_dim=128,      # Dimensione spazio latente aumentata
        dropout_rate=0.2     # Dropout per regolarizzazione
    ).to(DEVICE)
    
    # Crea il trainer con iperparametri ottimizzati
    trainer = CNNTrainer(
        cnn_model,
        learning_rate=5e-4,  # Learning rate iniziale aumentato
        device=DEVICE,
        weight_decay=2e-5    # Regolarizzazione L2 ottimizzata
    )
    
    # Genera dati di training con FEM solver per diversi valori di diffusività
    print("Inizializzazione FEM Solver per generazione dati di training...")
    fem_solver = FEMSolver(nvx=NVX, nvy=NVY, sigma_h=SIGMA_H, a=A, fr=FR, ft=FT, fd=FD)
    
    # Usa diversi valori di sigma per il dataset (includendo più valori per robustezza)
    sigma_values = [9.5298e-5, 4.7649e-4, 9.5298e-4, 4.7649e-3, 9.5298e-3]  # 0.1x, 0.5x, 1x, 5x, 10x
    
    # Genera i dati di training con più campioni
    print("Generazione dati di training dal FEM solver...")
    inputs, targets, sigmas = trainer.generate_training_data_from_fem(
        fem_solver, sigma_values, T=T, dt=DT, num_samples=500  # Aumentato numero di campioni
    )
    
    # Converti in dataset PyTorch
    from torch.utils.data import TensorDataset, DataLoader
    
    # Prepara i tensori per il dataloader
    input_tensor = torch.stack(inputs)
    sigma_tensor = torch.tensor(sigmas, dtype=torch.float32).view(-1, 1)
    
    # Creiamo una lista di liste di tensori target
    target_tensors = []
    for i in range(len(targets[0])):
        target_tensors.append(torch.stack([t[i] for t in targets]))
    
    # Dividi in training e validation (80% train, 20% validation)
    n_samples = len(inputs)
    train_idx = int(0.8 * n_samples)
    
    # Creazione dei dataset di train e validation
    train_dataset = (input_tensor[:train_idx], 
                     [t[:train_idx] for t in target_tensors], 
                     sigma_tensor[:train_idx])
    val_dataset = (input_tensor[train_idx:], 
                   [t[train_idx:] for t in target_tensors], 
                   sigma_tensor[train_idx:])
    
    # Crea i dataloader con parametri ottimizzati
    # Batch size ridotto per evitare problemi di memoria con architettura più grande
    train_loader = DataLoader(
        dataset=list(zip(train_dataset[0], train_dataset[1], train_dataset[2])),
        batch_size=4,
        shuffle=True,
        pin_memory=True if DEVICE == 'cuda' else False  # Ottimizzazione per GPU
    )
    val_loader = DataLoader(
        dataset=list(zip(val_dataset[0], val_dataset[1], val_dataset[2])),
        batch_size=8,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    # Training con 300 epoche per convergenza profonda
    print(f"Avvio del training CNN con modello avanzato ({model_name})...")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=1000,             # Triplicato il numero di epoche
        mse_weight=0.6,             # Peso della MSE loss
        mae_weight=0.4,             # Peso della MAE loss
        early_stop_patience=1000      # Patience aumentata per consentire plateau nel training
    )
    
    # Salva il modello addestrato
    torch.save(cnn_model.state_dict(), model_path)
    print(f"Modello CNN avanzato salvato in: {model_path}")
    
    # Salva il grafico dettagliato delle loss
    trainer.plot_loss(history, model_dir)
    
    # Memorizza l'ultima best loss per il report
    if 'valid_loss' in history and history['valid_loss']:
        best_loss = min(history['valid_loss'])
        with open(os.path.join(model_dir, 'training_info.txt'), 'w') as f:
            f.write(f"Modello: {model_name}\n")
            f.write(f"Latent dimension: {cnn_model.latent_dim}\n")
            f.write(f"Prediction steps: {cnn_model.prediction_steps}\n")
            f.write(f"Epoche di training: {len(history['train_loss'])}\n")
            f.write(f"Best validation loss: {best_loss:.6f}\n")
            f.write(f"Final training loss: {history['train_loss'][-1]:.6f}\n")
    
    return model_name

def generate_cnn_frames(case='normal', model_name=None):
    """
    Carica un modello CNN addestrato e genera i frame/video.
    
    Args:
        case (str): Caso di diffusività da simulare ('high', 'normal', 'low').
        model_name (str, optional): Nome del modello da caricare. Se None, usa l'ultimo modello disponibile.
    """
    # Individua il percorso del modello da caricare
    if model_name is None:
        # Trova l'ultimo modello CNN creato
        models_dir = 'models'
        cnn_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) 
                      and d.startswith('cnn_model_')]
        if not cnn_models:
            print("Errore: Nessun modello CNN trovato.")
            return
        cnn_models.sort(reverse=True)  # Ordina in ordine decrescente per ottenere il più recente
        model_name = cnn_models[0]  # Prendi l'ultimo modello creato
        print(f"Utilizzo dell'ultimo modello CNN: {model_name}")
    
    # Costruisci il percorso completo del modello
    model_path = os.path.join('models', model_name, 'model_weights.pth')
    
    if not os.path.exists(model_path):
        print(f"Errore: File del modello {model_path} non trovato.")
        return
    
    # Calcoliamo sigma_h in base al caso
    if case == 'high':
        sigma_h_value = SIGMA_H * 10.0
        case_display = "High_Diffusivity_10x"
    elif case == 'low':
        sigma_h_value = SIGMA_H * 0.1
        case_display = "Low_Diffusivity_01x"
    else:  # 'normal' o altri valori
        sigma_h_value = SIGMA_H
        case_display = "Normal_Diffusivity_1x"
    
    print(f"Generazione predizione CNN per {case_display} con sigma_h={sigma_h_value:.8e}")
    
    # Verifica se esiste un file di configurazione del modello per ottenere i parametri corretti
    model_info_path = os.path.join('models', model_name, 'training_info.txt')
    latent_dim = 128  # Default alla nuova dimensione latente
    
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, 'r') as f:
                for line in f:
                    if 'Latent dimension:' in line:
                        latent_dim = int(line.split(':')[1].strip())
                        break
        except:
            print(f"Impossibile leggere latent_dim dal file di configurazione, uso valore predefinito: {latent_dim}")
    
    print(f"Inizializzazione modello CNN con latent_dim={latent_dim}")
    
    # Inizializza il modello CNN con le stesse dimensioni usate durante il training
    model = CNNSolver(
        device=DEVICE, 
        latent_dim=latent_dim,  # Utilizza il valore letto o il default
        nvx=NVX,
        nvy=NVY
    ).to(DEVICE)
    
    # Carica i pesi pre-addestrati
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Organizza i risultati per modello e caso di diffusività
    # Crea una directory principale per il modello
    model_name_safe = model_name if model_name else "default_model"
    output_dir = os.path.join('assets', 'cnn', model_name_safe)
    
    # Sotto-directory per il caso specifico di diffusività
    case_name = f"CNN_Prediction_{case_display}"
    frames_dir = os.path.join(output_dir, case_name, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Risultati saranno salvati in: {os.path.dirname(frames_dir)}")
    
    # Calcola la soluzione utilizzando il metodo compute_solution
    print(f"Calcolo della soluzione CNN per il caso: {case_name}")
    solution_data = model.compute_solution(
        T=T,
        nvx=NVX,
        nvy=NVY,
        sigma_h=sigma_h_value,
        num_frames=100
    )
    
    # Genera i frame dalle soluzioni calcolate
    print(f"Generazione dei frame per il caso {case_name}...")
    times = solution_data['times']
    solutions = solution_data['solutions']
    
    from .plotting import create_single_frame
    for i, (t, solution) in enumerate(zip(times, solutions)):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        fig = create_single_frame(solution.flatten(), NVX, NVY, t, case_name)
        plt.savefig(frame_path, dpi=150)  # Aumenta la risoluzione
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{len(times)} generato.")
    
    print(f"  {len(times)} frame salvati in: {frames_dir}")
    
    # Crea il video dai frame
    create_video_from_frames(frames_dir, case_name, fps=10)
    
    # Salva anche i dati numerici della soluzione per eventuali analisi successive
    solution_data_path = os.path.join(output_dir, case_name, 'solution_data.npz')
    np.savez(
        solution_data_path,
        x=solution_data['x'],
        y=solution_data['y'],
        times=solution_data['times'],
        solutions=np.array(solution_data['solutions'])
    )
    print(f"Dati della soluzione salvati in: {solution_data_path}")
            
    return frames_dir

def train_deepritz_model(model_name=None):
    """
    Addestra il modello DeepRitz e salva modello, grafico delle loss e script in una cartella dedicata.
    
    Args:
        model_name (str, optional): Nome del modello da usare. Se None, verrà generato automaticamente.
    """
    # Crea il nome del modello se non specificato
    if model_name is None:
        timestamp = time.strftime("%H%M%S")
        model_name = f"deepritz_model_{timestamp}"
    
    # Crea directory per il modello specifico
    models_base_dir = 'models'
    model_dir = os.path.join(models_base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Definisci percorsi dei file
    model_path = os.path.join(model_dir, 'model_weights.pth')
    loss_plot_path = os.path.join(model_dir, 'training_loss.png')
    script_copy_path = os.path.join(model_dir, 'deepritz_solver_script.py')
    
    # Otteniamo il percorso del modulo corrente
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Costruiamo il percorso relativo al file deepritz_solver.py
    deepritz_script_path = os.path.join(current_dir, 'deepritz_solver.py')
    
    import shutil
    shutil.copy2(deepritz_script_path, script_copy_path)
    print(f"Script DeepRitz salvato in: {script_copy_path}")
    
    # Crea e addestra il modello
    deepritz = DeepRitzSolver(
        device=DEEPRITZ_DEVICE,
        sigma_h=SIGMA_H,
        a=A,
        fr=FR,
        ft=FT,
        fd=FD
    ).to(DEEPRITZ_DEVICE)
    
    trainer = DeepRitzTrainer(deepritz, device=DEEPRITZ_DEVICE)
    
    # Addestra il modello
    trainer.train(epochs=10000, lr=5e-4, n_domain=4000, n_boundary=800, n_initial=800, T=T)
    
    # Salva il modello usando il trainer
    trainer.save_model(model_dir)
    
    print(f"Training DeepRitz completato. Modello salvato in: {model_dir}")
    
    return model_name

def generate_deepritz_frames(case='normal', model_name=None):
    """
    Carica un modello DeepRitz addestrato e genera i frame/video.
    
    Args:
        case (str): Caso di diffusività da simulare ('high', 'normal', 'low').
        model_name (str, optional): Nome del modello da caricare. Se None, usa l'ultimo modello disponibile.
    """
    # Mappa dei casi con i valori di diffusività e nomi display
    case_map = {
        'high': (10.0, 'High_Diffusivity_10x'),
        'normal': (1.0, 'Normal_Diffusivity_1x'),
        'low': (0.1, 'Low_Diffusivity_01x')
    }
    
    if case not in case_map:
        print(f"Errore: Caso '{case}' non riconosciuto. Usa 'high', 'normal', o 'low'.")
        return
        
    sigma_factor, case_display = case_map[case]
    
    # Individua il percorso del modello da caricare
    if model_name is None:
        # Se non specificato, cerca l'ultimo modello DeepRitz disponibile
        models_dir = 'models'
        if os.path.exists(models_dir):
            available_models = [d for d in os.listdir(models_dir) 
                              if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('deepritz_model_')]
            if available_models:
                # Ordina per timestamp
                available_models.sort(reverse=True)
                model_name = available_models[0]
                print(f"Modello non specificato. Uso il più recente: {model_name}")
            else:
                print(f"Errore: Nessun modello DeepRitz trovato. Esegui prima il training con 'deepritz-train'.")
                return
        else:
            print(f"Errore: Directory models non trovata. Esegui prima il training.")
            return
    
    # Costruisci il percorso completo del modello
    model_dir = os.path.join('models', model_name)
    model_path = os.path.join(model_dir, 'model_weights.pth')
    
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato in {model_path}. Esegui prima il training o verifica il nome del modello.")
        return
    
    print(f"Caricamento modello DeepRitz da: {model_path}")
    
    # Crea e carica il modello DeepRitz
    deepritz = DeepRitzSolver(
        device=DEVICE,
        sigma_h=SIGMA_H * sigma_factor,  # Applica il fattore di diffusività
        a=A,
        fr=FR,
        ft=FT,
        fd=FD
    ).to(DEVICE)
    
    # Carica i pesi pre-addestrati
    deepritz.load_state_dict(torch.load(model_path, map_location=DEVICE))
    deepritz.eval()
    
    # Organizza i risultati per modello e caso di diffusività
    # Crea una directory principale per il modello
    output_dir = os.path.join('assets', 'deepritz', model_name)
    
    # Sotto-directory per il caso specifico di diffusività
    case_name = f"DeepRitz_Prediction_{case_display}"
    frames_dir = os.path.join(output_dir, case_name, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Risultati saranno salvati in: {os.path.dirname(frames_dir)}")
    
    # Calcola la soluzione utilizzando il metodo compute_solution
    print(f"Calcolo della soluzione DeepRitz per il caso: {case_name}")
    solution_data = deepritz.compute_solution(
        T=T,
        nvx=NVX,
        nvy=NVY,
        num_frames=100
    )
    
    # Genera i frame dalle soluzioni calcolate
    print(f"Generazione dei frame per il caso {case_name}...")
    times = solution_data['times']
    solutions = solution_data['solutions']
    
    from .plotting import create_single_frame
    for i, (t, solution) in enumerate(zip(times, solutions)):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        fig = create_single_frame(solution.flatten(), NVX, NVY, t, case_name)
        plt.savefig(frame_path, dpi=150)
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{len(times)} generato.")
    
    print(f"  {len(times)} frame salvati in: {frames_dir}")
    
    # Crea il video dai frame
    create_video_from_frames(frames_dir, case_name, fps=10)
    
    # Salva anche i dati numerici della soluzione per eventuali analisi successive
    solution_data_path = os.path.join(output_dir, case_name, 'solution_data.npz')
    np.savez(
        solution_data_path,
        x=solution_data['x'],
        y=solution_data['y'],
        times=solution_data['times'],
        solutions=np.array(solution_data['solutions'])
    )
    print(f"Dati della soluzione salvati in: {solution_data_path}")
            
    return frames_dir

def solve(solver_type, case="Normal_Diffusivity_1x", model_name=None, n_epochs=20000, num_frames=21):
    """
    Risolve l'equazione del monodominio utilizzando il solver specificato.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzo dispositivo: {device}")
    
    # Parametri fisici basati sul caso
    sigma_h, a, fr, ft, fd = get_physics_params(case)
    print(f"Parametri per {case}: sigma_h={sigma_h}, a={a}, fr={fr}, ft={ft}, fd={fd}")
    
    if solver_type == 'fem':
        print("Utilizzo solver FEM...")
    elif solver_type == 'cnn':
        print("Utilizzo solver CNN...")
        
        if model_name is None:
            # Crea un nuovo modello
            timestamp = datetime.now().strftime('%H%M%S')
            model_name = f"cnn_model_{timestamp}"
            print(f"Creazione nuovo modello CNN: {model_name}")
            
            # Crea la directory per salvare il modello
            model_dir = create_model_dir(model_name)
            
            # Inizializza il modello CNN
            cnn_model = CNNSolver(device=device, prediction_steps=5, latent_dim=64)
            trainer = CNNTrainer(cnn_model, learning_rate=1e-4, device=device)
            
            # Genera dati di training
            fem_solver = FEMSolver(nvx=NVX, nvy=NVY, sigma_h=sigma_h, a=a, fr=fr, ft=ft, fd=fd)
            inputs, targets, sigmas = trainer.generate_training_data_from_fem(
                fem_solver, [sigma_h*0.1, sigma_h, sigma_h*10], T=35.0, dt=0.1, num_samples=300
            )
            
            # Converti in dataset PyTorch
            from torch.utils.data import TensorDataset, DataLoader
            input_tensor = torch.stack(inputs)
            sigma_tensor = torch.tensor(sigmas, dtype=torch.float32).view(-1, 1)
            
            target_tensors = []
            for i in range(len(targets[0])):
                target_tensors.append(torch.stack([t[i] for t in targets]))
            
            # Dividi in training e validation
            n_samples = len(inputs)
            train_idx = int(0.8 * n_samples)
            
            train_dataset = (input_tensor[:train_idx], 
                             [t[:train_idx] for t in target_tensors], 
                             sigma_tensor[:train_idx])
            val_dataset = (input_tensor[train_idx:], 
                           [t[train_idx:] for t in target_tensors], 
                           sigma_tensor[train_idx:])
            
            # Crea i dataloader
            train_loader = DataLoader(
                dataset=list(zip(train_dataset[0], train_dataset[1], train_dataset[2])),
                batch_size=16,
                shuffle=True
            )
            val_loader = DataLoader(
                dataset=list(zip(val_dataset[0], val_dataset[1], val_dataset[2])),
                batch_size=16
            )
            
            # Addestramento
            history = trainer.train(train_loader, val_loader, num_epochs=100)
            
            # Salva il modello e gli altri artefatti
            model_path = os.path.join(model_dir, 'model_weights.pth')
            torch.save(cnn_model.state_dict(), model_path)
            print(f"Modello CNN salvato in: {model_path}")
            
            # Salva la loss
            trainer.plot_loss(history, model_dir)
            
            # Copia il file script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cnn_script_path = os.path.join(current_dir, 'cnn_solver.py')
            script_copy_path = os.path.join(model_dir, 'cnn_solver_script.py')
            shutil.copy2(cnn_script_path, script_copy_path)
            
            # Esegui la predizione con il modello appena addestrato
            solution_data = cnn_model.compute_solution(
                T=35.0, nvx=NVX, nvy=NVY, sigma_h=sigma_h, num_frames=num_frames
            )
        else:
            # Carica un modello esistente
            model_dir = os.path.join("models", model_name)
            if not os.path.exists(model_dir):
                print(f"Errore: Il modello {model_name} non esiste.")
                return
            
            # Inizializza il modello CNN
            cnn_model = CNNSolver(device=device).to(device)
            
            # Carica i pesi pre-addestrati
            model_path = os.path.join(model_dir, 'model_weights.pth')
            cnn_model.load_state_dict(torch.load(model_path, map_location=device))
            cnn_model.eval()
            
            # Esegui la predizione
            solution_data = cnn_model.compute_solution(
                T=35.0, nvx=NVX, nvy=NVY, sigma_h=sigma_h, num_frames=num_frames
            )
        
        save_solution(solution_data, solver_type, case, model_name)
    
    elif solver_type == 'pinn':
        print("Utilizzo solver PINN...")
        
        # Aggiorna l'architettura del modello: più profonda e più ampia
        pinn_layers = [3, 128, 128, 128, 128, 64, 1]  # Architettura più complessa
        
        pinn_solver = PINNSolver(device, sigma_h, a, fr, ft, fd, layers=pinn_layers)
        print(f"Architettura PINN: {pinn_layers}")
        
        # Crea il trainer con un learning rate appropriato
        trainer = PINNTrainer(pinn_solver, learning_rate=3e-4, device=device, T=35.0)
        
        # Se model_name non è fornito, crea un nome con timestamp
        if model_name is None:
            timestamp = time.strftime("%H%M%S")
            model_name = f"pinn_model_{timestamp}"
        
        # Crea la directory per salvare il modello
        model_dir = create_model_dir(model_name)
        
        # Training con più punti di collocation
        n_points_pde = 20000  # Aumentato per una migliore copertura del dominio
        n_points_ic = 5000    # Aumentato per una migliore rappresentazione della IC
        
        # Training del modello
        history = trainer.train(n_epochs, n_points_pde, n_points_ic)
        
        # Plot e salvataggio della loss
        trainer.plot_loss(history, model_dir)
        
        # Salvataggio del modello
        save_model(pinn_solver, model_dir)
        
        # Copia del file di script
        copy_script_to_model_dir(model_dir)
        
    elif solver_type == 'pinn_predict':
        if model_name is None:
            print("Errore: È necessario specificare model_name per la predizione con PINN")
            return
        
        # Carica il modello
        model_dir = os.path.join("models", model_name)
        if not os.path.exists(model_dir):
            print(f"Errore: Il modello {model_name} non esiste.")
            return
        
        # Ricrea il modello con gli stessi parametri fisici
        # Utilizziamo un'architettura più complessa se è un modello creato con questa versione
        if is_newer_model(model_name):
            pinn_layers = [3, 128, 128, 128, 128, 64, 1]
        else:
            pinn_layers = [3, 64, 64, 64, 1]  # Architettura originale
            
        pinn_solver = PINNSolver(device, sigma_h, a, fr, ft, fd, layers=pinn_layers)
        load_model(pinn_solver, model_dir)
        
        # Predizione
        solution_data = pinn_solver.predict(num_frames=num_frames)
        save_solution(solution_data, "pinn", case, model_name)
    
    elif solver_type == 'cnn':
        print("Utilizzo solver CNN...")
        
        # Crea il modello CNN
        cnn_model = CNNSolver(device=DEVICE, prediction_steps=5, latent_dim=64).to(DEVICE)
        
        # Crea il trainer
        trainer = CNNTrainer(cnn_model, learning_rate=1e-4, device=DEVICE)
        
        # Genera dati di training con FEM solver per diversi valori di diffusività
        fem_solver = FEMSolver(nvx=NVX, nvy=NVY, sigma_h=SIGMA_H, a=A, fr=FR, ft=FT, fd=FD)
        
        # Usa diversi valori di sigma per il dataset
        sigma_values = [9.5298e-5, 9.5298e-4, 9.5298e-3]  # 0.1x, 1x, 10x
        
        # Genera i dati di training
        print("Generazione dati di training dal FEM solver...")
        inputs, targets, sigmas = trainer.generate_training_data_from_fem(
            fem_solver, sigma_values, T=T, dt=DT, num_samples=300
        )
        
        # Converti in dataset PyTorch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Prepara i tensori per il dataloader
        input_tensor = torch.stack(inputs)
        sigma_tensor = torch.tensor(sigmas, dtype=torch.float32).view(-1, 1)
        
        # Creiamo una lista di liste di tensori target
        target_tensors = []
        for i in range(len(targets[0])):
            target_tensors.append(torch.stack([t[i] for t in targets]))
        
        # Dividi in training e validation
        n_samples = len(inputs)
        train_idx = int(0.8 * n_samples)
        
        train_dataset = (input_tensor[:train_idx], 
                         [t[:train_idx] for t in target_tensors], 
                         sigma_tensor[:train_idx])
        val_dataset = (input_tensor[train_idx:], 
                       [t[train_idx:] for t in target_tensors], 
                       sigma_tensor[train_idx:])
        
        # Crea i dataloader
        train_loader = DataLoader(
            dataset=list(zip(train_dataset[0], train_dataset[1], train_dataset[2])),
            batch_size=16,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset=list(zip(val_dataset[0], val_dataset[1], val_dataset[2])),
            batch_size=16
        )
        
        # Addestramento
        history = trainer.train(train_loader, val_loader, num_epochs=100)
        
        # Salva il modello addestrato
        torch.save(cnn_model.state_dict(), model_path)
        print(f"Modello CNN salvato in: {model_path}")
        
        # Salva il grafico delle loss
        trainer.plot_loss(history, model_dir)
        
    elif solver_type == 'cnn_predict':
        if model_name is None:
            print("Errore: È necessario specificare model_name per la predizione con CNN")
            return
        
        # Carica il modello
        model_dir = os.path.join("models", model_name)
        if not os.path.exists(model_dir):
            print(f"Errore: Il modello {model_name} non esiste.")
            return
        
        # Ricrea il modello CNN
        cnn_model = CNNSolver(device=DEVICE).to(DEVICE)
        cnn_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        cnn_model.eval()
        
        # Calcola la soluzione utilizzando il metodo compute_solution
        print(f"Calcolo della soluzione CNN per il caso: {case}")
        # Ottieni i parametri fisici dal caso
        sigma_h, _, _, _, _ = get_physics_params(case)
        solution_data = cnn_model.compute_solution(
            T=T,
            nvx=NVX,
            nvy=NVY,
            sigma_h=sigma_h,
            num_frames=num_frames
        )
        
        # Salva la soluzione come frame e video
        save_solution(solution_data, "cnn", case, model_name)
    
    else:
        print(f"Errore: Tipo di solver '{solver_type}' non supportato.")
        return

# --- Nuove funzioni helper per solve() ---

def get_physics_params(case):
    """
    Restituisce i parametri fisici in base al caso selezionato.
    """
    # Parametri base
    sigma_h = SIGMA_H  # Diffusività
    a = A              # Costante di diffusione
    fr = FR            # Forza di stimolo
    ft = FT            # Forza di stimolo temporale
    fd = FD            # Forza di stimolo spaziale
    
    # Modifica la diffusività in base al caso
    if case == "Low_Diffusivity_01x":
        sigma_h *= 0.1
    elif case == "High_Diffusivity_10x":
        sigma_h *= 10.0
    
    return sigma_h, a, fr, ft, fd

def create_model_dir(model_name):
    """
    Crea la directory per il modello.
    """
    models_base_dir = 'models'
    model_dir = os.path.join(models_base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def save_model(model, model_dir):
    """
    Salva i pesi del modello.
    """
    model_path = os.path.join(model_dir, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Modello salvato in: {model_path}")

def load_model(model, model_dir):
    """
    Carica i pesi del modello.
    """
    model_path = os.path.join(model_dir, 'model_weights.pth')
    model.load_state_dict(torch.load(model_path, map_location=model.device))
    model.eval()
    print(f"Modello caricato da: {model_path}")

def copy_script_to_model_dir(model_dir):
    """
    Copia il file pinn_solver.py nella directory del modello.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pinn_script_path = os.path.join(current_dir, 'pinn_solver.py')
    script_copy_path = os.path.join(model_dir, 'pinn_solver_script.py')
    
    import shutil
    shutil.copy2(pinn_script_path, script_copy_path)
    print(f"Script PINN copiato in: {script_copy_path}")

def save_solution(solution_data, solver_type, case, model_name):
    """
    Salva i risultati della soluzione come frame e video.
    """
    if solver_type == "fem":
        output_dir = os.path.join('assets', 'fem', case)
    elif solver_type == "cnn":
        model_name_safe = model_name if model_name else "default_model"
        output_dir = os.path.join('assets', 'cnn', model_name_safe, f"CNN_Prediction_{case}")
    else:  # pinn
        model_name_safe = model_name if model_name else "default_model"
        output_dir = os.path.join('assets', 'pinn', model_name_safe, f"PINN_Prediction_{case}")
    
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Salvataggio risultati in: {output_dir}")
    
    # Genera i frame dalle soluzioni calcolate
    print(f"Generazione dei frame...")
    times = solution_data['times']
    solutions = solution_data['solutions']
    x = solution_data['x']
    y = solution_data['y']
    nvx = len(x)
    nvy = len(y)
    
    for i, (t, solution) in enumerate(zip(times, solutions)):
        # Appiattire la soluzione per create_single_frame
        u_flat = solution.flatten(order='F')
        
        fig = create_single_frame(u_flat, nvx, nvy, t, case)
        plt.savefig(os.path.join(frames_dir, f'frame_{i:04d}.png'))
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{len(times)} generato.")
    
    print(f"  {len(times)} frame salvati in: {frames_dir}")
    
    # Crea il video dai frame
    create_video_from_frames(frames_dir, case)

def is_newer_model(model_name):
    """
    Controlla se il modello è stato creato con la versione più recente del codice.
    """
    # Semplice euristica: controllo del nome o della data di creazione
    try:
        # Se il modello è stato creato dopo l'aggiornamento del codice
        timestamp = int(model_name.split('_')[-1])
        return timestamp > 160000  # Solo un esempio di timestamp di riferimento
    except:
        return False  # In caso di errore, assume la versione precedente

# --- Interfaccia a Riga di Comando ---

def main():
    parser = argparse.ArgumentParser(description="SciML Monodomain Project")
    parser.add_argument(
        'action', 
        choices=['fem', 'pinn-train', 'pinn-predict', 'cnn-train', 'cnn-predict', 'deepritz-train', 'deepritz-predict'], 
        help="Azione da eseguire."
    )
    parser.add_argument(
        '--case', 
        choices=['high', 'normal', 'low', 'all'], 
        default='all',
        help="Caso di diffusività per la simulazione (FEM o PINN). Default 'all' esegue tutti i casi."
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help="Nome del modello da usare per il training o la predizione. Se non specificato, verrà usato un nome generato automaticamente per il training."
    )
    
    args = parser.parse_args()

    if args.action == 'fem':
        if args.case == 'all':
            print("Esecuzione di tutte le simulazioni FEM...")
            run_fem_simulation('High_Diffusivity_10x', 10.0)
            run_fem_simulation('Normal_Diffusivity_1x', 1.0)
            run_fem_simulation('Low_Diffusivity_01x', 0.1)
        elif args.case == 'high':
            run_fem_simulation('High_Diffusivity_10x', 10.0)
        elif args.case == 'normal':
            run_fem_simulation('Normal_Diffusivity_1x', 1.0)
        elif args.case == 'low':
            run_fem_simulation('Low_Diffusivity_01x', 0.1)
    
    elif args.action == 'pinn-train':
        trained_model = train_pinn_model(args.model_name)
        print(f"\nModello salvato come '{trained_model}'. Per generare predizioni, esegui:")
        print(f"python -m src.python.main pinn-predict --model-name {trained_model} --case [high|normal|low|all]")
        
    elif args.action == 'pinn-predict':
        if args.case == 'all':
            print("Generazione di tutte le predizioni PINN...")
            generate_pinn_frames('high', args.model_name)
            generate_pinn_frames('normal', args.model_name)
            generate_pinn_frames('low', args.model_name)
        else:
            generate_pinn_frames(args.case, args.model_name)
    
    elif args.action == 'cnn-train':
        trained_model = train_cnn_model(args.model_name)
        print(f"\nModello CNN salvato come '{trained_model}'.")
        
    elif args.action == 'cnn-predict':
        if args.case == 'all':
            print("Generazione di tutte le predizioni CNN...")
            generate_cnn_frames('high', args.model_name)
            generate_cnn_frames('normal', args.model_name)
            generate_cnn_frames('low', args.model_name)
        else:
            generate_cnn_frames(args.case, args.model_name)
    
    elif args.action == 'deepritz-train':
        trained_model = train_deepritz_model(args.model_name)
        print(f"\nModello DeepRitz salvato come '{trained_model}'.")
        
    elif args.action == 'deepritz-predict':
        if args.case == 'all':
            print("Generazione di tutte le predizioni DeepRitz...")
            generate_deepritz_frames('high', args.model_name)
            generate_deepritz_frames('normal', args.model_name)
            generate_deepritz_frames('low', args.model_name)
        else:
            generate_deepritz_frames(args.case, args.model_name)

if __name__ == "__main__":
    main()
