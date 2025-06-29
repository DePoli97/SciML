# src/main.py
import argparse
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import inspect
import os.path

from .fem_solver import FEMSolver
from .pinn_solver import PINNSolver, PINNTrainer
from .plotting import create_single_frame, create_video_from_frames

# --- Parametri Globali ---
# Parametri fisici
SIGMA_H = 9.5298e-4
A = 18.515
FR = 0.0
FT = 0.2383
FD = 1.0

# Parametri di simulazione
T = 35.0
DT = 0.1
NVX = NVY = 101



# --- Configurazione dispositivo ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Usando dispositivo: {DEVICE}")

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
    create_video_from_frames(frame_dir, case_name)

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
    create_video_from_frames(frames_dir, case_name)
            
    return frames_dir

# --- Interfaccia a Riga di Comando ---

def main():
    parser = argparse.ArgumentParser(description="SciML Monodomain Project")
    parser.add_argument(
        'action', 
        choices=['fem', 'pinn-train', 'pinn-predict'], 
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

if __name__ == "__main__":
    main()
