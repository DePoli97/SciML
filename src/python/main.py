# src/main.py
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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

def train_pinn_model():
    """Addestra il modello PINN e salva sia il modello che il grafico delle loss."""
    models_dir = 'models'
    model_path = os.path.join(models_dir, 'best_pinn_model.pth')
    loss_plot_path = os.path.join(models_dir, 'training_loss.png')
    os.makedirs(models_dir, exist_ok=True)
    
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

def generate_pinn_frames(case='normal'):
    """Carica un modello PINN addestrato e genera i frame/video."""
    model_path = os.path.join('models', 'best_pinn_model.pth')
    if not os.path.exists(model_path):
        print(f"Errore: Modello non trovato in {model_path}. Esegui prima il training.")
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

    # Configura le directory di output
    case_name = f"PINN_Prediction_{case_display}"
    output_dir = os.path.join('assets', 'pinn')
    frames_dir = os.path.join(output_dir, case_name, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

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
        train_pinn_model()
        
    elif args.action == 'pinn-predict':
        if args.case == 'all':
            print("Generazione di tutte le predizioni PINN...")
            generate_pinn_frames('high')
            generate_pinn_frames('normal')
            generate_pinn_frames('low')
        else:
            generate_pinn_frames(args.case)

if __name__ == "__main__":
    main()
