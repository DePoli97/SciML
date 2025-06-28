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
    output_dir = os.path.join('assets', 'frames', 'fem')
    
    solver = FEMSolver(nvx=NVX, nvy=NVY, sigma_h=SIGMA_H, a=A, fr=FR, ft=FT, fd=FD)
    
    frame_dir = solver.solve(
        T=T, 
        dt=DT, 
        sigma_d_factor=sigma_d_factor, 
        case_name=case_name, 
        output_dir=output_dir
    )
    
    create_video_from_frames(frame_dir, case_name)

def train_pinn_model():
    """Addestra il modello PINN e lo salva."""
    model_path = os.path.join('models', 'best_pinn_model.pth')
    os.makedirs('models', exist_ok=True)
    
    pinn = PINNSolver(
        device=DEVICE,
        sigma_h=SIGMA_H,
        a=A,
        fr=FR,
        ft=FT,
        fd=FD
    ).to(DEVICE)
    
    trainer = PINNTrainer(pinn, learning_rate=1e-3, device=DEVICE, T=T)
    
    trainer.train(n_epochs=10000, n_points_pde=4096, n_points_ic=1024)
    
    torch.save(pinn.state_dict(), model_path)
    print(f"Modello PINN salvato in: {model_path}")

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
    
    model = PINNSolver(
        device=DEVICE,
        sigma_h=sigma_h_value,
        a=A, 
        fr=FR, 
        ft=FT, 
        fd=FD
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    case_name = f"PINN_Prediction_{case_display}"
    frame_dir = os.path.join('assets', 'frames', 'pinn', case_name)
    os.makedirs(frame_dir, exist_ok=True)

    print(f"Generazione frame dalla PINN in: {frame_dir}")

    x = np.linspace(0, 1, NVX)
    y = np.linspace(0, 1, NVY)
    X, Y = np.meshgrid(x, y, indexing='ij')
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(DEVICE)
    y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).view(-1, 1).to(DEVICE)

    num_frames = 100
    times = np.linspace(0, T, num_frames)

    for i, t_val in enumerate(times):
        t_tensor = torch.full_like(x_flat, t_val)
        with torch.no_grad():
            u_pred = model(x_flat, y_flat, t_tensor).cpu().numpy()
        
        fig = create_single_frame(u_pred, NVX, NVY, t_val, case_name)
        plt.savefig(os.path.join(frame_dir, f'frame_{i:04d}.png'))
        plt.close(fig)
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{num_frames} generato.")

    create_video_from_frames(frame_dir, case_name)

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
