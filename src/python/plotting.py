# src/plotting.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import subprocess

def create_single_frame(u_data, nvx, nvy, time_val, case_name, vmin=0, vmax=1):
    """Crea un singolo frame per la visualizzazione."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    x = np.linspace(0, 1, nvx)
    y = np.linspace(0, 1, nvy)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # The solution u_data is a flat vector, needs to be reshaped
    u_grid = u_data.reshape(nvx, nvy, order='F')
    
    levels = np.linspace(vmin, vmax, 21)
    contour = ax.contourf(X, Y, u_grid, levels=levels, cmap='viridis', extend='both')
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Potenziale Transmembrana (u)')
    
    # Aggiungi regioni e stimolo
    circle1 = patches.Circle((0.3, 0.7), 0.1, lw=2, ec='red', fc='none', ls='--')
    circle2 = patches.Circle((0.7, 0.3), 0.15, lw=2, ec='red', fc='none', ls='--')
    circle3 = patches.Circle((0.5, 0.5), 0.1, lw=2, ec='red', fc='none', ls='--')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    rect = patches.Rectangle((0.9, 0.9), 0.1, 0.1, lw=2, ec='white', fc='none')
    ax.add_patch(rect)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{case_name}\nTime = {time_val:.2f} ms', fontsize=16)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def create_video_from_frames(frame_dir, case_name, fps=10):
    """Crea un video MP4 dai frame usando ffmpeg."""
    video_path = os.path.join(frame_dir, f"{case_name}.mp4")
    frame_pattern = os.path.join(frame_dir, "frame_%04d.png")
    
    print(f"Creazione video: {video_path}")
    
    command = [
        'ffmpeg',
        '-y',  # Sovrascrivi il file di output se esiste
        '-r', str(fps),  # Frame rate
        '-i', frame_pattern,
        '-c:v', 'libx264',  # Codec video
        '-pix_fmt', 'yuv420p',  # Formato pixel per compatibilità
        '-crf', '18',  # Qualità (valori più bassi sono migliori)
        video_path
    ]
    
    try:
        # Using capture_output=True to hide ffmpeg's verbose output from the console
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Video creato con successo.")
    except FileNotFoundError:
        print("Errore: ffmpeg non trovato. Assicurati che sia installato e nel PATH di sistema.")
    except subprocess.CalledProcessError as e:
        print("Errore durante la creazione del video con ffmpeg.")
        print(f"Comando: {' '.join(e.cmd)}")
        print(f"Errore: {e.stderr}")
