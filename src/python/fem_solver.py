# src/fem_solver.py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import os
import time
import matplotlib.pyplot as plt
from .plotting import create_single_frame

class FEMSolver:
    """Risolve l'equazione del monodominio usando il Metodo degli Elementi Finiti."""
    def __init__(self, nvx, nvy, sigma_h, a, fr, ft, fd):
        self.nvx, self.nvy = nvx, nvy
        self.hx = 1.0 / (nvx - 1)
        self.hy = 1.0 / (nvy - 1)
        self.sigma_h = sigma_h
        self.a, self.fr, self.ft, self.fd = a, fr, ft, fd

    def _assemble_mass(self):
        Mref = np.array([[1/3, 1/6], [1/6, 1/3]])
        Mx, My = self.hx * Mref, self.hy * Mref
        Aloc = np.kron(My, Mx)
        nv, ne = self.nvx * self.nvy, (self.nvx - 1) * (self.nvy - 1)
        id_matrix = np.arange(nv).reshape(self.nvx, self.nvy, order='F')
        a, b, c, d = (id_matrix[:-1, :-1].flatten(order='F'),
                      id_matrix[1:, :-1].flatten(order='F'),
                      id_matrix[:-1, 1:].flatten(order='F'),
                      id_matrix[1:, 1:].flatten(order='F'))
        conn = np.array([a, b, c, d])
        I, J, V = [], [], []
        for e in range(ne):
            for i in range(4):
                for j in range(4):
                    I.append(conn[i, e]); J.append(conn[j, e]); V.append(Aloc[i, j])
        return sp.csr_matrix((V, (I, J)), shape=(nv, nv))

    def _assemble_diffusion(self, sigma_elements):
        Aref, Mref = np.array([[1, -1], [-1, 1]]), np.array([[1/3, 1/6], [1/6, 1/3]])
        Ax, Ay = (1/self.hx) * Aref, (1/self.hy) * Aref
        Mx, My = self.hx * Mref, self.hy * Mref
        nv, ne = self.nvx * self.nvy, (self.nvx - 1) * (self.nvy - 1)
        id_matrix = np.arange(nv).reshape(self.nvx, self.nvy, order='F')
        a, b, c, d = (id_matrix[:-1, :-1].flatten(order='F'),
                      id_matrix[1:, :-1].flatten(order='F'),
                      id_matrix[:-1, 1:].flatten(order='F'),
                      id_matrix[1:, 1:].flatten(order='F'))
        conn = np.array([a, b, c, d])
        I, J, V = [], [], []
        for e in range(ne):
            Aloc = sigma_elements[e] * (np.kron(My, Ax) + np.kron(Ay, Mx))
            for i in range(4):
                for j in range(4):
                    I.append(conn[i, e]); J.append(conn[j, e]); V.append(Aloc[i, j])
        return sp.csr_matrix((V, (I, J)), shape=(nv, nv))

    def _setup_diffusivity(self, sigma_d_factor):
        ne = (self.nvx - 1) * (self.nvy - 1)
        sigma_elements = np.full(ne, self.sigma_h)
        x_centers = np.linspace(0, 1, self.nvx-1) + 0.5*self.hx
        y_centers = np.linspace(0, 1, self.nvy-1) + 0.5*self.hy
        X_centers, Y_centers = np.meshgrid(x_centers, y_centers, indexing='ij')
        
        for i in range(self.nvx-1):
            for j in range(self.nvy-1):
                e = i * (self.nvy-1) + j
                x_c, y_c = Y_centers[i, j], X_centers[i, j]
                d1 = (x_c - 0.5)**2 + (y_c - 0.5)**2 < 0.1**2
                d2 = (x_c - 0.7)**2 + (y_c - 0.3)**2 < 0.15**2
                d3 = (x_c - 0.3)**2 + (y_c - 0.7)**2 < 0.1**2
                
                if d1 or d2 or d3:
                    sigma_elements[e] = sigma_d_factor * self.sigma_h
        return sigma_elements

    def _setup_initial_condition(self):
        x = np.linspace(0, 1, self.nvx)
        y = np.linspace(0, 1, self.nvy)
        X, Y = np.meshgrid(x, y, indexing='ij')
        u0 = np.zeros((self.nvx, self.nvy))
        u0[(X >= 0.9) & (Y >= 0.9)] = 1.0
        return u0.flatten(order='F')

    def solve(self, T, dt, sigma_d_factor, case_name, output_dir):
        print(f"Avvio solver FEM per il caso: {case_name}")
        start_time = time.time()
        
        nt = int(T / dt)
        sigma_elements = self._setup_diffusivity(sigma_d_factor)
        
        print("  Assemblaggio matrici...")
        M = self._assemble_mass()
        K = self._assemble_diffusion(sigma_elements)
        A = M + dt * K
        
        u = self._setup_initial_condition()
        
        frame_dir = os.path.join(output_dir, case_name)
        os.makedirs(frame_dir, exist_ok=True)
        
        print("  Avvio ciclo di integrazione temporale...")
        save_every = max(1, int(nt / 100)) # Save ~100 frames
        frame_count = 0
        
        # Salva frame iniziale
        fig = create_single_frame(u, self.nvx, self.nvy, 0.0, case_name)
        plt.savefig(os.path.join(frame_dir, f'frame_{frame_count:04d}.png'))
        plt.close(fig)
        frame_count += 1
        
        for n in range(nt):
            t = (n + 1) * dt
            f_reaction = self.a * (u - self.fr) * (u - self.ft) * (u - self.fd)
            rhs = M.dot(u) - dt * M.dot(f_reaction)
            u_new = spsolve(A, rhs)
            u = u_new
            
            if n % save_every == 0 or n == nt - 1:
                fig = create_single_frame(u, self.nvx, self.nvy, t, case_name)
                plt.savefig(os.path.join(frame_dir, f'frame_{frame_count:04d}.png'))
                plt.close(fig)
                frame_count += 1

            if (n + 1) % (nt // 10) == 0:
                print(f"    Progresso: {100 * (n+1)/nt:.0f}% (t={t:.1f}s)")

        end_time = time.time()
        print(f"  Simulazione completata in {end_time - start_time:.2f} secondi.")
        print(f"  {frame_count} frame salvati in: {frame_dir}")
        return frame_dir
