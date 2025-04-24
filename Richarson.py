import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# --- Reutilizamos setup_grid y build_system con vorticidad ---
def setup_grid(nx, ny, initial_velocity=1.0):
    Ux = np.zeros((ny, nx))
    Ux[:, 0] = initial_velocity
    Ux[:, -1] = 0.0
    Ux[0, :] = 0.0
    Ux[-1, :] = 0.0
    return Ux

def build_system(Ux, omega=0.1):
    ny, nx = Ux.shape
    num = (nx-2)*(ny-2)
    F = np.zeros(num)
    idx_map = np.zeros(Ux.shape, int)
    idx_map[1:-1,1:-1] = np.arange(num).reshape(ny-2, nx-2)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            idx = idx_map[j,i]
            ue, uw = Ux[j,i+1], Ux[j,i-1]
            un, us = Ux[j+1,i], Ux[j-1,i]
            conv = 0.125 * Ux[j,i] * (ue - uw)
            vort = 0.5 * omega * (un - us)
            F[idx] = Ux[j,i] - 0.25*(ue+uw+un+us) + conv - vort
    return None, F  # Solo necesitamos F

# --- Método de Richardson ---
def solve_richardson(nx, ny, omega=0.1, alpha=0.5, tol=1e-6, max_iter=1000):
    U = setup_grid(nx, ny, 1.0)
    errors = []
    residuals = []
    start_time = time.time()
    
    for k in range(max_iter):
        _, F = build_system(U, omega)
        delta = -alpha * F
        max_error = np.max(np.abs(delta))
        
        # Actualizar interior
        U_interior = U[1:-1,1:-1].flatten() + delta
        U[1:-1,1:-1] = U_interior.reshape(ny-2, nx-2)
        
        errors.append(max_error)
        residuals.append(np.linalg.norm(F, 2))
        
        if max_error < tol:
            break
    
    elapsed = time.time() - start_time
    return U, errors, residuals, k+1, elapsed



def graficar_conjuntamente(metodo, errors, matriz, title, 
                           iteraciones=None, tiempo=None, 
                           error_final=None, residuo_final=None):
    """
    Grafica la evolución del error y un heatmap de la matriz en la misma ventana.
    También puede mostrar métricas si se proporcionan.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 fila, 2 columnas

    # Subplot 1: Convergencia
    axes[0].semilogy(errors, 'b-o', linewidth=2, markersize=4)
    axes[0].set_title(f"Convergencia del método de {metodo}", fontsize=14)
    axes[0].set_xlabel("Iteración", fontsize=12)
    axes[0].set_ylabel("Error máximo (escala log)", fontsize=12)
    axes[0].grid(True, which='both', linestyle='--', alpha=0.7)

    # Mostrar métricas si se proporcionan
    if iteraciones is not None or tiempo is not None or error_final is not None or residuo_final is not None:
        texto = ""
        if iteraciones is not None:
            texto += f"Iteraciones: {iteraciones}\n"
        if tiempo is not None:
            texto += f"Tiempo total: {tiempo:.4f} s\n"
        if error_final is not None:
            texto += f"Error final: {error_final:.2e}\n"
        if residuo_final is not None:
            texto += f"Residuo final: {residuo_final:.2e}"
        
        axes[0].text(0.95, 0.05, texto, transform=axes[0].transAxes,
                     fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # Subplot 2: Heatmap de la matriz
    sns.heatmap(
        matriz,
        ax=axes[1],
        annot=False,
        fmt=".1f",
        cmap="viridis",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={'label': 'Velocidad (Ux)'}
    )
    axes[1].set_title(title, fontsize=14, pad=20)
    axes[1].set_xlabel('Dirección X (izquierda -> derecha)', fontsize=12)
    axes[1].set_ylabel('Dirección Y (abajo -> arriba)', fontsize=12)

    # Ajustar ticks para heatmap
    axes[1].set_xticks(np.arange(matriz.shape[1]) + 0.5)
    axes[1].set_xticklabels(np.arange(matriz.shape[1]))
    axes[1].set_yticks(np.arange(matriz.shape[0]) + 0.5)
    axes[1].set_yticklabels(np.arange(matriz.shape[0]-1, -1, -1))
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()


# --- Ejecución y visualización ---
if __name__ == "__main__":
    nx, ny, omega = 100, 10, 0.1
    alpha = 0.5
    tol = 1e-6
    max_iter = 1000
    
    sol_rich, rich_errors, rich_res, rich_iters, rich_time = solve_richardson(
        nx, ny, omega, alpha, tol, max_iter
    )
    
    print(f"Richardson convergió en {rich_iters} iteraciones")
    print(f"Tiempo total: {rich_time:.4f} s")
    print(f"Error final: {rich_errors[-1]:.2e}")
    print(f"Norma final del residuo: {rich_res[-1]:.2e}")
    
    
    graficar_conjuntamente(
        metodo="Richardson",
        errors=rich_errors,
        matriz=np.round(sol_rich, 7),
        title=f"Distribución de Velocidad en el Fluido (Richardson) - {nx}x{ny}",
        iteraciones=rich_iters,
        tiempo=rich_time,
        error_final=rich_errors[-1],
        residuo_final=rich_res[-1]

    )