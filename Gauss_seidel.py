import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import lil_matrix, csr_matrix


import seaborn as sns

# Reutilizamos setup_grid y build_system del entorno
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
    return csr_matrix((num,num)), F  # Only residual necessary

def gauss_seidel_solver(nx, ny, omega, tol=1e-6, max_iter=500):
    U = setup_grid(nx, ny, 1.0)
    errors = []
    residuals = []
    start_time = time.time()
    
    for k in range(max_iter):
        max_diff = 0.0
        # Update in-place using latest values
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                ue, uw = U[j, i+1], U[j, i-1]
                un, us = U[j+1, i], U[j-1, i]
                conv = 0.125 * U[j,i] * (ue - uw)
                vort = 0.5 * omega * (un - us)
                new_val = 0.25*(ue + uw + un + us) - conv + vort
                diff = abs(new_val - U[j,i])
                if diff > max_diff:
                    max_diff = diff
                U[j,i] = new_val
        errors.append(max_diff)
        
        # Compute residual norm
        _, F = build_system(U, omega)
        res_norm = np.linalg.norm(F, ord=2)
        residuals.append(res_norm)
        
        if max_diff < tol:
            break
    
    elapsed = time.time() - start_time
    return U, errors, residuals, k+1, elapsed


def visualizar_matriz(matriz, title):
    """
    Visualiza una matriz 2D como heatmap con orientación física correcta.
    Ejes:
    - X: Horizontal (izquierda -> derecha)
    - Y: Vertical (abajo -> arriba)
    """
    plt.figure(figsize=(10, 6))
    
    # Crear heatmap y ajustar ejes
    ax = sns.heatmap(
        matriz,
        annot=False,
        fmt=".1f",
        cmap="viridis",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={'label': 'Velocidad (Ux)'}
    )
    
    # Configurar ejes para coincidir con orientación física
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Dirección X (izquierda -> derecha)', fontsize=12)
    ax.set_ylabel('Dirección Y (abajo -> arriba)', fontsize=12)
    
    # Ajustar ticks para mostrar índices físicos
    ax.set_xticks(np.arange(matriz.shape[1]) + 0.5)
    ax.set_xticklabels(np.arange(matriz.shape[1]))
    ax.set_yticks(np.arange(matriz.shape[0]) + 0.5)
    ax.set_yticklabels(np.arange(matriz.shape[0]-1, -1, -1))  # Invertir orden Y
    
    plt.gca().invert_yaxis()  
    plt.tight_layout()
    plt.show()




# Parámetros
nx, ny, omega = 30, 15, 0.1

# Ejecutar Gauss-Seidel
solution_gs, gs_errors, gs_residuals, gs_iters, gs_time = gauss_seidel_solver(nx, ny, omega)

# Mostrar métricas
print(f"Gauss-Seidel convergió en {gs_iters} iteraciones") 
print(f"Tiempo total: {gs_time:.4f} segundos")
print(f"Error final (max diff): {gs_errors[-1]:.2e}")
print(f"Norma final del residuo: {gs_residuals[-1]:.2e}")

    
visualizar_matriz(np.round(solution_gs, 7), 
                f"Distribución de Velocidades (Jacobi)\nRejilla ")


# Gráficas
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.semilogy(gs_errors, marker='o')
plt.xlabel('Iteración')
plt.ylabel('Error máximo')
plt.title('Convergencia de GS (error)')

plt.subplot(1,2,2)
plt.semilogy(gs_residuals, marker='o')
plt.xlabel('Iteración')
plt.ylabel('||F||₂')
plt.title('Residual norm')

plt.tight_layout()
plt.show()
