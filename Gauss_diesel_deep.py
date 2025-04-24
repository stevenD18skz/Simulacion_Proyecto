import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def setup_grid(nx, ny, initial_velocity):
    """
    Configura la malla computacional con condiciones de contorno
    
    Parámetros:
    nx -- Número de nodos en dirección x (horizontal)
    ny -- Número de nodos en dirección y (vertical)
    initial_velocity -- Velocidad inicial en el borde izquierdo
    
    Retorna:
    Matriz 2D inicializada con condiciones de contorno:
    - Borde izquierdo (i=0): Ux = initial_velocity
    - Bordes derecho, superior e inferior: Ux = 0
    """
    # Crear malla de ceros (ny filas × nx columnas)
    Ux = np.zeros((ny, nx))
    
    # Asignar condiciones de contorno
    Ux[:, 0] = initial_velocity # Borde izquierdo (toda la primera columna)
    Ux[:, -1] = 0.0   # Borde derecho (toda la última columna)
    Ux[0, :] = 0.0    # Borde inferior (primera fila)
    Ux[-1, :] = 0.0   # Borde superior (última fila)
    
    return Ux


# =============================================================================
# Implementación del Método de Gauss-Seidel
# =============================================================================
def gauss_seidel_iteration(Ux, omega=1.0):
    """
    Realiza una iteración de Gauss-Seidel con relajación.
    
    Parámetros:
    Ux -- Matriz de velocidades (se actualiza in-place)
    omega -- Factor de relajación (1: Gauss-Seidel estándar)
    
    Retorna:
    max_error -- Error máximo absoluto en esta iteración
    """
    ny, nx = Ux.shape
    max_error = 0.0
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            old_val = Ux[j,i]
            
            vecino_der = Ux[j, i+1]
            vecino_izq = Ux[j, i-1]
            vecino_sup = Ux[j+1, i]
            vecino_inf = Ux[j-1, i]

            termino_vorticida = 0.05 * (vecino_sup - vecino_inf)
            termino_convectivo = 0.125 * Ux[j,i] * (vecino_der - vecino_izq) - termino_vorticida
            
            # Cálculo del nuevo valor usando valores ya actualizados
            nuevo_valor = 0.25*(vecino_der + vecino_izq + vecino_sup + vecino_inf) - termino_convectivo
            
            # Actualización in-place con relajación
            Ux[j,i] = omega * nuevo_valor + (1 - omega) * old_val
            
            # Calcular error
            current_error = abs(Ux[j,i] - old_val)
            if current_error > max_error:
                max_error = current_error
                
    return max_error

def solve_gauss_seidel(nx=5, ny=5, tol=1e-6, max_iter=1000, omega=1.0):
    """
    Implementación completa de Gauss-Seidel con métricas extendidas
    
    Retorna:
    Ux -- Matriz solución
    stats -- Diccionario con métricas de convergencia
    """
    Ux = setup_grid(nx, ny, 1)
    errors = []
    residuals = []
    start_time = time.time()
    
    for iter in range(max_iter):
        error = gauss_seidel_iteration(Ux, omega)
        errors.append(error)
        
        # Calcular residuo cada 10 iteraciones para eficiencia
        if iter % 10 == 0:
            residual = np.linalg.norm(calculate_residual(Ux), 2)
            residuals.append(residual)
        
        if error < tol:
            break
            
    exec_time = time.time() - start_time
    
    return Ux, {
        'errors': errors,
        'residuals': residuals,
        'time': exec_time,
        'iterations': iter+1,
        'final_error': error,
        'final_residual': residuals[-1] if residuals else np.nan
    }

def calculate_residual(Ux):
    """Calcula el residuo L2 para todos los nodos interiores"""
    ny, nx = Ux.shape
    residuo = np.zeros_like(Ux)
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            vecino_der = Ux[j, i+1]
            vecino_izq = Ux[j, i-1]
            vecino_sup = Ux[j+1, i]
            vecino_inf = Ux[j-1, i]

            termino_vorticida = 0.05 * (vecino_sup - vecino_inf)
            termino_convectivo = 0.125 * Ux[j,i] * (vecino_der - vecino_izq) - termino_vorticida
            
            residuo[j,i] = Ux[j,i] - (0.25*(vecino_der + vecino_izq + vecino_sup + vecino_inf) - termino_convectivo)
    
    return residuo[1:-1, 1:-1].flatten()


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


# =============================================================================
# Visualización mejorada con métricas comparativas
# =============================================================================
def enhanced_convergence_plot(stats, method_name):
    """Grafica múltiples métricas de convergencia"""
    plt.figure(figsize=(12, 6))
    
    # Error máximo
    plt.subplot(1, 2, 1)
    plt.semilogy(stats['errors'], 'b-o', markersize=4, label='Error máximo')
    plt.title(f"Convergencia de {method_name}\nIteraciones: {stats['iterations']}\nTiempo: {stats['time']:.2f}s")
    plt.xlabel('Iteración')
    plt.ylabel('Error (log)')
    plt.grid(True)
    
    # Residuales L2
    plt.subplot(1, 2, 2)
    plt.semilogy(np.linspace(0, stats['iterations'], len(stats['residuals'])), 
                stats['residuals'], 'r--d', markersize=4, label='Residuo L2')
    plt.title('Evolución del Residuo L2')
    plt.xlabel('Iteración')
    plt.ylabel('Residuo (log)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Ejecución y análisis comparativo
# =============================================================================
if __name__ == "__main__":
    tamanno_n, tamanno_m = 30, 15
    
    # Resolver con Gauss-Seidel
    solucion_gs, stats_gs = solve_gauss_seidel(nx=tamanno_n, ny=tamanno_m, omega=1.0)
    
    # Visualización completa
    print("\nMétricas clave de Gauss-Seidel:")
    print(f"- Iteraciones totales: {stats_gs['iterations']}")
    print(f"- Tiempo ejecución: {stats_gs['time']:.4f} segundos")
    print(f"- Error final: {stats_gs['final_error']:.2e}")
    print(f"- Residuo L2 final: {stats_gs['final_residual']:.2e}")
    
    visualizar_matriz(np.round(solucion_gs, 7), 
                    f"Distribución de Velocidades (Gauss-Seidel)\nRejilla {tamanno_n}x{tamanno_m}")
    
    enhanced_convergence_plot(stats_gs, "Gauss-Seidel")
