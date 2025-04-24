import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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



def jacobi_iteration(Ux, omega=1.0):
    """
    Realiza una iteración del método de Jacobi con posible relajación.
    
    Parámetros:
    Ux -- Matriz actual de velocidades
    omega -- Factor de relajación (1: Jacobi estándar)
    
    Retorna:
    Ux_new -- Nueva matriz actualizada
    max_error -- Error máximo absoluto en esta iteración
    """
    ny, nx = Ux.shape
    Ux_new = Ux.copy()
    max_error = 0.0
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            vecino_der = Ux[j, i+1]
            vecino_izq = Ux[j, i-1]
            vecino_sup = Ux[j+1, i]
            vecino_inf = Ux[j-1, i]

            termino_vorticida = 0.05 * (vecino_sup - vecino_inf)
            termino_convectivo = 0.125 * Ux[j,i] * (vecino_der - vecino_izq) - termino_vorticida
            
            # Cálculo del nuevo valor
            nuevo_valor = 0.25*(vecino_der + vecino_izq + vecino_sup + vecino_inf) - termino_convectivo
            
            # Actualización con relajación
            Ux_new[j,i] = omega * nuevo_valor + (1 - omega) * Ux[j,i]
            
            # Calcular error
            current_error = abs(Ux_new[j,i] - Ux[j,i])
            if current_error > max_error:
                max_error = current_error
                
    return Ux_new, max_error



def solve_jacobi(nx=5, ny=5, tol=1e-6, max_iter=1000, omega=1.0):
    """
    Implementación del método de Jacobi con seguimiento de convergencia
    
    Retorna:
    Ux -- Matriz solución
    errors -- Lista de errores por iteración
    """
    Ux = setup_grid(nx, ny, 1)
    errors = []
    residuals  = []
    start_time = time.time()
    
    for iter in range(max_iter):
        Ux_new, error = jacobi_iteration(Ux, omega)
        errors.append(error)
        
        # Actualizar solución manteniendo bordes fijos
        Ux[1:-1, 1:-1] = Ux_new[1:-1, 1:-1]

        if error < tol:
            print(f"Convergencia en {iter+1} iteraciones")
            break
            
    
    elapsed = time.time() - start_time
    return Ux,  errors, iter+1, elapsed



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





# Ejecución modificada
if __name__ == "__main__":
    tamanno_n, tamanno_m = 100, 10
    
    # Resolver con Jacobi (omega=1 para Jacobi estándar)
    solution_j, errores_j, iters_j, time_j = solve_jacobi(nx=tamanno_n, ny=tamanno_m, omega=1.0)
    
    graficar_conjuntamente(
        metodo="Jacobi",
        errors=errores_j,
        matriz=np.round(solution_j, 7),
        title=f"Distribución de Velocidad en el Fluido (Jacobi) - {tamanno_n}x{tamanno_m}",
        iteraciones=iters_j,
        tiempo=time_j,
        error_final=errores_j[-1],
        # residuo_final=gs_residuals[-1]
    )
    
    

