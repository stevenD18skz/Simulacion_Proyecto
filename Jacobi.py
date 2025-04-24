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
    
    for iter in range(max_iter):
        Ux_new, error = jacobi_iteration(Ux, omega)
        errors.append(error)
        
        # Actualizar solución manteniendo bordes fijos
        Ux[1:-1, 1:-1] = Ux_new[1:-1, 1:-1]
        
        if error < tol:
            print(f"Convergencia en {iter+1} iteraciones")
            break
            
    return Ux, errors

def plot_convergence(errors):
    """Grafica la evolución del error por iteración"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(errors, 'b-o', linewidth=2, markersize=4)
    plt.title("Convergencia del método de Jacobi", fontsize=14)
    plt.xlabel("Iteración", fontsize=12)
    plt.ylabel("Error máximo (escala log)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


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



# Ejecución modificada
if __name__ == "__main__":
    tamanno_n, tamanno_m = 30, 15
    
    # Resolver con Jacobi (omega=1 para Jacobi estándar)
    solucion, errores = solve_jacobi(nx=tamanno_n, ny=tamanno_m, omega=1.0)
    
    # Visualizar resultados
    print("\nEstadísticas de convergencia:")
    print(f"Iteraciones realizadas: {len(errores)}")
    print(f"Error final: {errores[-1]:.2e}")
    
    visualizar_matriz(np.round(solucion, 7), 
                    f"Distribución de Velocidades (Jacobi)\nRejilla {tamanno_n}x{tamanno_m}")
    
    plot_convergence(errores)