import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
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

def build_system(Ux):
    """
    Construye el sistema no lineal F y la matriz Jacobiana J
    
    Parámetros:
    Ux -- Matriz 2D con los valores actuales de velocidad
    
    Retorna:
    J -- Matriz Jacobiana en formato disperso CSR
    F -- Vector de residuos
    """
    ny, nx = Ux.shape
    num_nodos = (nx-2)*(ny-2)  # Nodos interiores totales
    F = np.zeros(num_nodos) 
    J = lil_matrix((num_nodos, num_nodos)) # Matriz dispersa (LIL) para eficiencia en construcción
    
    # Mapa de índices 2D (i,j) a 1D para nodos interiores
    index_map = np.zeros(Ux.shape, dtype=int)
    index_map[1:-1, 1:-1] = np.arange(num_nodos).reshape(ny-2, nx-2)
    
    # Llenado del sistema ecuación por ecuación
    for j in range(1, ny-1):    # Recorrer filas (dirección y)
        for i in range(1, nx-1):  # Recorrer columnas (dirección x)
            idx = index_map[j, i]  # Índice 1D actual
            
            # -----------------------------------------------------------------
            # Ecuación discretizada: F(Ux) = 0
            # -----------------------------------------------------------------
            vecino_der = Ux[j, i+1]
            vecino_izq = Ux[j, i-1]
            vecino_sup = Ux[j+1, i]
            vecino_inf = Ux[j-1, i]

            termino_vorticida = 1/2 * 0.1 * (vecino_sup - vecino_inf)  
            
            termino_convectivo = 0.125 * Ux[j,i] * (vecino_der - vecino_izq) - termino_vorticida
            
            F[idx] = Ux[j,i] - 0.25*(vecino_der + vecino_izq + vecino_sup + vecino_inf) + termino_convectivo
            
            # -----------------------------------------------------------------
            # Construcción del Jacobiano: J[i,j] = ∂F[i]/∂Ux[j]
            # -----------------------------------------------------------------
            # Derivada respecto al nodo actual
            J[idx, idx] = 1 - 0.125*(vecino_der - vecino_izq)
            
            # Derivadas respecto a vecinos en x (i±1)
            if i+1 < nx-1:  # Vecino derecho existe (no es borde)
                J[idx, index_map[j, i+1]] = -0.25 + 0.125*Ux[j,i]
                
            if i-1 > 0:     # Vecino izquierdo existe
                J[idx, index_map[j, i-1]] = -0.25 - 0.125*Ux[j,i]
            
            # Derivadas respecto a vecinos en y (j±1) - solo términos lineales
            if j+1 < ny-1:  # Vecino superior existe
                J[idx, index_map[j+1, i]] = -0.25 + 0.0125
                
            if j-1 > 0:     # Vecino inferior existe
                J[idx, index_map[j-1, i]] = -0.25 - 0.0125

    return csr_matrix(J), F  # Convertir a formato CSR para operaciones eficientes

def newton_raphson(nx=5, ny=5, tol=1e-6, max_iter=50):
    """
    Implementación principal del método de Newton-Raphson
    
    Parámetros:
    nx -- Número de nodos en dirección x
    ny -- Número de nodos en dirección y
    tol -- Tolerancia para criterio de convergencia
    max_iter -- Número máximo de iteraciones permitidas
    initial_velocity -- Velocidad inicial en el borde izquierdo
    
    Retorna:
    Matriz 2D con la solución convergida
    """
    # 1. Configuración inicial de la malla
    Ux = setup_grid(nx, ny, 1)
    
    # 2. Bucle principal de Newton-Raphson
    for iteracion in range(max_iter):
        # Construir sistema lineal J·ΔUx = -F
        J, F = build_system(Ux)
        
        # Resolver sistema lineal usando método directo para matrices dispersas
        delta = spsolve(J, -F)
        
        # Actualizar solución: Ux_new = Ux_old + ΔUx
        # Solo actualizar nodos interiores (excluyendo bordes)
        Ux_interior = Ux[1:-1, 1:-1].flatten()
        Ux_interior += delta
        Ux[1:-1, 1:-1] = Ux_interior.reshape(ny-2, nx-2)
        
        # Calcular error máximo para criterio de convergencia
        max_error = np.max(np.abs(delta))
        print(f"Iter {iteracion+1}: Error = {max_error:.3e}")
        
        # Verificar convergencia
        if max_error < tol:
            print(f"Convergencia alcanzada en {iteracion+1} iteraciones")
            break
            
    return Ux

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
# Ejecución y visualización
# =============================================================================
if __name__ == "__main__":
    tamanno_n, tamanno_m = 30, 15 
    
    solucion = newton_raphson(nx=tamanno_n, ny=tamanno_m  )

    print("\nSolución final (orientación física):")
    print("Columnas = dirección x (izq -> der)")
    print("Filas = dirección y (inf -> sup)\n")
    print(solucion)  # Transponer para visualización correcta
    
    # Visualización mejorada
    visualizar_matriz(
        matriz=np.round(solucion, 7),
        title=f"Distribución de Velocidades - Rejilla {tamanno_n}x{tamanno_m}\n(Velocidad Inicial: Ux={1})"
    )
