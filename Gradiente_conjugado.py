import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def setup_grid(nx, ny, initial_velocity):
    """
    Configura la malla computacional con condiciones de contorno
    
    Parámetros:
    nx -- Número de nodos en dirección x
    ny -- Número de nodos en dirección y
    initial_velocity -- Velocidad inicial en el borde izquierdo
    
    Retorna:
    Ux -- Matriz 2D para velocidad
    """
    Ux = np.zeros((ny, nx))
    Ux[:, 0] = initial_velocity
    Ux[:, -1] = 0.0
    Ux[0, :] = 0.0
    Ux[-1, :] = 0.0
    return Ux

def build_system(Ux, nu=0.1):
    """
    Construye el sistema no lineal F y la matriz Jacobiana J, incluyendo vorticidad
    
    Parámetros:
    Ux -- Matriz 2D con los valores actuales de velocidad
    nu -- Viscosidad
    
    Retorna:
    J -- Matriz Jacobiana en formato disperso CSR
    F -- Vector de residuos
    """
    ny, nx = Ux.shape
    num_nodos = (nx-2)*(ny-2)
    F = np.zeros(num_nodos)
    J = lil_matrix((num_nodos, num_nodos))
    
    index_map = np.zeros(Ux.shape, dtype=int)
    index_map[1:-1, 1:-1] = np.arange(num_nodos).reshape(ny-2, nx-2)
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            idx = index_map[j, i]
            
            # Valores de los vecinos
            vecino_der = Ux[j, i+1]
            vecino_izq = Ux[j, i-1]
            vecino_sup = Ux[j+1, i]
            vecino_inf = Ux[j-1, i]
            
            # Calcular vorticidad: omega = -(dU_x/dy)
            omega = -(vecino_sup - vecino_inf) / 2.0
            
            # Término convectivo
            termino_convectivo = 0.125 * Ux[j,i] * (vecino_der - vecino_izq)
            
            # Ecuación discretizada
            F[idx] = (Ux[j,i] 
                     - 0.25 * (vecino_der + vecino_izq + vecino_sup + vecino_inf) 
                     + termino_convectivo 
                     - nu * omega)
            
            # Jacobiano
            J[idx, idx] = 1 - 0.125 * (vecino_der - vecino_izq)
            
            if i+1 < nx-1:
                J[idx, index_map[j, i+1]] = -0.25 + 0.125 * Ux[j,i]
            if i-1 > 0:
                J[idx, index_map[j, i-1]] = -0.25 - 0.125 * Ux[j,i]
            
            if j+1 < ny-1:
                J[idx, index_map[j+1, i]] = -0.25 + nu * 0.5
            if j-1 > 0:
                J[idx, index_map[j-1, i]] = -0.25 - nu * 0.5
    
    return csr_matrix(J), F

def gradiente_conjugado_solver(Ux, max_iter=15000, tol=1e-3):
    """
    Resuelve el sistema no lineal usando Gradiente Conjugado
    
    Parámetros:
    Ux -- Matriz 2D para velocidad
    max_iter -- Máximo de iteraciones
    tol -- Tolerancia
    
    Retorna:
    Ux -- Matriz convergida
    errors -- Lista de errores (norma del residuo) por iteración
    """
    ny, nx = Ux.shape
    Ux_new = Ux.copy()
    num_nodos = (nx-2)*(ny-2)
    
    # Lista para almacenar los errores
    errors = []
    
    J, F = build_system(Ux_new)
    r = -F
    p = r.copy()
    max_diff = np.linalg.norm(r)
    errors.append(max_diff)
    print(f"Iter 0: Error = {max_diff:.3e}")
    
    for iter in range(max_iter):
        if max_diff < tol:
            print(f"Convergencia alcanzada en {iter+1} iteraciones")
            break
        
        Ap = J.dot(p)
        rr = np.dot(r, r)
        if np.abs(np.dot(p, Ap)) < 1e-10:
            print(f"Advertencia: Producto p^T * Ap cercano a cero en iteración {iter+1}")
            break
        alpha = rr / np.dot(p, Ap)
        
        Ux_interior = Ux_new[1:-1, 1:-1].flatten()
        Ux_interior += alpha * p
        Ux_new[1:-1, 1:-1] = Ux_interior.reshape(ny-2, nx-2)
        
        J, F = build_system(Ux_new)
        r_new = -F
        beta = np.dot(r_new, r_new) / rr if rr > 1e-10 else 0
        r = r_new
        p = r + beta * p
        
        max_diff = np.linalg.norm(r_new)
        errors.append(max_diff)
        print(f"Iter {iter+1}: Error = {max_diff:.3e}")
    
    return Ux_new, errors

def visualizar_matriz(matriz, title):
    """
    Visualiza una matriz 2D como heatmap
    """
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        matriz,
        annot=False,
        fmt=".1f",
        cmap="viridis",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={'label': 'Velocidad (Ux)'}
    )
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Dirección X', fontsize=12)
    ax.set_ylabel('Dirección Y', fontsize=12)
    ax.set_xticks(np.arange(matriz.shape[1]) + 0.5)
    ax.set_xticklabels(np.arange(matriz.shape[1]))
    ax.set_yticks(np.arange(matriz.shape[0]) + 0.5)
    ax.set_yticklabels(np.arange(matriz.shape[0]-1, -1, -1))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def graficar_convergencia(errors):
    """
    Genera un gráfico de convergencia mostrando el error vs. iteraciones
    
    Parámetros:
    errors -- Lista de errores (norma del residuo) por iteración
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(errors)), errors, 'b-', label='Norma del Residuo')
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Error (Norma del Residuo)', fontsize=12)
    plt.title('Convergencia del Método de Gradiente Conjugado', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parámetros
    tamanno_n, tamanno_m = 100, 10
    initial_velocity = 1.0
    
    # Configurar malla inicial
    Ux = setup_grid(tamanno_n, tamanno_m, initial_velocity)
    
    # Resolver y obtener los errores
    solucion, errors = gradiente_conjugado_solver(Ux, max_iter=2500, tol=1e-3)

    print(errors)
    
    # Visualizar la distribución de velocidades
    visualizar_matriz(
        matriz=np.round(solucion, 7),
        title=f"Distribución de Velocidades - Gradiente Conjugado\n(Rejilla {tamanno_n}x{tamanno_m}, Ux={initial_velocity})"
    )
    
    # Graficar la convergencia
    graficar_convergencia(errors)
