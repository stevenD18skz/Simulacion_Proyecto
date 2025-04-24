import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import lil_matrix, csr_matrix


def setup_grid(nx, ny, initial_velocity):
    """
    Configura la malla computacional con condiciones de contorno.

    Parámetros:
    - nx: número de nodos en la dirección x (columnas)
    - ny: número de nodos en la dirección y (filas)
    - initial_velocity: valor de Ux en el borde izquierdo

    Retorna:
    - Ux: matriz (ny × nx) con velocidades iniciales y condiciones de contorno aplicadas
    """
    # Inicializar matriz de velocidad con ceros
    Ux = np.zeros((ny, nx))
    # Aplicar condiciones de contorno:
    Ux[:, 0] = initial_velocity  # Borde izquierdo: velocidad de entrada
    Ux[:, -1] = 0.0              # Borde derecho: velocidad cero (salida)
    Ux[0, :] = 0.0               # Borde inferior: no slip
    Ux[-1, :] = 0.0              # Borde superior: no slip
    return Ux


def build_system(Ux, vort_const):
    """
    Construye el vector de residuos F y la matriz Jacobiana J para el sistema no lineal,
    incluyendo el término de vorticidad parametrizado.

    Parámetros:
    - Ux: matriz 2D con valores actuales de la velocidad en la malla
    - vort_const: constante de vorticidad (por ejemplo, 0.1)

    Retorna:
    - J: matriz Jacobiana en formato disperso CSR
    - F: vector de residuos (1D)
    """
    ny, nx = Ux.shape
    num_nodos = (nx - 2) * (ny - 2)  # nodos interiores

    # Inicializar vector de residuos y Jacobiana dispersa
    F = np.zeros(num_nodos)
    J = lil_matrix((num_nodos, num_nodos))

    # Mapeo de índices 2D a 1D para nodos interiores
    index_map = np.zeros_like(Ux, dtype=int)
    index_map[1:-1, 1:-1] = np.arange(num_nodos).reshape(ny - 2, nx - 2)

    # Recorrer cada nodo interior
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            idx = index_map[j, i]
            # Velocidades de vecinos
            ve, vw = Ux[j, i + 1], Ux[j, i - 1]
            vn, vs = Ux[j + 1, i], Ux[j - 1, i]

            # Término de vorticidad: 0.5 * vort_const * (vn - vs)
            vorticity_term = 0.5 * vort_const * (vn - vs)
            # Término convectivo no lineal, se resta la vorticidad
            convective_term = 0.125 * Ux[j, i] * (ve - vw) - vorticity_term

            # Residuo: diferencia con promedio de vecinos más término convectivo
            F[idx] = Ux[j, i] - 0.25 * (ve + vw + vn + vs) + convective_term

            # Montaje de Jacobiana:
            # derivada parcial respecto a Ux[j,i]
            J[idx, idx] = 1 - 0.125 * (ve - vw)
            # derivadas parciales respecto a vecinos
            if i + 1 < nx - 1:
                J[idx, index_map[j, i + 1]] = -0.25 + 0.125 * Ux[j, i]
            if i - 1 > 0:
                J[idx, index_map[j, i - 1]] = -0.25 - 0.125 * Ux[j, i]
            if j + 1 < ny - 1:
                J[idx, index_map[j + 1, i]] = -0.25 + 0.0125
            if j - 1 > 0:
                J[idx, index_map[j - 1, i]] = -0.25 - 0.0125

    return csr_matrix(J), F


def gradient_descent(nx=5, ny=5, vort_const=0.1, lr=1e-2, tol=1e-6, max_iter=50):
    """
    Aplica el método de gradiente descendente para resolver F(U)=0,
    parametrizando la vorticidad y usando tolerancia y número máximo de iteraciones.

    Parámetros:
    - nx, ny: dimensiones de la rejilla (nodos en x e y)
    - vort_const: constante de vorticidad (igual que en código de Newton-Raphson)
    - lr: tasa de aprendizaje (learning rate)
    - tol: tolerancia para la norma del residuo ||F|| (criterio de convergencia)
    - max_iter: número máximo de iteraciones permitidas

    Retorna:
    - Ux: matriz 2D con la solución aproximada
    """
    # 1) Inicializar malla con condiciones de contorno
    Ux = setup_grid(nx, ny, initial_velocity=1)

    # 2) Ciclo de gradiente descendente
    for iteracion in range(1, max_iter + 1):
        # 2.1) Construir sistema y calcular gradiente
        J, F = build_system(Ux, vort_const)
        grad = J.T @ F  # gradiente de 1/2 ||F||^2

        # 2.2) Actualizar solo nodos interiores con paso de gradiente
        delta = lr * grad
        interior = Ux[1:-1, 1:-1].flatten()
        interior -= delta
        Ux[1:-1, 1:-1] = interior.reshape(ny - 2, nx - 2)

        # 2.3) Calcular norma del residuo y chequear convergencia
        normF = np.linalg.norm(F)
        print(f"Iteración {iteracion}: ||F|| = {normF:.3e}")
        if normF < tol:
            print(f"Convergencia lograda: ||F|| < {tol} en iteración {iteracion}")
            break
        if iteracion == max_iter:
            print(f"Máximo de iteraciones ({max_iter}) alcanzado. ||F|| = {normF:.3e}")

    return Ux


def visualizar_matriz(matriz, titulo):
    """
    Visualiza la matriz de velocidades como un heatmap con orientación física.

    Parámetros:
    - matriz: array 2D con valores de Ux
    - titulo: texto para el título del gráfico
    """
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        matriz,
        cmap="viridis",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={'label': 'Ux'}
    )
    ax.set_title(titulo)
    ax.set_xlabel("Dirección X (izq→der)")
    ax.set_ylabel("Dirección Y (inf→sup)")
    ax.set_xticks(np.arange(matriz.shape[1]) + 0.5)
    ax.set_xticklabels(np.arange(matriz.shape[1]))
    ax.set_yticks(np.arange(matriz.shape[0]) + 0.5)
    ax.set_yticklabels(np.arange(matriz.shape[0] - 1, -1, -1))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parámetros de simulación
    nx, ny = 30, 15       # resolución de la rejilla
    vort_const = 0.1      # constante de vorticidad (igual que en Newton-Raphson)
    lr = 1e-2             # learning rate
    tol = 1e-6            # tolerancia para ||F||
    max_iter = 5000         # iteraciones máximas

    # Ejecutar gradiente descendente con vorticidad parametrizada
    solucion = gradient_descent(nx=nx, ny=ny, vort_const=vort_const,
                                lr=lr, tol=tol, max_iter=max_iter)

    # Mostrar resultado final
    print("\nSolución final (Orientación física):")
    print(solucion)
    visualizar_matriz(solucion,
                     f"Gradiente Descendente {nx}×{ny} | vorticidad={vort_const} | lr={lr} | tol={tol}")
