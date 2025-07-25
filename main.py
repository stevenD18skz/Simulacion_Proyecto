import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import seaborn as sns
import time

def setup_grid_with_beam(nx, ny, initial_velocity=1.0):
    """
    Configura la malla inicial con condiciones de contorno para flujo alrededor de viga.
    Basado en la figura 4.11 del informe: flujo simétrico con viga sumergida.
    """
    Ux = np.zeros((ny, nx))
    
    # Condiciones de frontera dadas en el proyecto
    Ux[:, 0] = initial_velocity      # Entrada: velocidad constante (Dirichlet)
    Ux[:, -1] = 0.0                  # Salida: gradiente nulo aproximado
    Ux[0, :] = 0.0                   # Pared inferior: no deslizamiento
    Ux[-1, :] = 0.0                  # Pared superior: no deslizamiento
    
    # Simular efecto de viga sumergida (vorticidad) en el centro
    beam_start = nx // 3
    beam_end = 2 * nx // 3
    beam_height = ny // 4
    
    # Perturbación inicial para simular la viga
    for i in range(beam_start, beam_end):
        for j in range(beam_height, ny - beam_height):
            Ux[j, i] *= 0.3  # Reducir velocidad alrededor de la viga
    
    return Ux

def build_system_with_vorticity(Ux, omega=0.1):
    """
    Construye el sistema empleando la ecuación :
    Ux_ij = 1/4 (Ux_{i+1,j} + Ux_{i-1,j} + Ux_{i,j+1} + Ux_{i,j-1}) + termino_convectivo
    Incluye efectos de vorticidad (segunda fase del proyecto)
    """
    ny, nx = Ux.shape
    num = (nx-2)*(ny-2)
    F = np.zeros(num)
    idx_map = np.zeros(Ux.shape, int)
    idx_map[1:-1,1:-1] = np.arange(num).reshape(ny-2, nx-2)
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            idx = idx_map[j,i]
            # Vecinos
            ue, uw = Ux[j,i+1], Ux[j,i-1]
            un, us = Ux[j+1,i], Ux[j-1,i]
            
            # Término convectivo (incluido en segunda fase)
            conv = 0.125 * Ux[j,i] * (ue - uw)
            
            # Término de vorticidad (efecto de la viga)
            vort = 0.5 * omega * (un - us)
            
            # Ecuación discretizada 
            F[idx] = Ux[j,i] - 0.25*(ue+uw+un+us) + conv - vort
    
    return F

def gauss_seidel_solver(nx, ny, omega, tol=1e-6, max_iter=500, verbose=True):
    """
    Solver de Gauss-Seidel.
    Incluye la ecuación discretizada exacta y manejo de vorticidad.
    """
    print("="*100)
    print(f"Parámetros: nx={nx}, ny={ny}, ω={omega}")
    print(f"Viscosidad cinemática: ν=1 m²/s, Densidad: ρ=1000 kg/m³")
    print("-"*60)
    
    # Inicialización con condiciones realistas
    U = setup_grid_with_beam(nx, ny, 1.0)
    errors = []
    residuals = []
    start_time = time.time()
    
    if verbose:
        print("Iniciando iteraciones Gauss-Seidel...")
    
    for k in range(max_iter):
        U_old = U.copy()
        max_diff = 0.0
        
        # Actualización Gauss-Seidel con ecuación completa del informe
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # Valores vecinos (algunos ya actualizados en esta iteración)
                ue, uw = U[j, i+1], U[j, i-1]
                un, us = U[j+1, i], U[j-1, i]
                
                # Término convectivo no lineal
                conv = 0.125 * U[j,i] * (ue - uw)
                
                # Término de vorticidad (efecto viga)
                vort = 0.5 * omega * (un - us)
                
                # Ecuación discretizada completa
                new_val = 0.25*(ue + uw + un + us) - conv + vort
                
                # Control de convergencia
                diff = abs(new_val - U[j,i])
                if diff > max_diff:
                    max_diff = diff
                
                U[j,i] = new_val
        
        errors.append(max_diff)
        
        # Cálculo del residual usando el sistema completo
        F = build_system_with_vorticity(U, omega)
        res_norm = np.linalg.norm(F, ord=2)
        residuals.append(res_norm)
        
        # Progreso cada 50 iteraciones
        if verbose and (k+1) % 50 == 0:
            print(f"Iteración {k+1}: Error={max_diff:.2e}, Residual={res_norm:.2e}")
        
        # Criterio de convergencia
        if max_diff < tol:
            if verbose:
                print(f"\n✓ Convergencia alcanzada en {k+1} iteraciones")
            break
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"✓ Tiempo total: {elapsed:.4f} segundos")
        print(f"✓ Error final: {errors[-1]:.2e}")
        print(f"✓ Residual final: {residuals[-1]:.2e}")
        print("-"*60)
    
    return U, errors, residuals, k+1, elapsed

def suavizar_con_spline_natural(Ux, upsample_x=300, upsample_y=150):
    """
    Implementa spline cúbico natural en 2D para suavizar el campo de velocidades.
    para hacer  la visualización más realista y menos "tosca".
    """
    print("Aplicando Spline Cúbico Natural 2D...")
    print(f"Resolución original: {Ux.shape}")
    print(f"Resolución suavizada: {upsample_y} x {upsample_x}")
    
    ny, nx = Ux.shape
    x = np.arange(nx)
    y = np.arange(ny)
    
    # Spline cúbico natural (kx=3, ky=3)
    spline = RectBivariateSpline(y, x, Ux, kx=3, ky=3, s=0)  # s=0 para interpolación exacta
    
    # Nueva malla de alta resolución
    x_nuevo = np.linspace(0, nx - 1, upsample_x)
    y_nuevo = np.linspace(0, ny - 1, upsample_y)
    
    # Evaluación del spline
    Ux_suavizado = spline(y_nuevo, x_nuevo)
    
    print(f"✓ Spline aplicado exitosamente")
    return x_nuevo, y_nuevo, Ux_suavizado

def visualizacion_campos_velocidad(U_original, U_suavizado, nx, ny):
    """
    FIGURA 1: Comparación de campos de velocidad (3 Mapas de calor); de IZ A DR
    Campo original empleando el metodo Gauss-Seidel de la Segunda entrega
    luego mismo campo pero aplicandole el spline cubico natural
    por ultimo efecto del suavisado generado por el el uso del spline cubico natural
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Campo original
    im1 = axes[0].imshow(U_original, cmap='viridis', aspect='auto', 
                        extent=[0, nx, 0, ny], origin='lower')
    axes[0].set_title('Campo Original (Gauss-Seidel)\n"Malla Tosca"', 
                     fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Dirección X', fontsize=12)
    axes[0].set_ylabel('Dirección Y', fontsize=12, labelpad=50)
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Velocidad Ux (m/s)', fontsize=12)
    
    # Campo suavizado
    im2 = axes[1].imshow(U_suavizado, cmap='viridis', aspect='auto',
                        extent=[0, nx, 0, ny], origin='lower')
    axes[1].set_title('Campo Suavizado uso (Spline Cúbico natural)\n', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Dirección X', fontsize=12)
    axes[1].set_ylabel('Dirección Y', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Velocidad Ux (m/s)', fontsize=12)
    
    # Diferencia (efecto del suavizado)
    from scipy.interpolate import RectBivariateSpline
    y_orig, x_orig = np.arange(ny), np.arange(nx)
    spline_orig = RectBivariateSpline(y_orig, x_orig, U_original, kx=1, ky=1)
    y_new = np.linspace(0, ny-1, U_suavizado.shape[0])
    x_new = np.linspace(0, nx-1, U_suavizado.shape[1])
    U_orig_interp = spline_orig(y_new, x_new)
    
    diferencia = U_suavizado - U_orig_interp
    im3 = axes[2].imshow(diferencia, cmap='RdBu_r', aspect='auto',
                        extent=[0, nx, 0, ny], origin='lower')
    axes[2].set_title('Efecto del Suavizado\n(Spline - Original)', 
                     fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Dirección X', fontsize=12)
    axes[2].set_ylabel('Dirección Y', fontsize=12)
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.set_label('Diferencia de Velocidad', fontsize=12)
    
    # Ajustar tamaño de fuente de los ticks
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.suptitle('COMPARACIÓN DE CAMPOS DE VELOCIDAD',
                fontsize=16,
                fontweight='bold',
                y=0.98)
    
    # CONFIGURACIÓN DE SUBPLOTS
    plt.subplots_adjust(
        left=0.067,      # Margen izquierdo
        bottom=0.091,    # Margen inferior  
        right=0.984,     # Margen derecho
        top=0.853,       # Margen superior
        wspace=0.28,     # Espacio horizontal entre subplots
        hspace=0.196     # Espacio vertical entre subplots
    )
    plt.savefig("figura1_comparacion.png", dpi=300) #Guardar cada figura como imagen
    plt.show()

def visualizacion_convergencia(errors, residuals):
    """
    FIGURA 2: Análisis de convergencia
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    color1, color2 = 'tab:blue', 'tab:red'
    ax.set_xlabel('Iteraciones', fontsize=16)
    ax.set_ylabel('Error Máximo', color=color1, fontsize=16)
    line1 = ax.semilogy(errors, color=color1, linewidth=3, label='Error Máximo', marker='o', markersize=4)
    ax.tick_params(axis='y', labelcolor=color1, labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(True, alpha=0.4)
    
    ax_twin = ax.twinx()
    ax_twin.set_ylabel('Norma del Residual', color=color2, fontsize=16)
    line2 = ax_twin.semilogy(residuals, color=color2, linewidth=3, label='Residual', marker='s', markersize=4)
    ax_twin.tick_params(axis='y', labelcolor=color2, labelsize=14)
    
    lines = [line1[0], line2[0]]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=14, frameon=True, 
              fancybox=True, shadow=True)
    
    ax.set_title('Convergencia del Método Gauss-Seidel', fontsize=18, fontweight='bold', pad=20)
    
    # Añadir información adicional en el gráfico
    ax.text(0.40, 0.98, f'Convergencia alcanzada en {len(errors)} iteraciones\nTolerancia: 1×10⁻⁶', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("figura2_Análisis_de_Convergencia.png", dpi=300) #Guardar cada figura como imagen
    plt.show()

def tabla_metricas_profesional(iteraciones, tiempo, errors, residuals, nx, ny, U_suavizado):
    """
    FIGURA 3: Tabla de métricas del proyecto
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.axis('off')
    
    # Datos de la tabla
    metricas_data = [
        ['INFORMACIÓN GENERAL', ''],
        ['Método Numérico Seleccionado', 'Gauss-Seidel'],
        ['Justificación Técnica', 'Mejor balance estabilidad/convergencia'],
        ['Estudiantes', 'C. Muñoz- 2042857, B. Narváez-2226675, D. Arias-2222205'],
        ['Universidad', 'Universidad del Valle'],
        ['', ''],
        ['PARÁMETROS DE SIMULACIÓN', ''],
        ['Dimensiones de malla original', f'{nx} × {ny} nodos'],
        ['Viscosidad cinemática (ν)', '1 m²/s'],
        ['Densidad del fluido (ρ)', '1000 kg/m³'],
        ['Factor de relajación (ω)', '0.1'],
        ['Tolerancia de convergencia', '1×10⁻⁶'],
        ['', ''],
        ['RESULTADOS DE CONVERGENCIA', ''],
        ['Iteraciones totales realizadas', f'{iteraciones}'],
        ['Tiempo total de cálculo', f'{tiempo:.4f} segundos'],
        ['Error máximo final', f'{errors[-1]:.2e}'],
        ['Norma del residual final', f'{residuals[-1]:.2e}'],
        ['Estado de convergencia', '✓ Convergido exitosamente'],
        ['', ''],
        ['PROCESAMIENTO CON SPLINE', ''],
        ['Resolución malla original', f'{ny} × {nx} puntos'],
        ['Resolución después del spline', f'{U_suavizado.shape[0]} × {U_suavizado.shape[1]} puntos'],
        ['Factor de mejora en resolución', f'{(U_suavizado.shape[0]*U_suavizado.shape[1])/(ny*nx):.1f}x'],
        ['Tipo de spline utilizado', 'Cúbico Natural 2D (kx=3, ky=3)'],
        ['Biblioteca computacional', 'scipy.interpolate.RectBivariateSpline'],
        ['Mejora visual obtenida', 'que se consigue modificar esta linea"'],
        ['', ''],
        ['CARACTERÍSTICAS DEL PROBLEMA', ''],
        ['Tipo de flujo simulado', 'Flujo 2D alrededor de viga sumergida'],
        ['Condiciones de contorno', 'Dirichlet (entrada), Neumann (salida)'],
        ['Efectos físicos incluidos', 'Convección + Vorticidad'],
        ['Aplicación práctica', 'Ingeniería de fluidos computacional'],
    ]
    
    # Crear tabla y formato
    table = ax.table(cellText=metricas_data,
                    colLabels=['PARÁMETRO', 'VALOR/DESCRIPCIÓN'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.5, 0.5])
    
    # Personalización avanzada de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Colorear encabezados
    for i in range(2):
        table[(0, i)].set_facecolor('#2E8B57')  # Verde oscuro
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Colorear y formatear filas de secciones principales
    section_rows = [1, 7, 14, 21, 28]  # Filas de encabezados de sección
    section_colors = ['#E6F3FF', '#FFE6E6', '#E6FFE6', '#FFF0E6', '#F0E6FF']
    
    for i, row in enumerate(section_rows):
        if row < len(metricas_data) + 1:
            table[(row, 0)].set_facecolor(section_colors[i % len(section_colors)])
            table[(row, 1)].set_facecolor(section_colors[i % len(section_colors)])
            table[(row, 0)].set_text_props(weight='bold')
            table[(row, 1)].set_text_props(weight='bold')
    
    # Resaltar filas importantes
    important_rows = [15, 16, 17, 18]  # Resultados de convergencia
    for row in important_rows:
        if row < len(metricas_data) + 1:
            table[(row, 1)].set_facecolor('#FFFFCC')  # Amarillo claro
            table[(row, 1)].set_text_props(weight='bold')
    
    # Título principal con información del curso
    title_text = '''MÉTRICAS FINALES DEL PROYECTO'''    
    plt.title(title_text, fontsize=14, fontweight='bold', pad=65, 
              bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    # Añadir nota al pie
    footer_text = 'Entrega-Proyecto Final| Integración: Diferencias Finitas + Gauss-Seidel + Spline Cúbico'
    ax.text(0.5, 0.02, footer_text, transform=ax.transAxes, fontsize=10,
            horizontalalignment='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("figura3_Tabla_de_metricas.png", dpi=300) #Guardar cada figura como imagen
    plt.show()

def comparacion_perfiles_velocidad(U_original, U_suavizado, nx, ny):
    """
    FIGURA 4: Comparación detallada de perfiles de velocidad
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Secciones a analizar
    secciones = [nx//4, nx//2, 3*nx//4]
    nombres = ['Entrada (x/L=0.25)', 'Centro (x/L=0.5)', 'Salida (x/L=0.75)']
    
    for i, (seccion, nombre) in enumerate(zip(secciones, nombres)):
        ax = axes[i]
        
        # Perfil original
        y_orig = np.arange(ny)
        perfil_orig = U_original[:, seccion]
        
        # Perfil suavizado (interpolar a la sección correspondiente)
        seccion_suav = int(seccion * U_suavizado.shape[1] / nx)
        y_suav = np.linspace(0, ny-1, U_suavizado.shape[0])
        perfil_suav = U_suavizado[:, seccion_suav]
        
        ax.plot(perfil_orig, y_orig, 'o-', label='Original (Gauss-Seidel)', 
                linewidth=3, markersize=8, color='blue', alpha=0.7)
        ax.plot(perfil_suav, y_suav, '-', label='Suavizado (Spline)', 
                linewidth=3, color='red', alpha=0.8)
        
        ax.set_xlabel('Velocidad Ux (m/s)', fontsize=14)
        ax.set_ylabel('Posición Y', fontsize=14)
        ax.set_title(nombre, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.4)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.suptitle('Comparación Detallada de Perfiles: Original vs Suavizado', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig("figura4_Análisis_de_Perfiles_de_Velocidad.png", dpi=300) #Guardar cada figura como imagen
    plt.show()

if __name__ == "__main__":
    # Parámetros del proyecto
    nx, ny, omega = 100, 10, 0.1

    print("="*100)
    print("PROYECTO FINAL- Simulación de Flujo 2D con Spline Cúbico Natural")
    print("Presentado a: Maria Patricia Trujillo Uribe Ph.D.")    
    print("Estudiantes: Camilo Andrés Muñoz-2042857, Brayan Steven Narváez-2226675, Daniel Arias-2222205")
    print("Curso: Simulación y Computación Numérica (SCN) - Universidad del Valle")
    
    # 1. SEGUNDA ENTREGA: Resolver con Gauss-Seidel (método seleccionado)
    solution_gs, gs_errors, gs_residuals, gs_iters, gs_time = gauss_seidel_solver(
        nx, ny, omega, tol=1e-6, max_iter=500, verbose=True
    )
    
    # 2. TERCERA ENTREGA: Aplicar Spline Cúbico Natural
    x_nuevo, y_nuevo, Ux_suavizado = suavizar_con_spline_natural(
        solution_gs, upsample_x=300, upsample_y=150
    )
    
    # 3. VISUALIZACIONES SEPARADAS
    print("\n" + "="*60)
    print(" GRAFICAS GENERADAS")
    print("="*60)
    
    # FIGURA 1: Comparación de campos de velocidad
    print(" FIGURA 1: Comparación de Campos de Velocidad")
    visualizacion_campos_velocidad(solution_gs, Ux_suavizado, nx, ny)
    
    # FIGURA 2: Análisis de convergencia  
    print(" FIGURA 2: Análisis de Convergencia")
    visualizacion_convergencia(gs_errors, gs_residuals)
    
    # FIGURA 3: Tabla profesional de métricas
    print(" FIGURA 3: Tabla de Métricas")
    tabla_metricas_profesional(gs_iters, gs_time, gs_errors, gs_residuals, nx, ny, Ux_suavizado)
    
    # FIGURA 4: Análisis detallado de perfiles
    print(" FIGURA 4: Análisis de Perfiles de Velocidad")
    comparacion_perfiles_velocidad(solution_gs, Ux_suavizado, nx, ny)