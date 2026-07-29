import re
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

def parsear_log(archivo_log):
    """Lee el log de texto y extrae las métricas en un DataFrame."""
    if not os.path.exists(archivo_log):
        print(f"❌ Error: No encuentro el archivo '{archivo_log}'")
        print("   -> Crea un archivo de texto con el contenido del log y llámalo así.")
        sys.exit()

    with open(archivo_log, 'r') as f:
        content = f.read()

    # Expresión regular para encontrar pares | clave | valor |
    pattern = re.compile(r"\|\s+([a-zA-Z0-9_]+)\s+\|\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+\|")

    data = []
    current_step = {}
    
    # Dividimos por bloques de iteración
    lines = content.split('\n')
    for line in lines:
        if "ep_len_mean" in line and "ep_len_mean" in current_step:
            data.append(current_step)
            current_step = {}

        match = pattern.search(line)
        if match:
            key = match.group(1)
            value = float(match.group(2))
            current_step[key] = value
            
    if current_step:
        data.append(current_step)

    return pd.DataFrame(data)

def graficar_metricas(df):
    """Genera una imagen con 3 filas de gráficas."""
    
    # Estilo compatible
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Entrenamiento PPO (Log Terminal)', fontsize=16, fontweight='bold')

    # --- CORRECCIÓN CLAVE: Convertir a Numpy explícitamente ---
    # Esto evita el error "Multi-dimensional indexing..."
    x = df['total_timesteps'].to_numpy()

    # 1. RECOMPENSA
    y_rew = df['ep_rew_mean'].to_numpy()
    axs[0, 0].plot(x, y_rew, color='blue', linewidth=2, marker='o', markersize=3)
    axs[0, 0].set_title('Recompensa Media por Episodio', fontsize=12, color='darkblue')
    axs[0, 0].set_ylabel('Puntos')
    axs[0, 0].grid(True, alpha=0.3)

    # 2. DURACIÓN
    y_len = df['ep_len_mean'].to_numpy()
    axs[0, 1].plot(x, y_len, color='green', linewidth=2)
    axs[0, 1].set_title('Duración Media (Pasos)', fontsize=12, color='darkgreen')
    axs[0, 1].set_ylabel('Pasos')
    axs[0, 1].grid(True, alpha=0.3)

    # 3. VALUE LOSS
    y_vloss = df['value_loss'].to_numpy()
    axs[1, 0].plot(x, y_vloss, color='red', alpha=0.7)
    axs[1, 0].set_title('Value Loss (Error de Predicción)', fontsize=12, color='darkred')
    axs[1, 0].set_ylabel('Loss')
    
    # 4. ENTROPY LOSS
    y_ent = df['entropy_loss'].to_numpy()
    axs[1, 1].plot(x, y_ent, color='purple', alpha=0.7)
    axs[1, 1].set_title('Entropy Loss (Exploración)', fontsize=12, color='purple')
    axs[1, 1].set_ylabel('Entropía')

    # 5. EXPLAINED VARIANCE
    y_exp = df['explained_variance'].to_numpy()
    axs[2, 0].plot(x, y_exp, color='orange', linewidth=2)
    axs[2, 0].set_title('Explained Variance (Comprensión)', fontsize=12, color='darkorange')
    axs[2, 0].axhline(y=1, color='black', linestyle='--', alpha=0.3)

    # 6. STD
    if 'std' in df.columns:
        y_std = df['std'].to_numpy()
        axs[2, 1].plot(x, y_std, color='gray', linewidth=2)
        axs[2, 1].set_title('Desviación Estándar (Inseguridad)', fontsize=12, color='black')
        axs[2, 1].set_ylabel('Std Dev')
    else:
        axs[2, 1].text(0.5, 0.5, 'No disponible', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = "analisis_completo_tfg.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n✅ ¡Gráficas generadas! Abre el archivo: {output_file}")
    plt.show()

if __name__ == "__main__":
    archivo = "log_final.txt"
    print(f"📊 Leyendo {archivo}...")
    
    try:
        df = parsear_log(archivo)
        print(f"✅ Se han extraído {len(df)} iteraciones.")
        print("📊 Generando gráficas...")
        graficar_metricas(df)
    except Exception as e:
        print(f"❌ Error al procesar: {e}")
        # Imprimir el error completo para debug
        import traceback
        traceback.print_exc()