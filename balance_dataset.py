"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - BALANCEO DE DATASET                             ║
║                                                                              ║
║   PASO 2: Balancea el dataset para mejorar detección de caídas              ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PROBLEMA:
=========
    Dataset actual: 1,664 caídas vs 9,136 ADL (ratio 5.5:1)
    El modelo aprende a predecir siempre "NORMAL"

SOLUCIÓN:
=========
    Este script balancea el dataset usando:
    1. Undersampling de ADL (reduce la clase mayoritaria)
    2. (Opcional) SMOTE para generar más caídas sintéticas
    3. Stratified sampling para mantener diversidad

ENTRADA:
========
    - keypoints_HEAVY.csv (dataset desbalanceado)

SALIDA:
=======
    - keypoints_balanced.csv (dataset balanceado)
    - balance_report.json (estadísticas)
    - balance_comparison.png (visualización antes/después)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Intentar importar SMOTE (opcional)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️ SMOTE no disponible. Instala con: pip install imbalanced-learn")


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CONFIGURACIÓN
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Ruta al CSV original (desbalanceado)
INPUT_CSV = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_keypoints_heavy\keypoints_dataset.csv"

# Carpeta de salida
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\train_model_v2"

# ═══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA DE BALANCEO
# ═══════════════════════════════════════════════════════════════════════════════
# Opciones:
#   - "undersample": Solo reduce ADL al mismo nivel que caídas (recomendado)
#   - "oversample": Usa SMOTE para generar caídas sintéticas
#   - "hybrid": Combina ambos (reduce ADL + aumenta caídas)

BALANCE_STRATEGY = "undersample"  # ← Recomendado para empezar

# Ratio objetivo (ADL:Caídas)
# 1.0 = igual cantidad de ambos (1:1)
# 1.5 = 50% más ADL que caídas (1.5:1)
# 2.0 = doble de ADL que caídas (2:1)
TARGET_RATIO = 1.0  # ← 1:1 es lo más balanceado

# Semilla para reproducibilidad
RANDOM_SEED = 42


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         FUNCIONES DE BALANCEO
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_dataset(csv_path):
    """
    Carga el dataset y muestra estadísticas iniciales.
    """
    print("\n" + "="*70)
    print("📂 CARGANDO DATASET ORIGINAL")
    print("="*70)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    total = len(df)
    falls = len(df[df['label'] == 1])
    adl = len(df[df['label'] == 0])
    ratio = adl / falls if falls > 0 else 0
    
    print(f"\n   📊 Estadísticas ANTES del balanceo:")
    print(f"   ├── Total:     {total:,} imágenes")
    print(f"   ├── 🔴 Caídas: {falls:,} ({falls/total*100:.1f}%)")
    print(f"   ├── 🟢 ADL:    {adl:,} ({adl/total*100:.1f}%)")
    print(f"   └── Ratio:     {ratio:.1f}:1 (ADL:Caídas)")
    
    if ratio > 2:
        print(f"\n   ⚠️  ALERTA: Dataset muy desbalanceado!")
    
    return df, {'total': total, 'falls': falls, 'adl': adl, 'ratio': ratio}


def undersample_majority(df, target_ratio=1.0, random_seed=42):
    """
    Reduce la clase mayoritaria (ADL) usando undersampling estratificado.
    Mantiene diversidad seleccionando de diferentes carpetas.
    """
    print("\n" + "="*70)
    print("⚖️ APLICANDO UNDERSAMPLING")
    print("="*70)
    
    # Separar clases
    df_falls = df[df['label'] == 1].copy()
    df_adl = df[df['label'] == 0].copy()
    
    n_falls = len(df_falls)
    n_adl_target = int(n_falls * target_ratio)
    
    print(f"\n   📋 Plan de balanceo:")
    print(f"   ├── Caídas actuales:    {n_falls:,} (se mantienen todas)")
    print(f"   ├── ADL actuales:       {len(df_adl):,}")
    print(f"   ├── ADL objetivo:       {n_adl_target:,}")
    print(f"   └── ADL a eliminar:     {len(df_adl) - n_adl_target:,}")
    
    # Stratified sampling por carpeta para mantener diversidad
    if 'folder' in df_adl.columns:
        print("\n   🔄 Usando muestreo estratificado por carpeta...")
        
        # Calcular cuántas muestras tomar de cada carpeta
        folder_counts = df_adl['folder'].value_counts()
        total_adl = len(df_adl)
        
        sampled_adl = []
        
        for folder, count in folder_counts.items():
            # Proporción de esta carpeta en el total
            proportion = count / total_adl
            # Cuántas muestras tomar de esta carpeta
            n_samples = max(1, int(n_adl_target * proportion))
            n_samples = min(n_samples, count)  # No más de las disponibles
            
            folder_df = df_adl[df_adl['folder'] == folder]
            sampled = folder_df.sample(n=n_samples, random_state=random_seed)
            sampled_adl.append(sampled)
        
        df_adl_balanced = pd.concat(sampled_adl, ignore_index=True)
        
        # Ajustar si nos pasamos o quedamos cortos
        current_n = len(df_adl_balanced)
        if current_n > n_adl_target:
            df_adl_balanced = df_adl_balanced.sample(n=n_adl_target, random_state=random_seed)
        elif current_n < n_adl_target:
            # Tomar más muestras aleatorias
            remaining = n_adl_target - current_n
            already_sampled = set(df_adl_balanced.index)
            available = df_adl[~df_adl.index.isin(already_sampled)]
            if len(available) >= remaining:
                extra = available.sample(n=remaining, random_state=random_seed)
                df_adl_balanced = pd.concat([df_adl_balanced, extra], ignore_index=True)
    else:
        # Muestreo aleatorio simple
        df_adl_balanced = df_adl.sample(n=n_adl_target, random_state=random_seed)
    
    # Combinar
    df_balanced = pd.concat([df_falls, df_adl_balanced], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"\n   ✅ Undersampling completado")
    
    return df_balanced


def oversample_minority(df, target_ratio=1.0, random_seed=42):
    """
    Aumenta la clase minoritaria (caídas) usando SMOTE.
    Genera muestras sintéticas basadas en vecinos cercanos.
    """
    print("\n" + "="*70)
    print("⚖️ APLICANDO OVERSAMPLING (SMOTE)")
    print("="*70)
    
    if not SMOTE_AVAILABLE:
        print("   ❌ SMOTE no disponible. Usando undersampling en su lugar.")
        return undersample_majority(df, target_ratio, random_seed)
    
    # Separar features y labels
    metadata_cols = ['filename', 'folder', 'dataset', 'label', 'label_name']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Calcular cantidad objetivo
    n_adl = len(df[df['label'] == 0])
    n_falls = len(df[df['label'] == 1])
    n_falls_target = int(n_adl / target_ratio)
    
    print(f"\n   📋 Plan de balanceo:")
    print(f"   ├── ADL actuales:       {n_adl:,} (se mantienen)")
    print(f"   ├── Caídas actuales:    {n_falls:,}")
    print(f"   ├── Caídas objetivo:    {n_falls_target:,}")
    print(f"   └── Caídas a generar:   {n_falls_target - n_falls:,} (sintéticas)")
    
    # Aplicar SMOTE
    smote = SMOTE(
        sampling_strategy={1: n_falls_target},
        random_state=random_seed,
        k_neighbors=5
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"\n   ✅ SMOTE completado")
    print(f"   └── Nuevas muestras sintéticas: {len(X_resampled) - len(X):,}")
    
    # Reconstruir DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
    df_resampled['label'] = y_resampled
    df_resampled['label_name'] = df_resampled['label'].map({0: 'adl', 1: 'fall'})
    df_resampled['filename'] = 'synthetic'
    df_resampled['folder'] = 'synthetic'
    df_resampled['dataset'] = 'synthetic'
    
    # Restaurar metadatos originales donde sea posible
    original_indices = range(len(df))
    for col in ['filename', 'folder', 'dataset']:
        df_resampled.loc[original_indices, col] = df[col].values
    
    return df_resampled


def hybrid_balance(df, target_ratio=1.0, random_seed=42):
    """
    Estrategia híbrida: reduce ADL y aumenta caídas.
    Objetivo: llegar a un punto medio.
    """
    print("\n" + "="*70)
    print("⚖️ APLICANDO BALANCEO HÍBRIDO")
    print("="*70)
    
    n_falls = len(df[df['label'] == 1])
    n_adl = len(df[df['label'] == 0])
    
    # Calcular punto medio
    target_per_class = int((n_falls + n_adl) / (1 + target_ratio) * target_ratio)
    target_per_class = max(target_per_class, n_falls)  # No menos que las caídas actuales
    target_per_class = min(target_per_class, n_adl)    # No más que las ADL actuales
    
    print(f"\n   📋 Plan híbrido:")
    print(f"   ├── Objetivo por clase: ~{target_per_class:,}")
    
    # Paso 1: Undersample ADL
    df_falls = df[df['label'] == 1].copy()
    df_adl = df[df['label'] == 0].copy()
    
    df_adl_sampled = df_adl.sample(n=target_per_class, random_state=random_seed)
    
    # Paso 2: Oversample caídas si es necesario y SMOTE disponible
    if SMOTE_AVAILABLE and n_falls < target_per_class:
        # Crear dataset temporal para SMOTE
        df_temp = pd.concat([df_falls, df_adl_sampled], ignore_index=True)
        
        metadata_cols = ['filename', 'folder', 'dataset', 'label', 'label_name']
        feature_cols = [col for col in df_temp.columns if col not in metadata_cols]
        
        X = df_temp[feature_cols].values
        y = df_temp['label'].values
        
        smote = SMOTE(
            sampling_strategy={1: target_per_class},
            random_state=random_seed,
            k_neighbors=min(5, n_falls - 1)
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        df_balanced = pd.DataFrame(X_resampled, columns=feature_cols)
        df_balanced['label'] = y_resampled
        df_balanced['label_name'] = df_balanced['label'].map({0: 'adl', 1: 'fall'})
        df_balanced['filename'] = 'mixed'
        df_balanced['folder'] = 'mixed'
        df_balanced['dataset'] = 'mixed'
        
        # Restaurar metadatos originales
        original_len = len(df_temp)
        for col in ['filename', 'folder', 'dataset']:
            df_balanced.loc[:original_len-1, col] = df_temp[col].values
    else:
        # Solo undersample
        df_balanced = pd.concat([df_falls, df_adl_sampled], ignore_index=True)
    
    df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"   ✅ Balanceo híbrido completado")
    
    return df_balanced


def balance_dataset(df, strategy, target_ratio, random_seed):
    """
    Aplica la estrategia de balanceo seleccionada.
    """
    if strategy == "undersample":
        return undersample_majority(df, target_ratio, random_seed)
    elif strategy == "oversample":
        return oversample_minority(df, target_ratio, random_seed)
    elif strategy == "hybrid":
        return hybrid_balance(df, target_ratio, random_seed)
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")


def create_comparison_visualization(stats_before, stats_after, output_folder):
    """
    Crea gráfico comparativo antes/después del balanceo.
    """
    print("\n📊 Generando visualización comparativa...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('🛡️ SafeGuard Vision AI - Balanceo del Dataset\nAntes vs Después', 
                 fontsize=14, fontweight='bold')
    
    # 1. Barras comparativas
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.35
    
    before_vals = [stats_before['falls'], stats_before['adl']]
    after_vals = [stats_after['falls'], stats_after['adl']]
    
    bars1 = ax1.bar(x - width/2, before_vals, width, label='Antes', color=['#e74c3c', '#2ecc71'], alpha=0.5)
    bars2 = ax1.bar(x + width/2, after_vals, width, label='Después', color=['#c0392b', '#27ae60'])
    
    ax1.set_ylabel('Cantidad de Imágenes')
    ax1.set_title('Cantidad por Clase', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Caídas', 'ADL'])
    ax1.legend()
    
    # Añadir valores
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Pie charts
    ax2 = axes[1]
    colors = ['#e74c3c', '#2ecc71']
    ax2.pie([stats_before['falls'], stats_before['adl']], 
            labels=['Caídas', 'ADL'], colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0))
    ax2.set_title(f'ANTES\nRatio {stats_before["ratio"]:.1f}:1', fontweight='bold')
    
    ax3 = axes[2]
    ax3.pie([stats_after['falls'], stats_after['adl']], 
            labels=['Caídas', 'ADL'], colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0))
    ax3.set_title(f'DESPUÉS\nRatio {stats_after["ratio"]:.1f}:1', fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar
    plot_path = os.path.join(output_folder, "balance_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Gráfico guardado: {plot_path}")
    
    return plot_path


def save_balanced_dataset(df, stats_before, stats_after, output_folder, strategy):
    """
    Guarda el dataset balanceado y el reporte.
    """
    print("\n" + "="*70)
    print("💾 GUARDANDO RESULTADOS")
    print("="*70)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Guardar CSV balanceado
    csv_path = os.path.join(output_folder, "keypoints_balanced.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n   ✅ CSV balanceado: {csv_path}")
    
    # Guardar reporte
    report = {
        "project": "SafeGuard Vision AI",
        "step": "PASO 2 - Balanceo de Dataset",
        "date": datetime.now().isoformat(),
        "strategy": strategy,
        "target_ratio": TARGET_RATIO,
        "before": stats_before,
        "after": stats_after,
        "improvement": {
            "ratio_reduction": f"{stats_before['ratio']:.1f}:1 → {stats_after['ratio']:.1f}:1",
            "balance_achieved": stats_after['ratio'] <= 2.0
        },
        "next_step": "PASO 3: Re-entrenar modelo con train_fall_detector_v2.py"
    }
    
    report_path = os.path.join(output_folder, "balance_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Reporte: {report_path}")
    
    return csv_path, report_path


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                              MAIN
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SAFEGUARD VISION AI - BALANCEO DE DATASET ".center(70) + "║")
    print("║" + " PASO 2: Corregir desbalance de clases ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    print(f"\n   ⚙️  Configuración:")
    print(f"   ├── Estrategia: {BALANCE_STRATEGY}")
    print(f"   ├── Ratio objetivo: {TARGET_RATIO}:1")
    print(f"   └── Semilla: {RANDOM_SEED}")
    
    # Cargar dataset
    df_original, stats_before = load_dataset(INPUT_CSV)
    
    # Aplicar balanceo
    df_balanced = balance_dataset(
        df_original, 
        strategy=BALANCE_STRATEGY,
        target_ratio=TARGET_RATIO,
        random_seed=RANDOM_SEED
    )
    
    # Calcular estadísticas después
    total_after = len(df_balanced)
    falls_after = len(df_balanced[df_balanced['label'] == 1])
    adl_after = len(df_balanced[df_balanced['label'] == 0])
    ratio_after = adl_after / falls_after if falls_after > 0 else 0
    
    stats_after = {
        'total': total_after,
        'falls': falls_after,
        'adl': adl_after,
        'ratio': ratio_after
    }
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("📊 RESULTADOS DEL BALANCEO")
    print("="*70)
    
    print(f"\n   ANTES                          DESPUÉS")
    print(f"   ─────                          ──────")
    print(f"   Total:  {stats_before['total']:>7,}              Total:  {stats_after['total']:>7,}")
    print(f"   Caídas: {stats_before['falls']:>7,}              Caídas: {stats_after['falls']:>7,}")
    print(f"   ADL:    {stats_before['adl']:>7,}              ADL:    {stats_after['adl']:>7,}")
    print(f"   Ratio:  {stats_before['ratio']:>7.1f}:1            Ratio:  {stats_after['ratio']:>7.1f}:1")
    
    # Crear visualización
    create_comparison_visualization(stats_before, stats_after, OUTPUT_FOLDER)
    
    # Guardar
    csv_path, report_path = save_balanced_dataset(
        df_balanced, stats_before, stats_after, OUTPUT_FOLDER, BALANCE_STRATEGY
    )
    
    # Resumen final
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " ✅ BALANCEO COMPLETADO ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📉 Ratio: {stats_before['ratio']:.1f}:1 → {stats_after['ratio']:.1f}:1".ljust(71) + "║")
    print(f"║  📊 Dataset: {stats_after['total']:,} imágenes balanceadas".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📂 Archivos guardados en: {OUTPUT_FOLDER}".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print("║  📋 SIGUIENTE PASO:".ljust(71) + "║")
    print("║     Ejecuta train_fall_detector_v2.py con el CSV balanceado".ljust(71) + "║")
    print("╚" + "═"*70 + "╝")
    
    # Recordatorio de rutas
    print(f"\n   💡 IMPORTANTE: Actualiza la ruta en el script de entrenamiento:")
    print(f"      CSV_PATH = r\"{csv_path}\"")
    print()


if __name__ == "__main__":
    main()
