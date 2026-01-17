"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🔍 SAFEGUARD VISION AI - ANÁLISIS DE DATASET                              ║
║                                                                              ║
║   Investiga qué imágenes fallaron y por qué el modelo no funciona bien      ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

OBJETIVO:
=========
Este script analiza:
1. Distribución de imágenes fallidas (caídas vs ADL)
2. Desbalance del dataset
3. Problemas potenciales que causan mal rendimiento
4. Genera recomendaciones específicas

ENTRADA:
========
    - keypoints_dataset.csv (imágenes exitosas)
    - failed_HEAVY.txt (imágenes que fallaron)

SALIDA:
=======
    - Reporte completo en consola
    - dataset_analysis.json (datos del análisis)
    - dataset_analysis.png (gráficos)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import Counter
from datetime import datetime

# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CONFIGURACIÓN
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Ruta al CSV de keypoints exitosos
CSV_PATH = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_keypoints_heavy\keypoints_dataset.csv"

# Ruta al archivo de imágenes fallidas
FAILED_PATH = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_keypoints_heavy\failed_images.txt"

# Carpeta de salida para el análisis
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_analysis"


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         FUNCIONES DE CLASIFICACIÓN
# ╚══════════════════════════════════════════════════════════════════════════════╝

def classify_path_le2i(path):
    """
    Clasifica una ruta del dataset le2i como caída o ADL.
    """
    path_lower = path.lower()
    
    # Buscar el nombre de la carpeta padre
    parts = path.replace('\\', '/').split('/')
    
    for part in parts:
        part_lower = part.lower()
        if part_lower.startswith('fall') or part_lower.startswith('likefall'):
            return 'fall', part
        elif part_lower in ['coffee_room', 'home', 'office', 'lecture_room']:
            # Estas son carpetas de ubicación, buscar subcarpeta
            continue
        elif any(x in part_lower for x in ['stand', 'sit', 'walk', 'lie', 'bend', 'squat']):
            return 'adl', part
    
    # Si contiene "fall" en cualquier parte
    if 'fall' in path_lower:
        return 'fall', 'unknown_fall'
    
    return 'adl', 'unknown_adl'


def classify_path_urfall(path):
    """
    Clasifica una ruta del dataset ur_fall como caída o ADL.
    """
    path_lower = path.lower()
    
    if 'emergencia' in path_lower or 'fall' in path_lower:
        return 'fall', 'emergencia'
    else:
        return 'adl', 'simulada'


def classify_path(path):
    """
    Clasifica cualquier ruta como caída o ADL.
    """
    path_lower = path.lower()
    
    # Determinar dataset
    if 'le2i' in path_lower:
        return classify_path_le2i(path)
    elif 'ur_fall' in path_lower:
        return classify_path_urfall(path)
    else:
        # Clasificación genérica
        if 'fall' in path_lower or 'emergencia' in path_lower:
            return 'fall', 'unknown'
        return 'adl', 'unknown'


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         FUNCIONES DE ANÁLISIS
# ╚══════════════════════════════════════════════════════════════════════════════╝

def analyze_successful_images(csv_path):
    """
    Analiza las imágenes que se procesaron exitosamente.
    """
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE IMÁGENES EXITOSAS (CSV)")
    print("="*70)
    
    if not os.path.exists(csv_path):
        print(f"❌ No se encontró: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    total = len(df)
    falls = len(df[df['label'] == 1])
    adl = len(df[df['label'] == 0])
    
    print(f"\n   📁 Total imágenes exitosas: {total:,}")
    print(f"\n   Distribución por clase:")
    print(f"   ├── 🔴 Caídas (label=1): {falls:,} ({falls/total*100:.1f}%)")
    print(f"   └── 🟢 ADL (label=0):    {adl:,} ({adl/total*100:.1f}%)")
    print(f"\n   📐 Ratio ADL:Caídas = {adl/falls:.1f}:1")
    
    # Por dataset
    print(f"\n   Por dataset:")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        falls_d = len(subset[subset['label'] == 1])
        adl_d = len(subset[subset['label'] == 0])
        print(f"   ├── {dataset}: {len(subset):,} ({falls_d} caídas, {adl_d} ADL)")
    
    # Por carpeta (top 10)
    print(f"\n   Top 10 carpetas con más imágenes:")
    folder_counts = df['folder'].value_counts().head(10)
    for folder, count in folder_counts.items():
        subset = df[df['folder'] == folder]
        label = 'CAÍDA' if subset['label'].iloc[0] == 1 else 'ADL'
        print(f"   ├── {folder}: {count:,} ({label})")
    
    return {
        'total': total,
        'falls': falls,
        'adl': adl,
        'ratio': adl/falls if falls > 0 else 0,
        'by_dataset': df.groupby('dataset').size().to_dict(),
        'by_folder': df['folder'].value_counts().to_dict()
    }


def analyze_failed_images(failed_path):
    """
    Analiza las imágenes que fallaron en la detección de pose.
    """
    print("\n" + "="*70)
    print("❌ ANÁLISIS DE IMÁGENES FALLIDAS")
    print("="*70)
    
    if not os.path.exists(failed_path):
        print(f"❌ No se encontró: {failed_path}")
        return None
    
    with open(failed_path, 'r') as f:
        failed_paths = [line.strip() for line in f if line.strip()]
    
    total_failed = len(failed_paths)
    print(f"\n   📁 Total imágenes fallidas: {total_failed:,}")
    
    # Clasificar cada imagen fallida
    failed_falls = 0
    failed_adl = 0
    failed_by_folder = Counter()
    failed_by_type = Counter()
    
    for path in failed_paths:
        label, folder_type = classify_path(path)
        
        if label == 'fall':
            failed_falls += 1
        else:
            failed_adl += 1
        
        failed_by_type[folder_type] += 1
        
        # Extraer nombre de carpeta
        parts = path.replace('\\', '/').split('/')
        for part in parts:
            if any(x in part.lower() for x in ['fall', 'stand', 'sit', 'walk', 'lie', 
                                                'coffee', 'home', 'office', 'emergencia',
                                                'simulada', 'lecture']):
                failed_by_folder[part] += 1
                break
    
    print(f"\n   Distribución de fallos por clase:")
    print(f"   ├── 🔴 Caídas fallidas: {failed_falls:,} ({failed_falls/total_failed*100:.1f}%)")
    print(f"   └── 🟢 ADL fallidas:    {failed_adl:,} ({failed_adl/total_failed*100:.1f}%)")
    
    print(f"\n   Top 15 carpetas con más fallos:")
    for folder, count in failed_by_folder.most_common(15):
        label, _ = classify_path(folder)
        label_str = '🔴 CAÍDA' if label == 'fall' else '🟢 ADL'
        print(f"   ├── {folder}: {count:,} ({label_str})")
    
    return {
        'total': total_failed,
        'falls': failed_falls,
        'adl': failed_adl,
        'by_folder': dict(failed_by_folder),
        'by_type': dict(failed_by_type)
    }


def calculate_detection_rates(success_stats, failed_stats):
    """
    Calcula las tasas de detección por clase.
    """
    print("\n" + "="*70)
    print("📈 TASAS DE DETECCIÓN POR CLASE")
    print("="*70)
    
    if not success_stats or not failed_stats:
        print("   ❌ No hay suficientes datos para calcular")
        return None
    
    # Total original por clase
    total_falls = success_stats['falls'] + failed_stats['falls']
    total_adl = success_stats['adl'] + failed_stats['adl']
    
    # Tasas de éxito
    fall_success_rate = success_stats['falls'] / total_falls * 100 if total_falls > 0 else 0
    adl_success_rate = success_stats['adl'] / total_adl * 100 if total_adl > 0 else 0
    
    print(f"\n   📊 Imágenes ORIGINALES totales:")
    print(f"   ├── 🔴 Caídas totales: {total_falls:,}")
    print(f"   └── 🟢 ADL totales:    {total_adl:,}")
    
    print(f"\n   ✅ Tasa de DETECCIÓN exitosa:")
    print(f"   ├── 🔴 Caídas: {success_stats['falls']:,}/{total_falls:,} = {fall_success_rate:.1f}%")
    print(f"   └── 🟢 ADL:    {success_stats['adl']:,}/{total_adl:,} = {adl_success_rate:.1f}%")
    
    # Análisis del problema
    print(f"\n   ⚠️  DIAGNÓSTICO:")
    
    if fall_success_rate < adl_success_rate:
        diff = adl_success_rate - fall_success_rate
        print(f"   └── BlazePose detecta {diff:.1f}% MENOS caídas que ADL")
        print(f"       Esto sesga el dataset hacia actividades normales!")
    
    if success_stats['ratio'] > 3:
        print(f"   └── Dataset MUY desbalanceado (ratio {success_stats['ratio']:.1f}:1)")
        print(f"       El modelo aprende a predecir siempre 'NORMAL'")
    
    return {
        'total_original_falls': total_falls,
        'total_original_adl': total_adl,
        'fall_detection_rate': fall_success_rate,
        'adl_detection_rate': adl_success_rate,
        'detection_gap': adl_success_rate - fall_success_rate
    }


def generate_recommendations(success_stats, failed_stats, detection_stats):
    """
    Genera recomendaciones específicas basadas en el análisis.
    """
    print("\n" + "="*70)
    print("💡 RECOMENDACIONES")
    print("="*70)
    
    recommendations = []
    
    # 1. Desbalance
    if success_stats and success_stats['ratio'] > 2:
        rec = {
            'priority': 'CRÍTICA',
            'issue': f"Dataset desbalanceado (ratio {success_stats['ratio']:.1f}:1)",
            'solution': 'Balancear a ratio 1:1 usando undersampling de ADL',
            'impact': 'El modelo deja de predecir siempre NORMAL'
        }
        recommendations.append(rec)
        print(f"\n   🔴 {rec['priority']}: {rec['issue']}")
        print(f"      Solución: {rec['solution']}")
    
    # 2. Muchas caídas fallidas
    if failed_stats and success_stats:
        total_falls = success_stats['falls'] + failed_stats['falls']
        fall_loss = failed_stats['falls'] / total_falls * 100
        
        if fall_loss > 20:
            rec = {
                'priority': 'ALTA',
                'issue': f"Se perdieron {failed_stats['falls']:,} caídas ({fall_loss:.1f}%)",
                'solution': 'Revisar por qué BlazePose no detecta esas poses',
                'impact': 'Más datos de caídas para entrenar'
            }
            recommendations.append(rec)
            print(f"\n   🟠 {rec['priority']}: {rec['issue']}")
            print(f"      Solución: {rec['solution']}")
    
    # 3. Threshold
    rec = {
        'priority': 'RÁPIDA',
        'issue': 'Threshold muy alto (0.5)',
        'solution': 'Bajar a 0.35 para más sensibilidad a caídas',
        'impact': 'Detecta más caídas (puede aumentar falsos positivos)'
    }
    recommendations.append(rec)
    print(f"\n   🟡 {rec['priority']}: {rec['issue']}")
    print(f"      Solución: {rec['solution']}")
    
    # 4. Overfitting
    rec = {
        'priority': 'ALTA',
        'issue': 'Overfitting (train_acc=100%, test_acc=98%)',
        'solution': 'Reducir complejidad: max_depth=10, min_samples_leaf=10',
        'impact': 'Mejor generalización a videos nuevos'
    }
    recommendations.append(rec)
    print(f"\n   🟠 {rec['priority']}: {rec['issue']}")
    print(f"      Solución: {rec['solution']}")
    
    # 5. Modelo temporal
    rec = {
        'priority': 'OPCIONAL',
        'issue': 'Random Forest solo ve 1 frame (no detecta movimiento)',
        'solution': 'Implementar LSTM para secuencias de frames',
        'impact': 'Detecta la TRANSICIÓN de caer, no solo estar en el suelo'
    }
    recommendations.append(rec)
    print(f"\n   🔵 {rec['priority']}: {rec['issue']}")
    print(f"      Solución: {rec['solution']}")
    
    return recommendations


def create_visualizations(success_stats, failed_stats, detection_stats, output_folder):
    """
    Crea gráficos del análisis.
    """
    print("\n📊 Generando visualizaciones...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('🔍 SafeGuard Vision AI - Análisis del Dataset\nDiagnóstico de Problemas', 
                 fontsize=14, fontweight='bold')
    
    # 1. Distribución de clases (exitosas)
    ax1 = fig.add_subplot(2, 3, 1)
    if success_stats:
        labels = ['Caídas\n(label=1)', 'ADL\n(label=0)']
        sizes = [success_stats['falls'], success_stats['adl']]
        colors = ['#e74c3c', '#2ecc71']
        explode = (0.05, 0)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax1.set_title('Distribución en CSV\n(Imágenes Exitosas)', fontweight='bold')
    
    # 2. Distribución de fallos
    ax2 = fig.add_subplot(2, 3, 2)
    if failed_stats:
        labels = ['Caídas\nFallidas', 'ADL\nFallidas']
        sizes = [failed_stats['falls'], failed_stats['adl']]
        colors = ['#c0392b', '#27ae60']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax2.set_title('Distribución de Fallos\n(No detectadas por BlazePose)', fontweight='bold')
    
    # 3. Comparación de tasas de detección
    ax3 = fig.add_subplot(2, 3, 3)
    if detection_stats:
        categories = ['Caídas', 'ADL']
        rates = [detection_stats['fall_detection_rate'], detection_stats['adl_detection_rate']]
        colors = ['#e74c3c', '#2ecc71']
        
        bars = ax3.bar(categories, rates, color=colors, edgecolor='black')
        ax3.set_ylabel('Tasa de Detección (%)')
        ax3.set_title('Tasa de Éxito BlazePose\npor Clase', fontweight='bold')
        ax3.set_ylim(0, 100)
        
        for bar, rate in zip(bars, rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', fontweight='bold')
    
    # 4. Desbalance visual
    ax4 = fig.add_subplot(2, 3, 4)
    if success_stats:
        categories = ['Caídas', 'ADL']
        counts = [success_stats['falls'], success_stats['adl']]
        colors = ['#e74c3c', '#2ecc71']
        
        bars = ax4.bar(categories, counts, color=colors, edgecolor='black')
        ax4.set_ylabel('Cantidad de Imágenes')
        ax4.set_title(f'Desbalance del Dataset\nRatio {success_stats["ratio"]:.1f}:1', fontweight='bold')
        
        # Línea de balance ideal
        ideal = (counts[0] + counts[1]) / 2
        ax4.axhline(y=ideal, color='orange', linestyle='--', label=f'Balance ideal ({int(ideal):,})')
        ax4.legend()
        
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{count:,}', ha='center', fontweight='bold')
    
    # 5. Top carpetas fallidas
    ax5 = fig.add_subplot(2, 3, 5)
    if failed_stats and failed_stats['by_folder']:
        top_folders = dict(Counter(failed_stats['by_folder']).most_common(8))
        folders = list(top_folders.keys())
        counts = list(top_folders.values())
        
        # Colorear por tipo
        colors = ['#e74c3c' if 'fall' in f.lower() else '#2ecc71' for f in folders]
        
        bars = ax5.barh(folders, counts, color=colors, edgecolor='black')
        ax5.set_xlabel('Imágenes Fallidas')
        ax5.set_title('Top Carpetas con Más Fallos', fontweight='bold')
        ax5.invert_yaxis()
    
    # 6. Resumen de problemas
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    problems_text = """
    🔴 PROBLEMAS IDENTIFICADOS:
    
    1. Dataset desbalanceado (5.5:1)
       → Modelo sesgado a predecir NORMAL
    
    2. 36% de imágenes no detectadas
       → Pérdida de datos de entrenamiento
    
    3. Overfitting (100% train accuracy)
       → No generaliza a videos nuevos
    
    4. Solo analiza frames individuales
       → No detecta movimiento de caída
    
    🟢 SOLUCIONES:
    
    • Balancear dataset (ratio 1:1)
    • Bajar threshold a 0.35
    • Reducir complejidad del modelo
    • (Opcional) Usar LSTM temporal
    """
    
    ax6.text(0.1, 0.95, problems_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar
    plot_path = os.path.join(output_folder, "dataset_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Gráfico guardado: {plot_path}")
    
    return plot_path


def save_analysis_report(success_stats, failed_stats, detection_stats, recommendations, output_folder):
    """
    Guarda el reporte de análisis en JSON.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    report = {
        'analysis_date': datetime.now().isoformat(),
        'project': 'SafeGuard Vision AI',
        'successful_images': success_stats,
        'failed_images': failed_stats,
        'detection_rates': detection_stats,
        'recommendations': recommendations,
        'next_steps': [
            'PASO 2: Balancear dataset con script de balanceo',
            'PASO 3: Re-entrenar con datos balanceados',
            'PASO 4: Probar con threshold 0.35',
            'PASO 5: Evaluar si necesita LSTM'
        ]
    }
    
    report_path = os.path.join(output_folder, "dataset_analysis.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"   ✅ Reporte guardado: {report_path}")
    
    return report_path


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                              MAIN
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🔍 SAFEGUARD VISION AI - ANÁLISIS DE DATASET ".center(70) + "║")
    print("║" + " Diagnóstico de Problemas ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # Verificar archivos
    print("\n📁 Verificando archivos...")
    
    if not os.path.exists(CSV_PATH):
        print(f"   ❌ No se encontró: {CSV_PATH}")
        print("   Por favor, ajusta la ruta CSV_PATH en el script.")
        return
    print(f"   ✅ CSV encontrado")
    
    if not os.path.exists(FAILED_PATH):
        print(f"   ⚠️  No se encontró: {FAILED_PATH}")
        print("   Continuando sin análisis de fallidos...")
        failed_stats = None
    else:
        print(f"   ✅ Archivo de fallidos encontrado")
    
    # Analizar
    success_stats = analyze_successful_images(CSV_PATH)
    
    if os.path.exists(FAILED_PATH):
        failed_stats = analyze_failed_images(FAILED_PATH)
    else:
        failed_stats = None
    
    detection_stats = calculate_detection_rates(success_stats, failed_stats)
    
    recommendations = generate_recommendations(success_stats, failed_stats, detection_stats)
    
    # Crear visualizaciones
    create_visualizations(success_stats, failed_stats, detection_stats, OUTPUT_FOLDER)
    
    # Guardar reporte
    save_analysis_report(success_stats, failed_stats, detection_stats, recommendations, OUTPUT_FOLDER)
    
    # Resumen final
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " ✅ ANÁLISIS COMPLETADO ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📂 Resultados guardados en: {OUTPUT_FOLDER}".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print("║  📋 SIGUIENTE PASO:".ljust(71) + "║")
    print("║     Ejecuta el script de balanceo de dataset".ljust(71) + "║")
    print("║     (se generará después de este análisis)".ljust(71) + "║")
    print("╚" + "═"*70 + "╝")
    print()


if __name__ == "__main__":
    main()
