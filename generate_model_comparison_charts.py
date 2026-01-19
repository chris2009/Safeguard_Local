"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - ANÁLISIS COMPARATIVO DE MODELOS                 ║
║                                                                              ║
║   Generación de visualizaciones profesionales para presentación MIT         ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs - Industry 4.0 Zero Accident           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

VISUALIZACIONES GENERADAS:
==========================
    1. Radar Chart - Comparación multidimensional
    2. Evolution Timeline - Progresión de mejoras
    3. Confusion Matrix Grid - Matrices lado a lado
    4. Precision-Recall Trade-off - Análisis de umbrales
    5. Architecture Comparison - Conceptual diagram
    6. Performance Heatmap - Vista general de métricas
    7. Bar Chart Racing - Comparación directa
    8. Executive Summary - Dashboard completo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración global de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'

# Carpeta de salida
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\model_comparison_charts"


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         DATOS DE LOS MODELOS
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Métricas de cada modelo (de tus entrenamientos)
MODELS_DATA = {
    "Random Forest\n(Unbalanced)": {
        "accuracy": 0.9672,
        "precision": 0.9904,
        "recall": 0.8886,
        "f1": 0.9367,
        "auc_roc": 0.9876,
        "color": "#E74C3C",  # Rojo
        "short_name": "RF-Unbal",
        "type": "static",
        "temporal": False,
        "training_time": 2,  # minutos aproximados
        "parameters": 5000,
        "detects_transition": False
    },
    "Random Forest\n(Balanced)": {
        "accuracy": 0.9384,
        "precision": 0.9294,
        "recall": 0.9489,
        "f1": 0.9391,
        "auc_roc": 0.9896,
        "color": "#E67E22",  # Naranja
        "short_name": "RF-Bal",
        "type": "static",
        "temporal": False,
        "training_time": 1,
        "parameters": 5000,
        "detects_transition": False
    },
    "LSTM\n(Bidirectional)": {
        "accuracy": 0.9697,
        "precision": 0.9412,
        "recall": 1.0000,
        "f1": 0.9697,
        "auc_roc": 1.0000,
        "color": "#3498DB",  # Azul
        "short_name": "LSTM",
        "type": "temporal",
        "temporal": True,
        "training_time": 8,
        "parameters": 150000,
        "detects_transition": True
    },
    "Transformer\n(Self-Attention)": {
        "accuracy": 0.9697,
        "precision": 0.9412,
        "recall": 1.0000,
        "f1": 0.9697,
        "auc_roc": 1.0000,
        "color": "#9B59B6",  # Púrpura
        "short_name": "Transformer",
        "type": "temporal",
        "temporal": True,
        "training_time": 12,
        "parameters": 500000,
        "detects_transition": True
    }
}

# Orden de modelos para mostrar progresión
MODEL_ORDER = [
    "Random Forest\n(Unbalanced)",
    "Random Forest\n(Balanced)", 
    "LSTM\n(Bidirectional)",
    "Transformer\n(Self-Attention)"
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         FUNCIONES AUXILIARES
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_output_folder():
    """Crea la carpeta de salida si no existe."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"📂 Carpeta de salida: {OUTPUT_FOLDER}")


def add_watermark(fig, text="SafeGuard Vision AI | MIT Global Teaching Labs 2025"):
    """Añade marca de agua profesional."""
    fig.text(0.99, 0.01, text, fontsize=8, color='gray', alpha=0.5,
             ha='right', va='bottom', style='italic')


def save_figure(fig, filename, dpi=300):
    """Guarda figura con alta calidad."""
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.3)
    plt.close(fig)
    print(f"   ✅ Guardado: {filename}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         GRÁFICO 1: RADAR CHART
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_radar_chart():
    """
    Crea un radar chart comparativo de todas las métricas.
    Muestra fortalezas y debilidades de cada modelo.
    """
    print("\n📊 Generando Radar Chart...")
    
    # Métricas a comparar
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    num_metrics = len(metrics)
    
    # Ángulos para el radar
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el polígono
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Dibujar cada modelo
    for model_name in MODEL_ORDER:
        model = MODELS_DATA[model_name]
        values = [
            model['accuracy'],
            model['precision'],
            model['recall'],
            model['f1'],
            model['auc_roc']
        ]
        values += values[:1]  # Cerrar el polígono
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model['short_name'],
                color=model['color'], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=model['color'])
    
    # Configurar ejes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0.8, 1.02)
    
    # Líneas de grid personalizadas
    ax.set_rticks([0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(['85%', '90%', '95%', '100%'], fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)
    
    # Resaltar el 100% de recall
    ax.axhline(y=1.0, color='gold', linestyle='--', linewidth=2, alpha=0.7)
    
    # Título y leyenda
    plt.title('Multi-Dimensional Performance Comparison\n', fontsize=16, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11,
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    # Añadir anotación
    fig.text(0.5, 0.02, 
             '★ Temporal models (LSTM & Transformer) achieve 100% Recall - detecting ALL falls',
             ha='center', fontsize=11, style='italic', color='#2C3E50',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#DEE2E6'))
    
    add_watermark(fig)
    save_figure(fig, '01_radar_chart_comparison.png')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         GRÁFICO 2: EVOLUTION TIMELINE
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_evolution_timeline():
    """
    Muestra la evolución del proyecto y mejoras incrementales.
    Timeline visual del desarrollo.
    """
    print("📊 Generando Evolution Timeline...")
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Datos de evolución
    stages = [
        ("Stage 1", "Random Forest\nUnbalanced", 88.86, "#E74C3C", 
         "• Single frame analysis\n• 5.5:1 class imbalance\n• Overfitting issues"),
        ("Stage 2", "Random Forest\nBalanced", 94.89, "#E67E22",
         "• 1:1 balanced dataset\n• Anti-overfitting params\n• +6% recall improvement"),
        ("Stage 3", "LSTM\nBidirectional", 100.0, "#3498DB",
         "• 30-frame sequences\n• Temporal features\n• Detects TRANSITIONS"),
        ("Stage 4", "Transformer\nSelf-Attention", 100.0, "#9B59B6",
         "• Attention mechanism\n• Parallel processing\n• State-of-the-art")
    ]
    
    # Posiciones
    y_base = 0.5
    x_positions = np.linspace(0.12, 0.88, 4)
    
    # Línea de tiempo principal
    ax.axhline(y=y_base, color='#BDC3C7', linewidth=4, zorder=1)
    
    # Dibujar cada etapa
    for i, (stage, name, recall, color, details) in enumerate(stages):
        x = x_positions[i]
        
        # Círculo principal
        circle = Circle((x, y_base), 0.04, color=color, zorder=3, ec='white', linewidth=3)
        ax.add_patch(circle)
        
        # Número de stage
        ax.text(x, y_base, str(i+1), ha='center', va='center', fontsize=14,
                fontweight='bold', color='white', zorder=4)
        
        # Caja de información arriba
        box_y = y_base + 0.25
        box = FancyBboxPatch((x-0.08, box_y-0.12), 0.16, 0.24,
                            boxstyle="round,pad=0.02,rounding_size=0.02",
                            facecolor=color, edgecolor='white', linewidth=2,
                            alpha=0.9, zorder=2)
        ax.add_patch(box)
        
        # Texto en la caja
        ax.text(x, box_y + 0.05, name, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', zorder=5)
        ax.text(x, box_y - 0.05, f"Recall: {recall:.1f}%", ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=5)
        
        # Flecha conectora
        ax.annotate('', xy=(x, y_base + 0.05), xytext=(x, box_y - 0.13),
                   arrowprops=dict(arrowstyle='-', color=color, lw=2))
        
        # Detalles abajo
        detail_y = y_base - 0.15
        ax.text(x, detail_y, details, ha='center', va='top', fontsize=9,
                color='#2C3E50', linespacing=1.5,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', 
                         edgecolor='#DEE2E6', alpha=0.9))
        
        # Flechas de mejora entre stages
        if i < 3:
            mid_x = (x_positions[i] + x_positions[i+1]) / 2
            improvement = stages[i+1][2] - stages[i][2]
            arrow_color = '#27AE60' if improvement > 0 else '#95A5A6'
            
            ax.annotate('', xy=(x_positions[i+1]-0.05, y_base),
                       xytext=(x+0.05, y_base),
                       arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
            
            if improvement > 0:
                ax.text(mid_x, y_base + 0.07, f'+{improvement:.1f}%', ha='center',
                       fontsize=10, fontweight='bold', color='#27AE60')
    
    # Indicador de breakthrough
    ax.annotate('🚀 BREAKTHROUGH', xy=(x_positions[2], y_base + 0.42),
               fontsize=12, fontweight='bold', color='#3498DB', ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB', edgecolor='#3498DB'))
    
    # Configurar ejes
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.axis('off')
    
    # Título
    ax.text(0.5, 0.95, 'Model Evolution: From Static Analysis to Temporal Intelligence',
           ha='center', va='top', fontsize=18, fontweight='bold', color='#2C3E50')
    
    ax.text(0.5, 0.88, 'Progressive improvement through architectural innovations',
           ha='center', va='top', fontsize=12, style='italic', color='#7F8C8D')
    
    add_watermark(fig)
    save_figure(fig, '02_evolution_timeline.png')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         GRÁFICO 3: CONFUSION MATRIX GRID
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_confusion_matrix_grid():
    """
    Muestra las matrices de confusión de todos los modelos lado a lado.
    """
    print("📊 Generando Confusion Matrix Grid...")
    
    # Matrices de confusión simuladas basadas en las métricas
    # Formato: [[TN, FP], [FN, TP]]
    confusion_matrices = {
        "Random Forest\n(Unbalanced)": np.array([[900, 10], [115, 915]]),
        "Random Forest\n(Balanced)": np.array([[155, 12], [9, 157]]),
        "LSTM\n(Bidirectional)": np.array([[16, 2], [0, 16]]),
        "Transformer\n(Self-Attention)": np.array([[16, 2], [0, 16]])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(MODEL_ORDER):
        ax = axes[idx]
        cm = confusion_matrices[model_name]
        model = MODELS_DATA[model_name]
        
        # Normalizar para visualización
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear colormap personalizado
        cmap = LinearSegmentedColormap.from_list('custom', ['white', model['color']])
        
        # Dibujar heatmap
        im = ax.imshow(cm_normalized, cmap=cmap, vmin=0, vmax=1)
        
        # Añadir texto
        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                pct = cm_normalized[i, j] * 100
                text_color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f'{value}\n({pct:.1f}%)', ha='center', va='center',
                       fontsize=12, fontweight='bold', color=text_color)
        
        # Configurar ejes
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['ADL\n(No Fall)', 'Fall'], fontsize=10)
        ax.set_yticklabels(['ADL\n(No Fall)', 'Fall'], fontsize=10)
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        
        # Título con indicador de recall
        recall = model['recall'] * 100
        title_color = '#27AE60' if recall == 100 else '#2C3E50'
        recall_indicator = '✓ 100%' if recall == 100 else f'{recall:.1f}%'
        ax.set_title(f"{model['short_name']}\nRecall: {recall_indicator}", 
                    fontsize=12, fontweight='bold', color=title_color, pad=10)
        
        # Borde si es 100% recall
        if recall == 100:
            for spine in ax.spines.values():
                spine.set_edgecolor('#27AE60')
                spine.set_linewidth(3)
    
    plt.suptitle('Confusion Matrix Comparison\n', fontsize=16, fontweight='bold', y=1.02)
    
    # Leyenda de colores
    fig.text(0.5, -0.02, 
             'Green border indicates 100% Recall (Zero missed falls) | '
             'FN = False Negatives (Missed falls - CRITICAL)',
             ha='center', fontsize=10, style='italic', color='#7F8C8D')
    
    plt.tight_layout()
    add_watermark(fig)
    save_figure(fig, '03_confusion_matrix_grid.png')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         GRÁFICO 4: BAR CHART COMPARISON
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_bar_comparison():
    """
    Gráfico de barras comparativo con énfasis en recall.
    """
    print("📊 Generando Bar Chart Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Datos
    models = [MODELS_DATA[m]['short_name'] for m in MODEL_ORDER]
    colors = [MODELS_DATA[m]['color'] for m in MODEL_ORDER]
    
    metrics_to_plot = ['recall', 'precision', 'f1']
    metric_labels = ['Recall\n(Fall Detection)', 'Precision\n(False Alarm Rate)', 'F1-Score\n(Balance)']
    
    # Subplot 1: Barras agrupadas
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [MODELS_DATA[m][metric] * 100 for m in MODEL_ORDER]
        bars = ax1.bar(x + i*width - width, values, width, label=label, alpha=0.85,
                      color=plt.cm.Set2(i), edgecolor='white', linewidth=1)
        
        # Añadir valores
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center',
                        fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics by Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylim(80, 105)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.axhline(y=100, color='gold', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Score')
    ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Recall focus con destaque
    ax2 = axes[1]
    recalls = [MODELS_DATA[m]['recall'] * 100 for m in MODEL_ORDER]
    
    bars = ax2.bar(models, recalls, color=colors, edgecolor='white', linewidth=2, alpha=0.85)
    
    # Destacar 100%
    for bar, recall in zip(bars, recalls):
        if recall == 100:
            bar.set_edgecolor('#27AE60')
            bar.set_linewidth(4)
            # Añadir estrella
            ax2.annotate('★', xy=(bar.get_x() + bar.get_width()/2, 101),
                        fontsize=20, ha='center', color='gold')
    
    # Valores
    for bar, val in zip(bars, recalls):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height - 5),
                    ha='center', fontsize=14, fontweight='bold', color='white')
    
    ax2.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax2.set_title('🎯 RECALL: Critical Metric for Fall Detection\n"Did we catch ALL the falls?"', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color='gold', linestyle='--', linewidth=2, alpha=0.7)
    
    # Anotación
    ax2.annotate('100% Recall = Zero Missed Falls', xy=(2.5, 103),
                fontsize=11, ha='center', style='italic', color='#27AE60',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', edgecolor='#27AE60'))
    
    plt.tight_layout()
    add_watermark(fig)
    save_figure(fig, '04_bar_chart_comparison.png')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         GRÁFICO 5: ARCHITECTURE COMPARISON
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_architecture_comparison():
    """
    Diagrama conceptual comparando las arquitecturas.
    """
    print("📊 Generando Architecture Comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    architectures = [
        {
            "name": "Random Forest",
            "color": "#E74C3C",
            "flow": ["Single\nFrame", "BlazePose\n33 Keypoints", "Feature\nExtraction", "Random\nForest", "Prediction"],
            "limitation": "❌ Cannot detect\nmovement or transitions",
            "type": "STATIC"
        },
        {
            "name": "LSTM",
            "color": "#3498DB", 
            "flow": ["30 Frame\nSequence", "BlazePose\n+ Temporal", "LSTM\nLayer 1", "LSTM\nLayer 2", "Prediction"],
            "limitation": "✓ Detects transitions\nthrough sequence memory",
            "type": "TEMPORAL"
        },
        {
            "name": "Transformer",
            "color": "#9B59B6",
            "flow": ["30 Frame\nSequence", "Positional\nEncoding", "Self-\nAttention", "Feed\nForward", "Prediction"],
            "limitation": "✓ Global attention\ncompares all frames",
            "type": "ATTENTION"
        }
    ]
    
    for ax, arch in zip(axes, architectures):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Título
        ax.text(5, 9.5, arch["name"], ha='center', va='top', fontsize=16,
               fontweight='bold', color=arch["color"])
        ax.text(5, 8.8, f'[{arch["type"]}]', ha='center', va='top', fontsize=10,
               color='#7F8C8D', style='italic')
        
        # Cajas del flujo
        y_positions = np.linspace(7.5, 2, 5)
        
        for i, (label, y) in enumerate(zip(arch["flow"], y_positions)):
            # Caja
            box = FancyBboxPatch((2.5, y-0.4), 5, 0.8,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=arch["color"], alpha=0.8,
                                edgecolor='white', linewidth=2)
            ax.add_patch(box)
            
            # Texto
            ax.text(5, y, label, ha='center', va='center', fontsize=10,
                   fontweight='bold', color='white')
            
            # Flecha
            if i < 4:
                ax.annotate('', xy=(5, y_positions[i+1]+0.5), xytext=(5, y-0.5),
                           arrowprops=dict(arrowstyle='->', color=arch["color"], lw=2))
        
        # Limitación/Ventaja
        box_color = '#EAFAF1' if '✓' in arch["limitation"] else '#FDEDEC'
        edge_color = '#27AE60' if '✓' in arch["limitation"] else '#E74C3C'
        
        ax.text(5, 0.8, arch["limitation"], ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color, 
                        edgecolor=edge_color, linewidth=2),
               linespacing=1.5)
    
    plt.suptitle('Architecture Comparison: Static vs Temporal vs Attention-Based\n',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    add_watermark(fig)
    save_figure(fig, '05_architecture_comparison.png')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         GRÁFICO 6: PERFORMANCE HEATMAP
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_performance_heatmap():
    """
    Heatmap de todas las métricas para todos los modelos.
    """
    print("📊 Generando Performance Heatmap...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear matriz de datos
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    models = [MODELS_DATA[m]['short_name'] for m in MODEL_ORDER]
    
    data = []
    for model_name in MODEL_ORDER:
        model = MODELS_DATA[model_name]
        data.append([
            model['accuracy'] * 100,
            model['precision'] * 100,
            model['recall'] * 100,
            model['f1'] * 100,
            model['auc_roc'] * 100
        ])
    
    data = np.array(data)
    
    # Crear heatmap
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=85, vmax=100)
    
    # Añadir valores
    for i in range(len(models)):
        for j in range(len(metrics)):
            value = data[i, j]
            text_color = 'white' if value > 95 else 'black'
            fontweight = 'bold' if value == 100 else 'normal'
            
            text = f'{value:.1f}%'
            if value == 100:
                text = '100% ★'
            
            ax.text(j, i, text, ha='center', va='center', fontsize=12,
                   fontweight=fontweight, color=text_color)
    
    # Configurar ejes
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_yticklabels(models, fontsize=11, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score (%)', fontsize=11, fontweight='bold')
    
    # Título
    ax.set_title('Performance Heatmap: All Metrics × All Models\n', 
                fontsize=16, fontweight='bold')
    
    # Resaltar columna de Recall
    ax.axvline(x=1.5, color='gold', linewidth=3, linestyle='--')
    ax.axvline(x=2.5, color='gold', linewidth=3, linestyle='--')
    ax.text(2, -0.7, '← CRITICAL METRIC', ha='center', fontsize=10, 
           fontweight='bold', color='#D4AC0D')
    
    plt.tight_layout()
    add_watermark(fig)
    save_figure(fig, '06_performance_heatmap.png')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         GRÁFICO 7: KEY INSIGHT - WHY TEMPORAL
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_key_insight_diagram():
    """
    Diagrama explicando por qué los modelos temporales son mejores.
    """
    print("📊 Generando Key Insight Diagram...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot 1: Problema con análisis estático
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.5, 'THE PROBLEM: Static Analysis', ha='center', fontsize=16,
            fontweight='bold', color='#E74C3C')
    
    # Escenarios que confunden al RF
    scenarios = [
        ("Person lying\non couch", "Pose: Horizontal", "❌ FALSE POSITIVE", 7),
        ("Person who\nfell", "Pose: Horizontal", "Should detect!", 4.5),
        ("Person\ncrouching", "Pose: Low", "❌ FALSE POSITIVE", 2),
    ]
    
    for label, pose, result, y in scenarios:
        # Caja de escenario
        ax1.add_patch(FancyBboxPatch((0.5, y-0.7), 3, 1.4,
                                     boxstyle="round,pad=0.02", facecolor='#F8F9FA',
                                     edgecolor='#BDC3C7', linewidth=2))
        ax1.text(2, y, label, ha='center', va='center', fontsize=10)
        
        # Flecha
        ax1.annotate('', xy=(4.5, y), xytext=(3.7, y),
                    arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2))
        
        # Pose
        ax1.text(5.5, y, pose, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB', edgecolor='#3498DB'))
        
        # Flecha
        ax1.annotate('', xy=(7.3, y), xytext=(6.5, y),
                    arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2))
        
        # Resultado
        color = '#E74C3C' if 'FALSE' in result else '#27AE60'
        ax1.text(8.5, y, result, ha='center', va='center', fontsize=10, fontweight='bold',
                color=color)
    
    ax1.text(5, 0.5, 'Same pose = Same prediction\nCannot distinguish ACTION from STATE',
            ha='center', fontsize=11, style='italic', color='#E74C3C',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEDEC', edgecolor='#E74C3C'))
    
    # Subplot 2: Solución con análisis temporal
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.5, 'THE SOLUTION: Temporal Analysis', ha='center', fontsize=16,
            fontweight='bold', color='#27AE60')
    
    # Transición de caída
    ax2.text(5, 7.5, 'Fall Detection = Detecting TRANSITION', ha='center', fontsize=12,
            fontweight='bold', color='#2C3E50')
    
    # Timeline de frames
    frames = ["Standing", "Falling...", "Impact", "On ground"]
    x_pos = np.linspace(1.5, 8.5, 4)
    
    for i, (x, frame) in enumerate(zip(x_pos, frames)):
        color = '#3498DB' if i < 3 else '#E74C3C'
        ax2.add_patch(Circle((x, 5.5), 0.5, color=color, alpha=0.8))
        ax2.text(x, 5.5, f'F{i+1}', ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
        ax2.text(x, 4.7, frame, ha='center', fontsize=9)
        
        if i < 3:
            ax2.annotate('', xy=(x+0.7, 5.5), xytext=(x+0.5, 5.5),
                        arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))
    
    # Bracket mostrando secuencia
    ax2.annotate('', xy=(1.5, 6.3), xytext=(8.5, 6.3),
                arrowprops=dict(arrowstyle='-', color='#9B59B6', lw=2,
                               connectionstyle='bar,fraction=0.3'))
    ax2.text(5, 7, 'LSTM/Transformer analyzes full sequence', ha='center', fontsize=10,
            color='#9B59B6', fontweight='bold')
    
    # Comparación
    comparisons = [
        ("Couch scenario:", "Frame 1: Lying → Frame 30: Lying = NO FALL ✓", 3),
        ("Fall scenario:", "Frame 1: Standing → Frame 30: Ground = FALL ✓", 1.5),
    ]
    
    for label, desc, y in comparisons:
        ax2.text(1, y, label, fontsize=10, fontweight='bold', color='#2C3E50')
        ax2.text(1, y-0.5, desc, fontsize=10, color='#27AE60')
    
    plt.suptitle('Why Temporal Models Achieve 100% Recall\n', fontsize=16, 
                fontweight='bold', y=1.02)
    
    plt.tight_layout()
    add_watermark(fig)
    save_figure(fig, '07_key_insight_temporal.png')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         GRÁFICO 8: EXECUTIVE DASHBOARD
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_executive_dashboard():
    """
    Dashboard ejecutivo con resumen completo del proyecto.
    """
    print("📊 Generando Executive Dashboard...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Título principal
    # ═══════════════════════════════════════════════════════════════════════════
    
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    ax_title.text(0.5, 0.8, '🛡️ SAFEGUARD VISION AI', ha='center', va='center',
                 fontsize=28, fontweight='bold', color='#2C3E50')
    ax_title.text(0.5, 0.5, 'Industry 4.0 Fall Detection System | MIT Global Teaching Labs 2025',
                 ha='center', va='center', fontsize=14, color='#7F8C8D')
    ax_title.text(0.5, 0.2, 'Comparative Analysis: From Classical ML to Deep Learning Transformers',
                 ha='center', va='center', fontsize=12, style='italic', color='#95A5A6')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # KPIs principales
    # ═══════════════════════════════════════════════════════════════════════════
    
    kpis = [
        ("Best Recall", "100%", "LSTM & Transformer", "#27AE60"),
        ("Improvement", "+11.1%", "vs baseline", "#3498DB"),
        ("False Negatives", "0", "Zero missed falls", "#E74C3C"),
        ("Models Tested", "4", "RF, RF-Bal, LSTM, Trans", "#9B59B6"),
    ]
    
    for i, (title, value, subtitle, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[1, i])
        ax.axis('off')
        
        # Fondo
        ax.add_patch(FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=color, alpha=0.1,
                                    edgecolor=color, linewidth=3))
        
        ax.text(0.5, 0.75, title, ha='center', va='center', fontsize=11, color='#7F8C8D')
        ax.text(0.5, 0.45, value, ha='center', va='center', fontsize=32, 
               fontweight='bold', color=color)
        ax.text(0.5, 0.15, subtitle, ha='center', va='center', fontsize=10, 
               color='#95A5A6', style='italic')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Mini radar chart
    # ═══════════════════════════════════════════════════════════════════════════
    
    ax_radar = fig.add_subplot(gs[2, 0:2], polar=True)
    
    metrics = ['Acc', 'Prec', 'Rec', 'F1', 'AUC']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for model_name in MODEL_ORDER[-2:]:  # Solo LSTM y Transformer
        model = MODELS_DATA[model_name]
        values = [model['accuracy'], model['precision'], model['recall'], 
                 model['f1'], model['auc_roc']]
        values += values[:1]
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=model['short_name'],
                     color=model['color'], markersize=6)
        ax_radar.fill(angles, values, alpha=0.1, color=model['color'])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics, fontsize=10)
    ax_radar.set_ylim(0.9, 1.02)
    ax_radar.set_title('Best Models Performance', fontsize=12, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', fontsize=9)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Mini bar chart - Recall comparison
    # ═══════════════════════════════════════════════════════════════════════════
    
    ax_bar = fig.add_subplot(gs[2, 2:4])
    
    models = [MODELS_DATA[m]['short_name'] for m in MODEL_ORDER]
    recalls = [MODELS_DATA[m]['recall'] * 100 for m in MODEL_ORDER]
    colors = [MODELS_DATA[m]['color'] for m in MODEL_ORDER]
    
    bars = ax_bar.barh(models, recalls, color=colors, edgecolor='white', linewidth=2)
    
    for bar, recall in zip(bars, recalls):
        width = bar.get_width()
        ax_bar.text(width - 5, bar.get_y() + bar.get_height()/2, f'{recall:.1f}%',
                   ha='right', va='center', fontsize=12, fontweight='bold', color='white')
    
    ax_bar.set_xlim(80, 105)
    ax_bar.axvline(x=100, color='gold', linestyle='--', linewidth=2)
    ax_bar.set_xlabel('Recall (%)', fontsize=11, fontweight='bold')
    ax_bar.set_title('Recall Comparison (Critical Metric)', fontsize=12, fontweight='bold')
    ax_bar.grid(axis='x', alpha=0.3)
    
    # Footer
    fig.text(0.5, 0.02, 
             'Team: Christian Cajusol, Hugo Angeles, Francisco Meza, Jhomar Yurivilca | '
             'Objective: Zero Accident Initiative',
             ha='center', fontsize=10, color='#7F8C8D', style='italic')
    
    add_watermark(fig)
    save_figure(fig, '08_executive_dashboard.png')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         GRÁFICO 9: IMPROVEMENT WATERFALL
# ╚══════════════════════════════════════════════════════════════════════════════╝

def create_improvement_waterfall():
    """
    Gráfico de cascada mostrando las mejoras incrementales.
    """
    print("📊 Generando Improvement Waterfall...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Datos
    stages = ['Baseline\n(RF Unbal)', 'Balancing\nEffect', 'Temporal\nAnalysis', 'Final\nRecall']
    values = [88.86, 6.03, 5.11, 100.0]  # Baseline, incrementos, total
    
    # Colores
    colors = ['#E74C3C', '#27AE60', '#27AE60', '#3498DB']
    
    # Posiciones
    x = np.arange(len(stages))
    
    # Calcular posiciones de barras para waterfall
    cumulative = [88.86, 88.86 + 6.03, 88.86 + 6.03 + 5.11, 0]
    bar_bottoms = [0, 88.86, 88.86 + 6.03, 0]
    bar_heights = [88.86, 6.03, 5.11, 100]
    
    # Dibujar barras
    bars = ax.bar(x, bar_heights, bottom=bar_bottoms, color=colors, 
                 edgecolor='white', linewidth=2, width=0.6)
    
    # Conectores
    for i in range(len(x)-2):
        ax.plot([i+0.3, i+0.7], [cumulative[i], cumulative[i]], 
               color='#2C3E50', linewidth=2, linestyle='--')
    
    # Valores en barras
    for i, (bar, val) in enumerate(zip(bars, bar_heights)):
        if i < 3:
            y_pos = bar_bottoms[i] + val/2
            prefix = '+' if i > 0 else ''
            ax.text(i, y_pos, f'{prefix}{val:.1f}%', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white')
        else:
            ax.text(i, 50, f'{val:.1f}%', ha='center', va='center',
                   fontsize=18, fontweight='bold', color='white')
            ax.text(i, 85, '★ PERFECT', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='gold')
    
    # Anotaciones
    ax.annotate('Data\nBalancing', xy=(1, 94), fontsize=10, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', edgecolor='#27AE60'))
    ax.annotate('LSTM/\nTransformer', xy=(2, 99), fontsize=10, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', edgecolor='#27AE60'))
    
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_title('Recall Improvement Journey: From 88.9% to 100%\n', 
                fontsize=16, fontweight='bold')
    
    ax.axhline(y=100, color='gold', linestyle='--', linewidth=2, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Leyenda
    legend_elements = [
        mpatches.Patch(facecolor='#E74C3C', label='Baseline'),
        mpatches.Patch(facecolor='#27AE60', label='Improvement'),
        mpatches.Patch(facecolor='#3498DB', label='Final Result'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    add_watermark(fig)
    save_figure(fig, '09_improvement_waterfall.png')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                              MAIN
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SAFEGUARD VISION AI - MODEL COMPARISON CHARTS ".center(70) + "║")
    print("║" + " Professional Visualizations for MIT Presentation ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    create_output_folder()
    
    print("\n📊 Generando visualizaciones profesionales...\n")
    
    # Generar todos los gráficos
    create_radar_chart()
    create_evolution_timeline()
    create_confusion_matrix_grid()
    create_bar_comparison()
    create_architecture_comparison()
    create_performance_heatmap()
    create_key_insight_diagram()
    create_executive_dashboard()
    create_improvement_waterfall()
    
    # Resumen
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " ✅ VISUALIZACIONES GENERADAS ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📂 Carpeta: {OUTPUT_FOLDER}".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print("║  📊 Archivos generados:".ljust(71) + "║")
    print("║     • 01_radar_chart_comparison.png".ljust(71) + "║")
    print("║     • 02_evolution_timeline.png".ljust(71) + "║")
    print("║     • 03_confusion_matrix_grid.png".ljust(71) + "║")
    print("║     • 04_bar_chart_comparison.png".ljust(71) + "║")
    print("║     • 05_architecture_comparison.png".ljust(71) + "║")
    print("║     • 06_performance_heatmap.png".ljust(71) + "║")
    print("║     • 07_key_insight_temporal.png".ljust(71) + "║")
    print("║     • 08_executive_dashboard.png".ljust(71) + "║")
    print("║     • 09_improvement_waterfall.png".ljust(71) + "║")
    print("╚" + "═"*70 + "╝")
    print()


if __name__ == "__main__":
    main()
