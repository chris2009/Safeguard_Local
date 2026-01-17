"""
🛡️ SafeGuard Vision AI - Benchmark BlazePose
==============================================
Compara los 3 modelos: LITE, FULL, HEAVY

Métricas:
- Tiempo total de procesamiento
- FPS (imágenes por segundo)
- Latencia promedio por imagen
- Tasa de detección (success rate)
- Distribución de tiempos

Autor: Christian Cajusol - MIT Global Teaching Labs
Fecha: Enero 2026
"""

import cv2
import numpy as np
import pandas as pd
import os
import json
import urllib.request
from datetime import datetime
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    ⚠️ CONFIGURA TUS RUTAS AQUÍ ⚠️                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DATASET_LE2I = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\dataset\le2i"
DATASET_URFALL = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\dataset\ur_fall\frames_from_videos"
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_benchmark"

# Número de imágenes para benchmark (None = todas, o un número para prueba rápida)
# Recomendado: 500-1000 para prueba rápida, None para benchmark completo
SAMPLE_SIZE = 500  # Cambiar a 500 para prueba rápida

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURACIÓN DE MODELOS                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MODELS = {
    "LITE": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "path": "pose_landmarker_lite.task",
        "color": "#2ecc71",  # Verde
        "description": "Más rápido, menor precisión"
    },
    "FULL": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "path": "pose_landmarker_full.task",
        "color": "#3498db",  # Azul
        "description": "Balance velocidad/precisión"
    },
    "HEAVY": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "path": "pose_landmarker_heavy.task",
        "color": "#e74c3c",  # Rojo
        "description": "Mayor precisión, más lento"
    }
}

KEYPOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]


def download_models():
    """Descarga todos los modelos si no existen."""
    print("\n📥 Verificando modelos...")
    for name, config in MODELS.items():
        if not os.path.exists(config["path"]):
            print(f"   Descargando {name}...")
            urllib.request.urlretrieve(config["url"], config["path"])
            print(f"   ✅ {name} descargado")
        else:
            print(f"   ✅ {name} encontrado")


def init_detector(model_path):
    """Inicializa un detector de pose."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return vision.PoseLandmarker.create_from_options(options)


def extract_keypoints(image_path, detector):
    """Extrae keypoints y mide el tiempo."""
    import mediapipe as mp
    
    start_time = time.perf_counter()
    
    try:
        image = mp.Image.create_from_file(image_path)
        detection_result = detector.detect(image)
        
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        
        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            return True, elapsed
        else:
            return False, elapsed
    
    except Exception as e:
        elapsed = (time.perf_counter() - start_time) * 1000
        return False, elapsed


def get_all_images(dataset_path, dataset_name):
    """Obtiene lista de imágenes."""
    images = []
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    if not os.path.exists(dataset_path):
        return images
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(extensions):
                images.append(os.path.join(root, file))
    
    return images


def run_benchmark(model_name, model_config, images):
    """Ejecuta benchmark para un modelo."""
    print(f"\n{'='*60}")
    print(f"🔥 Ejecutando benchmark: {model_name}")
    print(f"   {model_config['description']}")
    print(f"{'='*60}")
    
    detector = init_detector(model_config["path"])
    
    latencies = []
    detections = 0
    
    for img_path in tqdm(images, desc=f"{model_name}", unit="img"):
        detected, latency = extract_keypoints(img_path, detector)
        latencies.append(latency)
        if detected:
            detections += 1
    
    # Calcular métricas
    total_time = sum(latencies) / 1000  # segundos
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    fps = len(images) / total_time
    detection_rate = (detections / len(images)) * 100
    
    results = {
        "model": model_name,
        "total_images": len(images),
        "total_time_seconds": round(total_time, 2),
        "total_time_minutes": round(total_time / 60, 2),
        "fps": round(fps, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "std_latency_ms": round(std_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "p99_latency_ms": round(p99_latency, 2),
        "detections": detections,
        "failed": len(images) - detections,
        "detection_rate": round(detection_rate, 2),
        "latencies": latencies  # Para histograma
    }
    
    print(f"\n   ✅ Completado: {fps:.1f} FPS, {detection_rate:.1f}% detección")
    
    return results


def create_comparison_charts(all_results, output_folder):
    """Crea gráficos comparativos."""
    print("\n📊 Generando gráficos comparativos...")
    
    models = [r["model"] for r in all_results]
    colors = [MODELS[m]["color"] for m in models]
    
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('🛡️ SafeGuard Vision AI - Benchmark BlazePose\nMIT Global Teaching Labs', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. FPS Comparison (Bar chart)
    ax1 = fig.add_subplot(2, 3, 1)
    fps_values = [r["fps"] for r in all_results]
    bars1 = ax1.bar(models, fps_values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('FPS (imágenes/segundo)', fontweight='bold')
    ax1.set_title('⚡ Velocidad (FPS)', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, max(fps_values) * 1.2)
    for bar, val in zip(bars1, fps_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Detection Rate (Bar chart)
    ax2 = fig.add_subplot(2, 3, 2)
    detection_rates = [r["detection_rate"] for r in all_results]
    bars2 = ax2.bar(models, detection_rates, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Tasa de Detección (%)', fontweight='bold')
    ax2.set_title('🎯 Tasa de Detección', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 105)
    for bar, val in zip(bars2, detection_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Average Latency (Bar chart)
    ax3 = fig.add_subplot(2, 3, 3)
    avg_latencies = [r["avg_latency_ms"] for r in all_results]
    bars3 = ax3.bar(models, avg_latencies, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Latencia Promedio (ms)', fontweight='bold')
    ax3.set_title('⏱️ Latencia Promedio', fontweight='bold', fontsize=12)
    ax3.set_ylim(0, max(avg_latencies) * 1.3)
    for bar, val in zip(bars3, avg_latencies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Total Time (Bar chart)
    ax4 = fig.add_subplot(2, 3, 4)
    total_times = [r["total_time_minutes"] for r in all_results]
    bars4 = ax4.bar(models, total_times, color=colors, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Tiempo Total (minutos)', fontweight='bold')
    ax4.set_title('🕐 Tiempo Total de Procesamiento', fontweight='bold', fontsize=12)
    ax4.set_ylim(0, max(total_times) * 1.3)
    for bar, val in zip(bars4, total_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{val:.1f} min', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Latency Distribution (Box plot)
    ax5 = fig.add_subplot(2, 3, 5)
    latency_data = [r["latencies"] for r in all_results]
    bp = ax5.boxplot(latency_data, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_ylabel('Latencia (ms)', fontweight='bold')
    ax5.set_title('📈 Distribución de Latencia', fontweight='bold', fontsize=12)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Latency Percentiles (Grouped bar)
    ax6 = fig.add_subplot(2, 3, 6)
    x = np.arange(len(models))
    width = 0.25
    
    p50 = [r["p50_latency_ms"] for r in all_results]
    p95 = [r["p95_latency_ms"] for r in all_results]
    p99 = [r["p99_latency_ms"] for r in all_results]
    
    bars_p50 = ax6.bar(x - width, p50, width, label='P50 (Mediana)', color='#3498db', edgecolor='black')
    bars_p95 = ax6.bar(x, p95, width, label='P95', color='#f39c12', edgecolor='black')
    bars_p99 = ax6.bar(x + width, p99, width, label='P99', color='#e74c3c', edgecolor='black')
    
    ax6.set_ylabel('Latencia (ms)', fontweight='bold')
    ax6.set_title('📊 Percentiles de Latencia', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels(models)
    ax6.legend(loc='upper left')
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar gráfico
    chart_path = os.path.join(output_folder, "benchmark_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Gráfico guardado: {chart_path}")
    
    return chart_path


def create_summary_table(all_results, output_folder):
    """Crea tabla resumen."""
    
    # Crear DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            "Modelo": r["model"],
            "FPS": r["fps"],
            "Tiempo Total (min)": r["total_time_minutes"],
            "Latencia Prom (ms)": r["avg_latency_ms"],
            "Latencia P95 (ms)": r["p95_latency_ms"],
            "Tasa Detección (%)": r["detection_rate"],
            "Detecciones": r["detections"],
            "Fallidas": r["failed"]
        })
    
    df = pd.DataFrame(summary_data)
    
    # Guardar CSV
    csv_path = os.path.join(output_folder, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Crear imagen de tabla
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db']*len(df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Colorear header
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Colorear filas por modelo
    for i, model in enumerate(df["Modelo"]):
        for j in range(len(df.columns)):
            table[(i+1, j)].set_facecolor(MODELS[model]["color"])
            table[(i+1, j)].set_alpha(0.3)
    
    plt.title('🛡️ SafeGuard Vision AI - Benchmark Summary', fontsize=14, fontweight='bold', pad=20)
    
    table_path = os.path.join(output_folder, "benchmark_table.png")
    plt.savefig(table_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Tabla guardada: {table_path}")
    
    return csv_path, table_path


def create_visual_comparison(images, output_folder):
    """Crea comparativa visual de keypoints con los 3 modelos."""
    import mediapipe as mp
    
    print("\n🖼️ Generando comparativa visual de keypoints...")
    
    # Seleccionar 3 imágenes aleatorias
    np.random.seed(123)
    sample_images = list(np.random.choice(images, min(3, len(images)), replace=False))
    
    # Conexiones del esqueleto para dibujar
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Brazos
        (11, 23), (12, 24), (23, 24),  # Torso
        (23, 25), (25, 27), (24, 26), (26, 28),  # Piernas
        (0, 1), (1, 2), (2, 3), (3, 7),  # Cara izq
        (0, 4), (4, 5), (5, 6), (6, 8),  # Cara der
    ]
    
    fig, axes = plt.subplots(len(sample_images), 4, figsize=(20, 5 * len(sample_images)))
    fig.suptitle('🛡️ SafeGuard Vision AI - Comparativa Visual de Keypoints\nLITE vs FULL vs HEAVY', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if len(sample_images) == 1:
        axes = [axes]
    
    for img_idx, img_path in enumerate(sample_images):
        # Cargar imagen original
        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        h, w = img_original.shape[:2]
        
        # Mostrar original
        axes[img_idx][0].imshow(img_original)
        axes[img_idx][0].set_title('Original', fontweight='bold', fontsize=12)
        axes[img_idx][0].axis('off')
        
        # Procesar con cada modelo
        for model_idx, (model_name, model_config) in enumerate(MODELS.items()):
            detector = init_detector(model_config["path"])
            
            try:
                image = mp.Image.create_from_file(img_path)
                detection_result = detector.detect(image)
                
                # Copiar imagen para dibujar
                img_annotated = img_original.copy()
                
                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                    landmarks = detection_result.pose_landmarks[0]
                    
                    # Dibujar conexiones
                    for connection in POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks) and end_idx < len(landmarks):
                            start = landmarks[start_idx]
                            end = landmarks[end_idx]
                            
                            x1, y1 = int(start.x * w), int(start.y * h)
                            x2, y2 = int(end.x * w), int(end.y * h)
                            
                            cv2.line(img_annotated, (x1, y1), (x2, y2), 
                                    (255, 0, 0), 2)
                    
                    # Dibujar keypoints
                    for landmark in landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(img_annotated, (x, y), 4, (0, 255, 0), -1)
                        cv2.circle(img_annotated, (x, y), 5, (0, 0, 0), 1)
                    
                    status = f"✅ {len(landmarks)} keypoints"
                else:
                    status = "❌ No detectado"
                
                axes[img_idx][model_idx + 1].imshow(img_annotated)
                axes[img_idx][model_idx + 1].set_title(
                    f'{model_name}\n{status}', 
                    fontweight='bold', 
                    fontsize=12,
                    color=model_config["color"]
                )
                axes[img_idx][model_idx + 1].axis('off')
                
            except Exception as e:
                axes[img_idx][model_idx + 1].imshow(img_original)
                axes[img_idx][model_idx + 1].set_title(f'{model_name}\n❌ Error', fontweight='bold', fontsize=12)
                axes[img_idx][model_idx + 1].axis('off')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    visual_path = os.path.join(output_folder, "keypoints_visual_comparison.png")
    plt.savefig(visual_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Comparativa visual guardada: {visual_path}")
    
    return visual_path


def create_detailed_report(all_results, output_folder, total_benchmark_time):
    """Crea reporte JSON detallado."""
    
    report = {
        "project": "SafeGuard Vision AI",
        "author": "Christian Cajusol - MIT Global Teaching Labs",
        "benchmark_date": datetime.now().isoformat(),
        "total_benchmark_time_minutes": round(total_benchmark_time / 60, 2),
        "sample_size": all_results[0]["total_images"],
        "models_compared": list(MODELS.keys()),
        "results": {}
    }
    
    for r in all_results:
        model_results = {k: v for k, v in r.items() if k != "latencies"}
        report["results"][r["model"]] = model_results
    
    # Añadir comparativa
    fps_values = {r["model"]: r["fps"] for r in all_results}
    detection_values = {r["model"]: r["detection_rate"] for r in all_results}
    
    fastest = max(fps_values, key=fps_values.get)
    most_accurate = max(detection_values, key=detection_values.get)
    
    report["comparison"] = {
        "fastest_model": fastest,
        "fastest_fps": fps_values[fastest],
        "best_detection_model": most_accurate,
        "best_detection_rate": detection_values[most_accurate],
        "speed_improvement_lite_vs_heavy": round(fps_values.get("LITE", 0) / fps_values.get("HEAVY", 1), 2),
        "recommendation": {
            "edge_devices": "LITE - Mayor velocidad para tiempo real",
            "general_use": "FULL - Balance entre velocidad y precisión",
            "research": "HEAVY - Máxima precisión para análisis detallado"
        }
    }
    
    report_path = os.path.join(output_folder, "benchmark_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Reporte guardado: {report_path}")
    
    return report_path


def print_final_summary(all_results):
    """Imprime resumen final en consola."""
    
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🏆 RESULTADOS DEL BENCHMARK ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print("║" + " Modelo      │    FPS    │  Latencia  │ Detección │   Tiempo   " + "║")
    print("║" + "─"*12 + "┼" + "─"*11 + "┼" + "─"*12 + "┼" + "─"*11 + "┼" + "─"*12 + "║")
    
    for r in all_results:
        line = f" {r['model']:<10} │ {r['fps']:>8.1f}  │ {r['avg_latency_ms']:>8.1f}ms │ {r['detection_rate']:>8.1f}% │ {r['total_time_minutes']:>8.1f}min "
        print("║" + line + "║")
    
    print("╠" + "═"*70 + "╣")
    
    # Determinar ganadores
    fps_values = {r["model"]: r["fps"] for r in all_results}
    detection_values = {r["model"]: r["detection_rate"] for r in all_results}
    
    fastest = max(fps_values, key=fps_values.get)
    most_accurate = max(detection_values, key=detection_values.get)
    
    print(f"║  ⚡ Más rápido:      {fastest:<10} ({fps_values[fastest]:.1f} FPS){' '*24}║")
    print(f"║  🎯 Mejor detección: {most_accurate:<10} ({detection_values[most_accurate]:.1f}%){' '*23}║")
    print("╠" + "═"*70 + "╣")
    print("║  📋 RECOMENDACIONES:                                                 ║")
    print("║     • Edge/Tiempo real → LITE                                        ║")
    print("║     • Uso general      → FULL                                        ║")
    print("║     • Investigación    → HEAVY                                       ║")
    print("╚" + "═"*70 + "╝")


def main():
    import mediapipe as mp
    
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SafeGuard Vision AI - Benchmark BlazePose ".center(70) + "║")
    print("║" + " Comparativa: LITE vs FULL vs HEAVY ".center(70) + "║")
    print("║" + " MIT Global Teaching Labs ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # Verificar rutas
    print("\n📁 Verificando datasets...")
    if not os.path.exists(DATASET_LE2I):
        print(f"   ❌ No existe: {DATASET_LE2I}")
        return
    if not os.path.exists(DATASET_URFALL):
        print(f"   ❌ No existe: {DATASET_URFALL}")
        return
    print(f"   ✅ Datasets encontrados")
    
    # Descargar modelos
    download_models()
    
    # Obtener imágenes
    print("\n📂 Cargando imágenes...")
    images_le2i = get_all_images(DATASET_LE2I, 'le2i')
    images_urfall = get_all_images(DATASET_URFALL, 'ur_fall')
    all_images = images_le2i + images_urfall
    
    # Aplicar sample si está configurado
    if SAMPLE_SIZE and SAMPLE_SIZE < len(all_images):
        np.random.seed(42)  # Reproducibilidad
        all_images = list(np.random.choice(all_images, SAMPLE_SIZE, replace=False))
        print(f"   📊 Usando muestra de {SAMPLE_SIZE} imágenes (de {len(images_le2i) + len(images_urfall)} totales)")
    else:
        print(f"   📊 Total: {len(all_images)} imágenes")
    
    # Crear carpeta de salida
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Ejecutar benchmarks
    all_results = []
    benchmark_start = time.time()
    
    for model_name, model_config in MODELS.items():
        results = run_benchmark(model_name, model_config, all_images)
        all_results.append(results)
    
    total_benchmark_time = time.time() - benchmark_start
    
    # Generar reportes
    print("\n" + "="*60)
    print("📊 GENERANDO REPORTES")
    print("="*60)
    
    create_comparison_charts(all_results, OUTPUT_FOLDER)
    create_summary_table(all_results, OUTPUT_FOLDER)
    create_visual_comparison(all_images, OUTPUT_FOLDER)
    create_detailed_report(all_results, OUTPUT_FOLDER, total_benchmark_time)
    
    # Resumen final
    print_final_summary(all_results)
    
    print(f"\n📂 Todos los archivos guardados en: {OUTPUT_FOLDER}")
    print(f"⏱️  Tiempo total del benchmark: {total_benchmark_time/60:.1f} minutos\n")


if __name__ == "__main__":
    main()
