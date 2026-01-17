"""
🛡️ SafeGuard Vision AI - Extracción de Keypoints con BlazePose
================================================================
Script para ejecutar localmente en tu laptop.

Autor: Christian Cajusol - MIT Global Teaching Labs
Fecha: Enero 2026

REQUISITOS:
    pip install mediapipe opencv-python pandas tqdm

HARDWARE RECOMENDADO:
    - GPU: RTX 4070 (o superior)
    - CPU: i9-14gen
    - RAM: 32 GB

TIEMPO ESTIMADO: ~10-15 minutos para ~17,000 imágenes

USO:
    python safeguard_keypoints_local.py
"""

import cv2
import numpy as np
import pandas as pd
import os
import json
import urllib.request
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    ⚠️ CONFIGURA TUS RUTAS AQUÍ ⚠️                             ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Si usas Google Drive for Desktop, las rutas serían algo como:               ║
# ║  Windows: G:\Mi unidad\...  o  G:\My Drive\...                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Dataset le2i (estructura: carpeta/subcarpetas/imágenes)
DATASET_LE2I = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\dataset\le2i"

# Dataset ur_fall (estructura: carpeta/emergencias|simuladas/imágenes)
DATASET_URFALL = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\dataset\ur_fall\frames_from_videos"

# Carpeta de salida para los CSVs
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_keypoints_heavy"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURACIÓN DEL MODELO                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker_heavy.task"

# Nombres de los 33 keypoints de BlazePose
KEYPOINT_NAMES = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky', 'right_pinky',
    'left_index', 'right_index',
    'left_thumb', 'right_thumb',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]


def download_model():
    """Descarga el modelo de BlazePose si no existe."""
    if not os.path.exists(MODEL_PATH):
        print(f"📥 Descargando modelo BlazePose...")
        print(f"   URL: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"   ✅ Modelo descargado: {MODEL_PATH}")
    else:
        print(f"✅ Modelo encontrado: {MODEL_PATH}")


def init_detector():
    """Inicializa el detector de pose."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector


def extract_keypoints(image_path, detector):
    """
    Extrae los 33 keypoints de una imagen.
    
    Returns:
        dict con keypoints o None si no detecta persona
    """
    import mediapipe as mp
    
    try:
        image = mp.Image.create_from_file(image_path)
        detection_result = detector.detect(image)
        
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return None
        
        landmarks = detection_result.pose_landmarks[0]
        
        keypoints = {}
        for idx, landmark in enumerate(landmarks):
            name = KEYPOINT_NAMES[idx]
            keypoints[f"{name}_x"] = landmark.x
            keypoints[f"{name}_y"] = landmark.y
            keypoints[f"{name}_z"] = landmark.z
            keypoints[f"{name}_vis"] = landmark.visibility
        
        return keypoints
    
    except Exception as e:
        return None


def classify_le2i(folder_name):
    """Clasifica carpeta del dataset le2i."""
    name_lower = folder_name.lower()
    if name_lower.startswith('fall') or name_lower.startswith('likefall'):
        return 1, 'fall'
    else:
        return 0, 'adl'


def classify_urfall(folder_name):
    """Clasifica carpeta del dataset ur_fall."""
    name_lower = folder_name.lower()
    if 'emergencia' in name_lower or 'fall' in name_lower:
        return 1, 'fall'
    else:
        return 0, 'adl'


def get_all_images(dataset_path, dataset_name):
    """Obtiene lista de todas las imágenes con su clasificación."""
    images = []
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    if not os.path.exists(dataset_path):
        print(f"⚠️ No existe: {dataset_path}")
        return images
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(extensions):
                image_path = os.path.join(root, file)
                parent_folder = os.path.basename(root)
                
                if dataset_name == 'le2i':
                    label, label_name = classify_le2i(parent_folder)
                else:
                    label, label_name = classify_urfall(parent_folder)
                
                images.append({
                    'path': image_path,
                    'filename': file,
                    'folder': parent_folder,
                    'label': label,
                    'label_name': label_name,
                    'dataset': dataset_name
                })
    
    return images


def main():
    import time
    import mediapipe as mp
    
    print("\n")
    print("╔" + "═"*60 + "╗")
    print("║" + " 🛡️  SafeGuard Vision AI - Extracción de Keypoints ".center(60) + "║")
    print("║" + " BlazePose (MediaPipe) - Versión Local ".center(60) + "║")
    print("║" + " MIT Global Teaching Labs ".center(60) + "║")
    print("╚" + "═"*60 + "╝")
    
    # Verificar rutas
    print("\n📁 Verificando rutas...")
    
    if not os.path.exists(DATASET_LE2I):
        print(f"\n❌ ERROR: No existe la carpeta le2i:")
        print(f"   {DATASET_LE2I}")
        print("\n📝 Edita la variable DATASET_LE2I al inicio del script.")
        return
    
    if not os.path.exists(DATASET_URFALL):
        print(f"\n❌ ERROR: No existe la carpeta ur_fall:")
        print(f"   {DATASET_URFALL}")
        print("\n📝 Edita la variable DATASET_URFALL al inicio del script.")
        return
    
    print(f"   ✅ le2i:    {DATASET_LE2I}")
    print(f"   ✅ ur_fall: {DATASET_URFALL}")
    
    # Descargar modelo si no existe
    print("\n📦 Verificando modelo...")
    download_model()
    
    # Inicializar detector
    print("\n🔧 Inicializando BlazePose...")
    detector = init_detector()
    print(f"   ✅ MediaPipe v{mp.__version__}")
    print(f"   ✅ Modelo: pose_landmarker_heavy")
    
    # Obtener todas las imágenes
    print("\n📂 Escaneando datasets...")
    images_le2i = get_all_images(DATASET_LE2I, 'le2i')
    images_urfall = get_all_images(DATASET_URFALL, 'ur_fall')
    
    all_images = images_le2i + images_urfall
    
    falls = sum(1 for img in all_images if img['label'] == 1)
    adls = sum(1 for img in all_images if img['label'] == 0)
    
    print("\n" + "═"*60)
    print(f"   📁 Dataset le2i:      {len(images_le2i):>6} imágenes")
    print(f"   📁 Dataset ur_fall:   {len(images_urfall):>6} imágenes")
    print("─"*60)
    print(f"   🔴 Caídas (label=1):  {falls:>6} imágenes")
    print(f"   🟢 ADL (label=0):     {adls:>6} imágenes")
    print("─"*60)
    print(f"   📊 TOTAL:             {len(all_images):>6} imágenes")
    print("═"*60)
    
    # Estimar tiempo
    tiempo_est = len(all_images) * 0.05 / 60  # ~0.05 seg/img con RTX 4070
    print(f"\n⏱️  Tiempo estimado: ~{tiempo_est:.0f}-{tiempo_est*1.5:.0f} minutos")
    
    # Crear carpeta de salida
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Procesar imágenes
    print("\n🚀 Iniciando extracción de keypoints...\n")
    
    all_data = []
    failed_images = []
    
    start_time = time.time()
    
    for img_info in tqdm(all_images, desc="Procesando", unit="img", 
                         bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
        try:
            keypoints = extract_keypoints(img_info['path'], detector)
            
            if keypoints:
                row = {
                    'filename': img_info['filename'],
                    'folder': img_info['folder'],
                    'dataset': img_info['dataset'],
                    'label': img_info['label'],
                    'label_name': img_info['label_name']
                }
                row.update(keypoints)
                all_data.append(row)
            else:
                failed_images.append(img_info['path'])
        
        except Exception as e:
            failed_images.append(img_info['path'])
    
    elapsed_time = time.time() - start_time
    
    # Crear DataFrame y guardar
    print("\n💾 Guardando resultados...")
    
    df = pd.DataFrame(all_data)
    
    csv_path = os.path.join(OUTPUT_FOLDER, "keypoints_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    if failed_images:
        failed_path = os.path.join(OUTPUT_FOLDER, "failed_images.txt")
        with open(failed_path, 'w') as f:
            for path in failed_images:
                f.write(path + '\n')
    
    # Guardar resumen JSON
    summary = {
        "created": datetime.now().isoformat(),
        "project": "SafeGuard Vision AI",
        "author": "Christian Cajusol - MIT Global Teaching Labs",
        "model": "BlazePose (MediaPipe PoseLandmarker Heavy)",
        "mediapipe_version": mp.__version__,
        "execution": "Local (RTX 4070 + i9-14gen)",
        "statistics": {
            "total_processed": len(all_data),
            "failed": len(failed_images),
            "success_rate": f"{len(all_data)/(len(all_data)+len(failed_images))*100:.1f}%",
            "falls": int(df[df['label']==1].shape[0]) if len(df) > 0 else 0,
            "adl": int(df[df['label']==0].shape[0]) if len(df) > 0 else 0,
            "by_dataset": df.groupby('dataset').size().to_dict() if len(df) > 0 else {}
        },
        "features": {
            "keypoints": 33,
            "values_per_keypoint": 4,
            "total_features": 132
        },
        "processing_time_minutes": round(elapsed_time/60, 2)
    }
    
    json_path = os.path.join(OUTPUT_FOLDER, "keypoints_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Resumen final
    print("\n")
    print("╔" + "═"*60 + "╗")
    print("║" + " 🎉 EXTRACCIÓN COMPLETADA ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print(f"║  ⏱️  Tiempo total:       {elapsed_time/60:>10.1f} minutos{' '*17}║")
    print(f"║  ⚡ Velocidad:          {len(all_data)/elapsed_time:>10.1f} imgs/seg{' '*16}║")
    print("╠" + "═"*60 + "╣")
    print(f"║  📊 Imágenes procesadas: {len(all_data):>10}{' '*23}║")
    print(f"║  ❌ Imágenes fallidas:   {len(failed_images):>10}{' '*23}║")
    print(f"║  ✅ Tasa de éxito:      {summary['statistics']['success_rate']:>10}{' '*23}║")
    print("╠" + "═"*60 + "╣")
    print(f"║  🔴 Caídas (label=1):    {summary['statistics']['falls']:>10}{' '*23}║")
    print(f"║  🟢 ADL (label=0):       {summary['statistics']['adl']:>10}{' '*23}║")
    print("╠" + "═"*60 + "╣")
    print(f"║  📄 CSV:  keypoints_dataset.csv{' '*26}║")
    print(f"║  📄 JSON: keypoints_summary.json{' '*25}║")
    print(f"║  📂 Ruta: {OUTPUT_FOLDER[:48]:<49}║")
    print("╚" + "═"*60 + "╝")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("   1. Entrenar clasificador (Random Forest, SVM, Neural Network)")
    print("   2. Evaluar métricas (accuracy, precision, recall, F1)")
    print("   3. Deploy en Jetson Orin Nano")
    print()


if __name__ == "__main__":
    main()
