"""
🛡️ SafeGuard Vision AI - BlazePose FULL
=========================================
Modelo FULL: Balance entre velocidad y precisión
Ideal para: Aplicaciones generales

Autor: Christian Cajusol - MIT Global Teaching Labs
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
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Dataset le2i (estructura: carpeta/subcarpetas/imágenes)
DATASET_LE2I = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\dataset\le2i"

# Dataset ur_fall (estructura: carpeta/emergencias|simuladas/imágenes)
DATASET_URFALL = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\dataset\ur_fall\frames_from_videos"

# Carpeta de salida para los CSVs
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_keypoints_full"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    MODELO FULL                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker_full.task"
MODEL_NAME = "FULL"

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
    if not os.path.exists(MODEL_PATH):
        print(f"📥 Descargando modelo BlazePose {MODEL_NAME}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"   ✅ Modelo descargado: {MODEL_PATH}")
    else:
        print(f"✅ Modelo encontrado: {MODEL_PATH}")


def init_detector():
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
    
    return vision.PoseLandmarker.create_from_options(options)


def extract_keypoints(image_path, detector):
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
    name_lower = folder_name.lower()
    if name_lower.startswith('fall') or name_lower.startswith('likefall'):
        return 1, 'fall'
    return 0, 'adl'


def classify_urfall(folder_name):
    name_lower = folder_name.lower()
    if 'emergencia' in name_lower or 'fall' in name_lower:
        return 1, 'fall'
    return 0, 'adl'


def get_all_images(dataset_path, dataset_name):
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
    print("║" + f" 🛡️  SafeGuard Vision AI - BlazePose {MODEL_NAME} ".center(60) + "║")
    print("║" + " (Balance velocidad/precisión) ".center(60) + "║")
    print("╚" + "═"*60 + "╝")
    
    # Verificar rutas
    if not os.path.exists(DATASET_LE2I):
        print(f"\n❌ ERROR: No existe: {DATASET_LE2I}")
        return
    if not os.path.exists(DATASET_URFALL):
        print(f"\n❌ ERROR: No existe: {DATASET_URFALL}")
        return
    
    download_model()
    
    print("\n🔧 Inicializando BlazePose FULL...")
    detector = init_detector()
    print(f"   ✅ MediaPipe v{mp.__version__}")
    
    print("\n📂 Escaneando datasets...")
    images_le2i = get_all_images(DATASET_LE2I, 'le2i')
    images_urfall = get_all_images(DATASET_URFALL, 'ur_fall')
    all_images = images_le2i + images_urfall
    
    print(f"   📊 Total: {len(all_images)} imágenes")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    all_data = []
    failed_images = []
    
    print("\n🚀 Extrayendo keypoints con modelo FULL...\n")
    start_time = time.time()
    
    for img_info in tqdm(all_images, desc="FULL", unit="img"):
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
        except:
            failed_images.append(img_info['path'])
    
    elapsed_time = time.time() - start_time
    
    # Guardar resultados
    df = pd.DataFrame(all_data)
    csv_path = os.path.join(OUTPUT_FOLDER, "keypoints_FULL.csv")
    df.to_csv(csv_path, index=False)
    
    if failed_images:
        with open(os.path.join(OUTPUT_FOLDER, "failed_FULL.txt"), 'w') as f:
            for path in failed_images:
                f.write(path + '\n')
    
    summary = {
        "model": "BlazePose FULL",
        "model_file": MODEL_PATH,
        "created": datetime.now().isoformat(),
        "processing_time_seconds": round(elapsed_time, 2),
        "processing_time_minutes": round(elapsed_time/60, 2),
        "images_per_second": round(len(all_images)/elapsed_time, 2),
        "total_processed": len(all_data),
        "failed": len(failed_images),
        "success_rate": f"{len(all_data)/(len(all_data)+len(failed_images))*100:.2f}%",
        "falls": int(df[df['label']==1].shape[0]) if len(df) > 0 else 0,
        "adl": int(df[df['label']==0].shape[0]) if len(df) > 0 else 0
    }
    
    with open(os.path.join(OUTPUT_FOLDER, "summary_FULL.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Resumen
    print("\n")
    print("╔" + "═"*60 + "╗")
    print("║" + f" 🎉 COMPLETADO - BlazePose {MODEL_NAME} ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print(f"║  ⏱️  Tiempo total:       {elapsed_time/60:>10.2f} minutos{' '*17}║")
    print(f"║  ⚡ Velocidad:          {len(all_images)/elapsed_time:>10.2f} imgs/seg{' '*16}║")
    print(f"║  📊 Procesadas:         {len(all_data):>10}{' '*23}║")
    print(f"║  ❌ Fallidas:           {len(failed_images):>10}{' '*23}║")
    print(f"║  ✅ Tasa éxito:         {summary['success_rate']:>10}{' '*23}║")
    print("╚" + "═"*60 + "╝")


if __name__ == "__main__":
    main()
