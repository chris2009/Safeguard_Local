"""
🛡️ SafeGuard Vision AI - Frame Extractor para Dataset
======================================================
Extrae imágenes de videos para entrenar BlazePose y modelo de clasificación.

Autor: Christian Cajusol - MIT Global Teaching Labs
Fecha: Enero 2026

USO:
    python extract_frames_safeguard.py

ESTRUCTURA DE ENTRADA ESPERADA:
    videos/
    ├── emergencias/     ← 30 videos de 2 seg (caídas reales)
    └── simuladas/       ← 40 videos de 11 seg (caídas actuadas)

ESTRUCTURA DE SALIDA:
    dataset/
    ├── emergencias/     ← Frames etiquetados como emergencia
    │   ├── emergencia_video01_frame001.jpg
    │   ├── emergencia_video01_frame002.jpg
    │   └── ...
    └── simuladas/       ← Frames etiquetados como simulada
        ├── simulada_video01_frame001.jpg
        └── ...
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           CONFIGURACIÓN                                       ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  MODIFICA ESTAS VARIABLES SEGÚN TU ESTRUCTURA DE CARPETAS                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Carpetas de entrada (donde están tus videos)
VIDEOS_EMERGENCIAS = r"C:\Users\TU_USUARIO\Videos\emergencias"
VIDEOS_SIMULADAS = r"C:\Users\TU_USUARIO\Videos\simuladas"

# Carpeta de salida (donde se guardarán los frames)
OUTPUT_DATASET = r"C:\Users\TU_USUARIO\Dataset\safeguard_frames"

# Configuración de extracción
CONFIG = {
    "emergencias": {
        "fps_extract": 15,      # Extraer 15 frames por segundo (de ~30fps original)
        "prefix": "emerg",      # Prefijo para nombres de archivo
        "label": 1              # Etiqueta numérica: 1 = emergencia real
    },
    "simuladas": {
        "fps_extract": 10,      # Extraer 10 frames por segundo (menos porque son más largos)
        "prefix": "simul",      # Prefijo para nombres de archivo  
        "label": 0              # Etiqueta numérica: 0 = simulada/no emergencia
    }
}

# Formato de imagen de salida
IMAGE_FORMAT = "jpg"           # jpg o png
JPEG_QUALITY = 95              # Calidad si usas jpg (1-100)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           FUNCIONES                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_video_info(video_path):
    """Obtiene información del video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return info


def extract_frames_from_video(video_path, output_folder, prefix, fps_extract, label, video_num):
    """
    Extrae frames de un video a una tasa específica.
    
    Args:
        video_path: Ruta al video
        output_folder: Carpeta donde guardar frames
        prefix: Prefijo para nombres (emerg/simul)
        fps_extract: Cuántos frames extraer por segundo
        label: Etiqueta de clase (0 o 1)
        video_num: Número de video para naming
    
    Returns:
        dict con estadísticas de extracción
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"No se pudo abrir: {video_path}"}
    
    # Info del video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calcular intervalo de extracción
    # Si video es 30fps y queremos 15fps, extraemos cada 2 frames
    frame_interval = max(1, int(video_fps / fps_extract))
    
    frames_extracted = 0
    frame_count = 0
    frame_data = []  # Para el CSV/JSON de metadata
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extraer solo cada N frames según el intervalo
        if frame_count % frame_interval == 0:
            frames_extracted += 1
            
            # Nombre del archivo: prefix_videoXX_frameXXX.jpg
            filename = f"{prefix}_v{video_num:03d}_f{frames_extracted:04d}.{IMAGE_FORMAT}"
            filepath = os.path.join(output_folder, filename)
            
            # Guardar frame
            if IMAGE_FORMAT == "jpg":
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            else:
                cv2.imwrite(filepath, frame)
            
            # Metadata para CSV
            frame_data.append({
                "filename": filename,
                "video_source": os.path.basename(video_path),
                "frame_number": frame_count,
                "label": label,
                "label_name": "emergencia" if label == 1 else "simulada",
                "timestamp_sec": frame_count / video_fps
            })
        
        frame_count += 1
    
    cap.release()
    
    return {
        "video": os.path.basename(video_path),
        "total_frames_video": total_frames,
        "frames_extracted": frames_extracted,
        "frame_interval": frame_interval,
        "frame_data": frame_data
    }


def process_category(videos_folder, output_folder, config, category_name):
    """Procesa todos los videos de una categoría."""
    
    # Crear carpeta de salida
    os.makedirs(output_folder, exist_ok=True)
    
    # Listar videos
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI')
    videos = sorted([f for f in os.listdir(videos_folder) 
                     if f.endswith(video_extensions)])
    
    if not videos:
        print(f"⚠️  No se encontraron videos en: {videos_folder}")
        return None
    
    print(f"\n{'='*60}")
    print(f"📂 Procesando: {category_name.upper()}")
    print(f"{'='*60}")
    print(f"   📁 Entrada:  {videos_folder}")
    print(f"   📁 Salida:   {output_folder}")
    print(f"   🎬 Videos:   {len(videos)}")
    print(f"   ⚡ FPS ext:  {config['fps_extract']} fps")
    print(f"{'='*60}\n")
    
    all_frame_data = []
    total_frames = 0
    
    for i, video_name in enumerate(tqdm(videos, desc=f"   {category_name}")):
        video_path = os.path.join(videos_folder, video_name)
        
        result = extract_frames_from_video(
            video_path=video_path,
            output_folder=output_folder,
            prefix=config["prefix"],
            fps_extract=config["fps_extract"],
            label=config["label"],
            video_num=i + 1
        )
        
        if "error" not in result:
            total_frames += result["frames_extracted"]
            all_frame_data.extend(result["frame_data"])
    
    return {
        "category": category_name,
        "videos_processed": len(videos),
        "total_frames": total_frames,
        "frame_data": all_frame_data
    }


def save_metadata(all_data, output_folder):
    """Guarda metadata en CSV y JSON para uso posterior."""
    
    # Combinar todos los frame_data
    all_frames = []
    for category_data in all_data:
        if category_data:
            all_frames.extend(category_data["frame_data"])
    
    # Guardar CSV
    csv_path = os.path.join(output_folder, "dataset_metadata.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("filename,video_source,frame_number,label,label_name,timestamp_sec\n")
        for frame in all_frames:
            f.write(f"{frame['filename']},{frame['video_source']},{frame['frame_number']},"
                   f"{frame['label']},{frame['label_name']},{frame['timestamp_sec']:.3f}\n")
    
    # Guardar JSON con info completa
    json_path = os.path.join(output_folder, "dataset_info.json")
    summary = {
        "created": datetime.now().isoformat(),
        "project": "SafeGuard Vision AI",
        "author": "Christian Cajusol - MIT Global Teaching Labs",
        "image_size": "320x240",
        "format": IMAGE_FORMAT,
        "categories": {
            "emergencias": {"label": 1, "description": "Caídas reales/emergencias"},
            "simuladas": {"label": 0, "description": "Caídas actuadas/simuladas"}
        },
        "statistics": {
            cat["category"]: {
                "videos": cat["videos_processed"],
                "frames": cat["total_frames"]
            } for cat in all_data if cat
        },
        "total_frames": sum(cat["total_frames"] for cat in all_data if cat)
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return csv_path, json_path


def print_summary(all_data, csv_path, json_path):
    """Imprime resumen final."""
    
    total_frames = sum(cat["total_frames"] for cat in all_data if cat)
    
    print("\n")
    print("╔" + "═"*60 + "╗")
    print("║" + " 🎉 EXTRACCIÓN COMPLETADA ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    
    for cat in all_data:
        if cat:
            print(f"║  📂 {cat['category'].upper():<20} {cat['total_frames']:>6} frames  ({cat['videos_processed']} videos)  ║")
    
    print("╠" + "═"*60 + "╣")
    print(f"║  📊 TOTAL DATASET: {total_frames:>10} frames".ljust(61) + "║")
    print("╠" + "═"*60 + "╣")
    print(f"║  📄 CSV:  {os.path.basename(csv_path):<47} ║")
    print(f"║  📄 JSON: {os.path.basename(json_path):<47} ║")
    print("╚" + "═"*60 + "╝")
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Usa el CSV para cargar datos en tu entrenamiento")
    print("   2. Aplica BlazePose para extraer keypoints")
    print("   3. Entrena clasificador emergencia vs simulada")
    print()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MAIN                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n")
    print("╔" + "═"*60 + "╗")
    print("║" + " 🛡️  SafeGuard Vision AI - Frame Extractor ".center(60) + "║")
    print("║" + " MIT Global Teaching Labs ".center(60) + "║")
    print("╚" + "═"*60 + "╝")
    
    # Verificar carpetas de entrada
    if not os.path.exists(VIDEOS_EMERGENCIAS):
        print(f"\n❌ ERROR: No existe la carpeta de emergencias:")
        print(f"   {VIDEOS_EMERGENCIAS}")
        print("\n📝 Edita las rutas al inicio del script.")
        return
    
    if not os.path.exists(VIDEOS_SIMULADAS):
        print(f"\n❌ ERROR: No existe la carpeta de simuladas:")
        print(f"   {VIDEOS_SIMULADAS}")
        print("\n📝 Edita las rutas al inicio del script.")
        return
    
    # Crear carpeta principal de salida
    os.makedirs(OUTPUT_DATASET, exist_ok=True)
    
    # Mostrar preview de un video de cada categoría
    print("\n📹 Preview de videos encontrados:")
    for name, folder in [("Emergencias", VIDEOS_EMERGENCIAS), ("Simuladas", VIDEOS_SIMULADAS)]:
        videos = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if videos:
            info = get_video_info(os.path.join(folder, videos[0]))
            if info:
                print(f"\n   {name}: {len(videos)} videos")
                print(f"   └─ Ejemplo: {videos[0]}")
                print(f"      Resolución: {info['width']}x{info['height']}")
                print(f"      FPS: {info['fps']:.1f}")
                print(f"      Duración: {info['duration_sec']:.1f} seg")
    
    # Procesar cada categoría
    all_results = []
    
    # Emergencias
    result_emerg = process_category(
        videos_folder=VIDEOS_EMERGENCIAS,
        output_folder=os.path.join(OUTPUT_DATASET, "emergencias"),
        config=CONFIG["emergencias"],
        category_name="emergencias"
    )
    all_results.append(result_emerg)
    
    # Simuladas
    result_simul = process_category(
        videos_folder=VIDEOS_SIMULADAS,
        output_folder=os.path.join(OUTPUT_DATASET, "simuladas"),
        config=CONFIG["simuladas"],
        category_name="simuladas"
    )
    all_results.append(result_simul)
    
    # Guardar metadata
    csv_path, json_path = save_metadata(all_results, OUTPUT_DATASET)
    
    # Resumen final
    print_summary(all_results, csv_path, json_path)


if __name__ == "__main__":
    main()
