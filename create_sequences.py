"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - CREADOR DE SECUENCIAS                           ║
║                                                                              ║
║   Convierte frames individuales en secuencias temporales para LSTM          ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

¿POR QUÉ SECUENCIAS?
====================
    Random Forest: Ve 1 frame → "¿Pose de caída?" → Falla con personas acostadas
    
    LSTM: Ve 30 frames → "¿Hubo TRANSICIÓN de caída?" → Detecta el MOVIMIENTO

LÓGICA:
=======
    1. Agrupa frames por carpeta (cada carpeta = 1 video)
    2. Ordena frames cronológicamente
    3. Crea ventanas deslizantes de N frames
    4. Etiqueta: ¿Esta secuencia contiene una transición de caída?

ENTRADA:
========
    - keypoints_HEAVY.csv (frames individuales con keypoints)

SALIDA:
=======
    - sequences_train.npy (secuencias de entrenamiento)
    - sequences_test.npy (secuencias de test)
    - labels_train.npy (etiquetas)
    - labels_test.npy (etiquetas)
    - sequences_metadata.json (información del procesamiento)
"""

import pandas as pd
import numpy as np
import os
import json
import re
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CONFIGURACIÓN
# ╚══════════════════════════════════════════════════════════════════════════════╝

# CSV con keypoints (el original, no el balanceado)
INPUT_CSV = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_keypoints_heavy\keypoints_dataset.csv"

# Carpeta de salida
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_sequences"

# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE SECUENCIAS
# ═══════════════════════════════════════════════════════════════════════════════

# Longitud de cada secuencia (número de frames)
# 30 frames ≈ 1 segundo a 30fps
SEQUENCE_LENGTH = 30

# Paso entre secuencias (stride)
# 15 = 50% overlap entre secuencias consecutivas
SEQUENCE_STRIDE = 15

# Mínimo de frames por video para crear secuencias
MIN_FRAMES_PER_VIDEO = SEQUENCE_LENGTH

# Test split
TEST_SIZE = 0.2
RANDOM_SEED = 42


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         FUNCIONES
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_frame_number(filename):
    """
    Extrae el número de frame del nombre de archivo.
    Ejemplos: 'frame_001.png' → 1, 'img0042.jpg' → 42
    """
    # Buscar números en el nombre
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # Tomar el último número (usualmente es el número de frame)
        return int(numbers[-1])
    return 0


def load_and_organize_data(csv_path):
    """
    Carga el CSV y organiza los datos por video/carpeta.
    """
    print("\n" + "="*70)
    print("📂 CARGANDO Y ORGANIZANDO DATOS")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    print(f"\n   📊 Total frames: {len(df):,}")
    
    # Identificar columnas
    metadata_cols = ['filename', 'folder', 'dataset', 'label', 'label_name']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"   📐 Features por frame: {len(feature_cols)}")
    
    # Organizar por carpeta (cada carpeta = 1 video)
    videos = defaultdict(list)
    
    for idx, row in df.iterrows():
        folder = row['folder']
        frame_num = extract_frame_number(row['filename'])
        
        videos[folder].append({
            'frame_num': frame_num,
            'features': row[feature_cols].values.astype(np.float32),
            'label': row['label'],
            'filename': row['filename']
        })
    
    # Ordenar frames dentro de cada video
    for folder in videos:
        videos[folder].sort(key=lambda x: x['frame_num'])
    
    print(f"   📁 Videos encontrados: {len(videos)}")
    
    # Estadísticas
    frames_per_video = [len(v) for v in videos.values()]
    print(f"   📈 Frames por video: min={min(frames_per_video)}, max={max(frames_per_video)}, avg={np.mean(frames_per_video):.1f}")
    
    # Contar videos con suficientes frames
    valid_videos = sum(1 for v in videos.values() if len(v) >= MIN_FRAMES_PER_VIDEO)
    print(f"   ✅ Videos con ≥{MIN_FRAMES_PER_VIDEO} frames: {valid_videos}")
    
    return videos, feature_cols


def create_sequences_from_video(frames, sequence_length, stride):
    """
    Crea secuencias a partir de los frames de un video.
    
    Args:
        frames: Lista de dicts con 'features' y 'label'
        sequence_length: Longitud de cada secuencia
        stride: Paso entre secuencias
        
    Returns:
        sequences: Lista de arrays (sequence_length, n_features)
        labels: Lista de etiquetas (0 o 1)
        types: Lista de tipos de secuencia ('transition', 'fall', 'adl')
    """
    sequences = []
    labels = []
    seq_types = []
    
    n_frames = len(frames)
    
    if n_frames < sequence_length:
        return sequences, labels, seq_types
    
    # Crear ventanas deslizantes
    for start in range(0, n_frames - sequence_length + 1, stride):
        end = start + sequence_length
        
        # Extraer features de la secuencia
        seq_features = np.array([f['features'] for f in frames[start:end]])
        
        # Extraer labels de cada frame
        frame_labels = [f['label'] for f in frames[start:end]]
        
        # Determinar etiqueta de la secuencia
        # LÓGICA CLAVE: Detectar TRANSICIÓN (de normal a caída)
        
        n_falls = sum(frame_labels)
        n_adl = len(frame_labels) - n_falls
        
        # Dividir secuencia en mitades
        first_half_labels = frame_labels[:sequence_length//2]
        second_half_labels = frame_labels[sequence_length//2:]
        
        first_half_falls = sum(first_half_labels)
        second_half_falls = sum(second_half_labels)
        
        # TRANSICIÓN: Primera mitad mayormente ADL, segunda mitad mayormente caída
        if first_half_falls < len(first_half_labels) * 0.3 and second_half_falls > len(second_half_labels) * 0.7:
            # ¡Transición detectada! Esta es una caída real
            seq_label = 1
            seq_type = 'transition'
        elif n_falls > sequence_length * 0.8:
            # Mayormente caída (persona ya en el suelo)
            seq_label = 1
            seq_type = 'fall_static'
        elif n_falls < sequence_length * 0.2:
            # Mayormente ADL
            seq_label = 0
            seq_type = 'adl'
        else:
            # Mixto - usar etiqueta del último frame
            seq_label = frame_labels[-1]
            seq_type = 'mixed'
        
        sequences.append(seq_features)
        labels.append(seq_label)
        seq_types.append(seq_type)
    
    return sequences, labels, seq_types


def create_all_sequences(videos, sequence_length, stride):
    """
    Crea secuencias de todos los videos.
    """
    print("\n" + "="*70)
    print("🔄 CREANDO SECUENCIAS TEMPORALES")
    print("="*70)
    
    all_sequences = []
    all_labels = []
    all_types = []
    video_info = []
    
    skipped = 0
    
    for folder, frames in videos.items():
        if len(frames) < sequence_length:
            skipped += 1
            continue
        
        seqs, labels, types = create_sequences_from_video(frames, sequence_length, stride)
        
        for seq, label, seq_type in zip(seqs, labels, types):
            all_sequences.append(seq)
            all_labels.append(label)
            all_types.append(seq_type)
            video_info.append(folder)
    
    print(f"\n   ⏭️  Videos saltados (muy cortos): {skipped}")
    print(f"   📊 Secuencias creadas: {len(all_sequences):,}")
    
    # Convertir a arrays
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    print(f"   📐 Shape de datos: {X.shape}")
    print(f"      → ({X.shape[0]} secuencias, {X.shape[1]} frames, {X.shape[2]} features)")
    
    # Estadísticas de etiquetas
    type_counts = defaultdict(int)
    for t in all_types:
        type_counts[t] += 1
    
    print(f"\n   📋 Distribución por tipo de secuencia:")
    for t, count in sorted(type_counts.items()):
        print(f"      • {t}: {count:,} ({count/len(all_types)*100:.1f}%)")
    
    # Distribución de clases
    n_falls = sum(y)
    n_adl = len(y) - n_falls
    print(f"\n   📊 Distribución de clases:")
    print(f"      • Caídas: {n_falls:,} ({n_falls/len(y)*100:.1f}%)")
    print(f"      • ADL:    {n_adl:,} ({n_adl/len(y)*100:.1f}%)")
    
    return X, y, all_types, video_info


def add_temporal_features(X):
    """
    Añade features temporales a las secuencias.
    Estas features ayudan al LSTM a detectar cambios bruscos.
    
    Features añadidas por frame:
    - Velocidad (diferencia con frame anterior)
    - Aceleración (diferencia de velocidad)
    """
    print("\n🔧 Añadiendo features temporales...")
    
    n_sequences, n_frames, n_features = X.shape
    
    # Calcular velocidades (diferencias entre frames consecutivos)
    velocities = np.zeros_like(X)
    velocities[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    
    # Calcular aceleraciones (diferencias de velocidades)
    accelerations = np.zeros_like(X)
    accelerations[:, 1:, :] = velocities[:, 1:, :] - velocities[:, :-1, :]
    
    # Concatenar: [posición, velocidad, aceleración]
    X_enhanced = np.concatenate([X, velocities, accelerations], axis=2)
    
    print(f"   📐 Shape original: {X.shape}")
    print(f"   📐 Shape con temporales: {X_enhanced.shape}")
    print(f"   ✅ Features por frame: {n_features} → {X_enhanced.shape[2]}")
    
    return X_enhanced


def balance_sequences(X, y, types, random_seed=42):
    """
    Balancea las secuencias priorizando transiciones.
    """
    print("\n⚖️ Balanceando secuencias...")
    
    np.random.seed(random_seed)
    
    # Separar por clase
    fall_indices = np.where(y == 1)[0]
    adl_indices = np.where(y == 0)[0]
    
    n_falls = len(fall_indices)
    n_adl = len(adl_indices)
    
    print(f"   Antes: {n_falls} caídas, {n_adl} ADL")
    
    # Balancear
    if n_falls < n_adl:
        # Reducir ADL
        adl_indices = np.random.choice(adl_indices, size=n_falls, replace=False)
    elif n_adl < n_falls:
        # Reducir caídas
        fall_indices = np.random.choice(fall_indices, size=n_adl, replace=False)
    
    # Combinar y mezclar
    balanced_indices = np.concatenate([fall_indices, adl_indices])
    np.random.shuffle(balanced_indices)
    
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    print(f"   Después: {sum(y_balanced)} caídas, {len(y_balanced) - sum(y_balanced)} ADL")
    print(f"   ✅ Total: {len(y_balanced)} secuencias balanceadas")
    
    return X_balanced, y_balanced


def split_and_save(X, y, output_folder, test_size=0.2, random_seed=42):
    """
    Divide en train/test y guarda los arrays.
    """
    print("\n" + "="*70)
    print("💾 GUARDANDO SECUENCIAS")
    print("="*70)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_seed,
        stratify=y
    )
    
    print(f"\n   📊 División de datos:")
    print(f"      Train: {len(X_train):,} secuencias ({sum(y_train)} caídas)")
    print(f"      Test:  {len(X_test):,} secuencias ({sum(y_test)} caídas)")
    
    # Guardar arrays
    np.save(os.path.join(output_folder, "X_train.npy"), X_train)
    np.save(os.path.join(output_folder, "X_test.npy"), X_test)
    np.save(os.path.join(output_folder, "y_train.npy"), y_train)
    np.save(os.path.join(output_folder, "y_test.npy"), y_test)
    
    print(f"\n   ✅ Archivos guardados:")
    print(f"      • X_train.npy: {X_train.shape}")
    print(f"      • X_test.npy:  {X_test.shape}")
    print(f"      • y_train.npy: {y_train.shape}")
    print(f"      • y_test.npy:  {y_test.shape}")
    
    # Guardar metadata
    metadata = {
        "created": datetime.now().isoformat(),
        "project": "SafeGuard Vision AI",
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_stride": SEQUENCE_STRIDE,
        "features_per_frame": X.shape[2],
        "total_sequences": len(X),
        "train_sequences": len(X_train),
        "test_sequences": len(X_test),
        "train_falls": int(sum(y_train)),
        "test_falls": int(sum(y_test)),
        "shape": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape)
        }
    }
    
    meta_path = os.path.join(output_folder, "sequences_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"      • sequences_metadata.json")
    
    return X_train, X_test, y_train, y_test


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                              MAIN
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SAFEGUARD VISION AI - CREADOR DE SECUENCIAS ".center(70) + "║")
    print("║" + " Preparando datos para LSTM ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    print(f"\n   ⚙️  Configuración:")
    print(f"   ├── Longitud de secuencia: {SEQUENCE_LENGTH} frames")
    print(f"   ├── Stride: {SEQUENCE_STRIDE} frames")
    print(f"   └── Test size: {TEST_SIZE*100}%")
    
    # Verificar archivo
    if not os.path.exists(INPUT_CSV):
        print(f"\n❌ ERROR: No se encontró: {INPUT_CSV}")
        return
    
    # Cargar y organizar datos
    videos, feature_cols = load_and_organize_data(INPUT_CSV)
    
    # Crear secuencias
    X, y, types, video_info = create_all_sequences(videos, SEQUENCE_LENGTH, SEQUENCE_STRIDE)
    
    # Añadir features temporales (velocidad, aceleración)
    X = add_temporal_features(X)
    
    # Balancear
    X, y = balance_sequences(X, y, types, RANDOM_SEED)
    
    # Guardar
    X_train, X_test, y_train, y_test = split_and_save(X, y, OUTPUT_FOLDER, TEST_SIZE, RANDOM_SEED)
    
    # Resumen final
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " ✅ SECUENCIAS CREADAS EXITOSAMENTE ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📊 Total secuencias: {len(X):,}".ljust(71) + "║")
    print(f"║  📐 Shape: ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})".ljust(71) + "║")
    print(f"║     → {X.shape[0]} secuencias de {X.shape[1]} frames".ljust(71) + "║")
    print(f"║     → {X.shape[2]} features por frame".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📂 Guardado en: {OUTPUT_FOLDER}".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print("║  📋 SIGUIENTE PASO:".ljust(71) + "║")
    print("║     python train_lstm_detector.py".ljust(71) + "║")
    print("╚" + "═"*70 + "╝")
    print()


if __name__ == "__main__":
    main()
