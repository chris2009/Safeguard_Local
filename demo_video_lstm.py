"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - DEMO LSTM EN TIEMPO REAL                        ║
║                                                                              ║
║   Detecta caídas analizando SECUENCIAS de frames (no frames individuales)   ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

DIFERENCIA CLAVE:
=================
    Demo anterior: Analiza 1 frame → "¿Pose de caída?"
    
    Esta demo: Analiza 30 frames → "¿Hubo TRANSICIÓN de caída?"
    
    RESULTADO: Distingue entre "persona acostada" y "persona que cayó"

CONTROLES:
==========
    Q: Salir
    P: Pausar
    S: Screenshot
    R: Reset
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CONFIGURACIÓN
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════════════════════════
# FUENTE DE VIDEO
# ═══════════════════════════════════════════════════════════════════════════════

VIDEO_SOURCE = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\video_prueba_4.mp4"  # O usa 0 para webcam

# ═══════════════════════════════════════════════════════════════════════════════
# MODELO LSTM
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_model_lstm"
SEQUENCES_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_sequences"  # Para normalization params

# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Secuencias
    "sequence_length": 30,      # Frames por secuencia (debe coincidir con entrenamiento)
    
    # Detección
    "fall_threshold": 0.5,      # Umbral para detectar caída
    "confirmation_frames": 2,   # Secuencias consecutivas para confirmar
    
    # Visualización
    "display_width": 1280,
    "display_height": 720,
    "save_output": True,
    "output_path": r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_lstm_output.mp4",
    
    # Colores (BGR)
    "color_safe": (0, 255, 0),
    "color_warning": (0, 255, 255),
    "color_danger": (0, 0, 255),
    "color_skeleton": (255, 128, 0),
}

# Nombres de keypoints
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

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CLASES
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LSTMFallDetector:
    """
    Detector de caídas basado en LSTM.
    Analiza secuencias temporales para detectar TRANSICIONES de caída.
    """
    
    def __init__(self, model_folder, sequences_folder, sequence_length=30):
        self.model_folder = model_folder
        self.sequences_folder = sequences_folder
        self.sequence_length = sequence_length
        
        # Buffer de frames (para crear secuencias)
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Estado
        self.fall_history = deque(maxlen=CONFIG['confirmation_frames'])
        self.last_state = "initializing"
        self.fall_count = 0
        self.frame_count = 0
        
        # Cargar modelo y componentes
        self._load_model()
        self._init_blazepose()
        self._load_normalization()
        
    def _load_model(self):
        """Carga el modelo LSTM."""
        print("📂 Cargando modelo LSTM...")
        
        model_path = os.path.join(self.model_folder, "modelo_lstm.h5")
        if not os.path.exists(model_path):
            # Intentar con best_model
            model_path = os.path.join(self.model_folder, "best_model.h5")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró modelo en: {self.model_folder}")
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"   ✅ Modelo LSTM cargado")
        
    def _init_blazepose(self):
        """Inicializa BlazePose."""
        print("🔧 Inicializando BlazePose...")
        
        model_filename = "pose_landmarker_full.task"
        
        if not os.path.exists(model_filename):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            urllib.request.urlretrieve(url, model_filename)
        
        base_options = python.BaseOptions(model_asset_path=model_filename)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)
        print("   ✅ BlazePose inicializado")
        
    def _load_normalization(self):
        """Carga parámetros de normalización."""
        print("📊 Cargando parámetros de normalización...")
        
        mean_path = os.path.join(self.sequences_folder, "norm_mean.npy")
        std_path = os.path.join(self.sequences_folder, "norm_std.npy")
        
        if os.path.exists(mean_path) and os.path.exists(std_path):
            self.norm_mean = np.load(mean_path)
            self.norm_std = np.load(std_path)
            print("   ✅ Normalización cargada")
        else:
            print("   ⚠️ No se encontraron parámetros de normalización")
            self.norm_mean = None
            self.norm_std = None
    
    def extract_keypoints(self, frame):
        """Extrae keypoints del frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = self.pose_detector.detect(mp_image)
        
        if not result.pose_landmarks:
            return None, None
        
        landmarks = result.pose_landmarks[0]
        
        # Extraer features (x, y, z, visibility por cada keypoint)
        features = []
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return np.array(features, dtype=np.float32), landmarks
    
    def add_temporal_features(self, sequence):
        """
        Añade velocidad y aceleración a la secuencia.
        """
        # sequence shape: (seq_len, n_features)
        velocities = np.zeros_like(sequence)
        velocities[1:] = sequence[1:] - sequence[:-1]
        
        accelerations = np.zeros_like(sequence)
        accelerations[1:] = velocities[1:] - velocities[:-1]
        
        # Concatenar: [posición, velocidad, aceleración]
        enhanced = np.concatenate([sequence, velocities, accelerations], axis=1)
        
        return enhanced
    
    def predict_sequence(self):
        """
        Predice si la secuencia actual contiene una caída.
        """
        if len(self.frame_buffer) < self.sequence_length:
            return 0, 0.0, "buffering"
        
        # Crear secuencia
        sequence = np.array(list(self.frame_buffer))
        
        # Añadir features temporales
        sequence = self.add_temporal_features(sequence)
        
        # Normalizar
        if self.norm_mean is not None:
            sequence = (sequence - self.norm_mean) / (self.norm_std + 1e-8)
        
        # Predecir
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        probability = self.model.predict(sequence, verbose=0)[0][0]
        
        prediction = 1 if probability >= CONFIG['fall_threshold'] else 0
        
        return prediction, probability, "ready"
    
    def process_frame(self, frame):
        """
        Procesa un frame: extrae keypoints, actualiza buffer, predice.
        """
        self.frame_count += 1
        start_time = time.perf_counter()
        
        result = {
            "frame_number": self.frame_count,
            "person_detected": False,
            "buffer_status": f"{len(self.frame_buffer)}/{self.sequence_length}",
            "prediction": 0,
            "probability": 0.0,
            "state": "no_person",
            "landmarks": None,
            "processing_time_ms": 0,
            "fall_count": self.fall_count
        }
        
        # Extraer keypoints
        keypoints, landmarks = self.extract_keypoints(frame)
        
        if keypoints is None:
            # No hay persona - mantener buffer pero no añadir
            result["state"] = "no_person"
            result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
            return result
        
        result["person_detected"] = True
        result["landmarks"] = landmarks
        
        # Añadir al buffer
        self.frame_buffer.append(keypoints)
        result["buffer_status"] = f"{len(self.frame_buffer)}/{self.sequence_length}"
        
        # Predecir si tenemos suficientes frames
        prediction, probability, status = self.predict_sequence()
        
        result["prediction"] = prediction
        result["probability"] = probability
        
        if status == "buffering":
            result["state"] = "buffering"
        else:
            # Actualizar historial
            self.fall_history.append(prediction)
            
            # Determinar estado
            recent_falls = sum(self.fall_history)
            
            if recent_falls >= CONFIG['confirmation_frames']:
                result["state"] = "fall_confirmed"
                if self.last_state != "fall_confirmed":
                    self.fall_count += 1
                    result["fall_count"] = self.fall_count
            elif prediction == 1:
                result["state"] = "fall_possible"
            else:
                result["state"] = "normal"
            
            self.last_state = result["state"]
        
        result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def reset(self):
        """Resetea el detector."""
        self.frame_buffer.clear()
        self.fall_history.clear()
        self.fall_count = 0
        self.last_state = "initializing"
        print("🔄 Detector reseteado")


class VideoVisualizer:
    """Visualizador de video con anotaciones."""
    
    def __init__(self, config):
        self.config = config
    
    def draw_skeleton(self, frame, landmarks, color=None):
        """Dibuja el esqueleto."""
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        color = color or self.config["color_skeleton"]
        
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        return frame
    
    def draw_status_panel(self, frame, result, fps):
        """Dibuja el panel de estado."""
        h, w = frame.shape[:2]
        
        # Determinar color y texto según estado
        state = result["state"]
        
        if state == "fall_confirmed":
            bg_color = (0, 0, 150)
            status_color = self.config["color_danger"]
            status_text = "🚨 CAIDA DETECTADA (LSTM)"
        elif state == "fall_possible":
            bg_color = (0, 100, 100)
            status_color = self.config["color_warning"]
            status_text = "⚠️ POSIBLE CAIDA"
        elif state == "buffering":
            bg_color = (100, 100, 0)
            status_color = (0, 255, 255)
            status_text = f"📊 BUFFERING {result['buffer_status']}"
        elif state == "normal":
            bg_color = (0, 100, 0)
            status_color = self.config["color_safe"]
            status_text = "✅ NORMAL"
        else:
            bg_color = (50, 50, 50)
            status_color = (128, 128, 128)
            status_text = "👤 SIN PERSONA"
        
        # Panel superior
        cv2.rectangle(frame, (0, 0), (w, 90), bg_color, -1)
        cv2.rectangle(frame, (0, 0), (w, 90), status_color, 3)
        
        # Título
        cv2.putText(frame, "SAFEGUARD VISION AI - LSTM", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Estado
        cv2.putText(frame, status_text, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Buffer status
        buffer_text = f"Buffer: {result['buffer_status']}"
        cv2.putText(frame, buffer_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Probabilidad
        prob_text = f"Prob: {result['probability']:.1%}"
        cv2.putText(frame, prob_text, (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 150, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Panel inferior
        panel_h = 80
        cv2.rectangle(frame, (0, h - panel_h), (280, h), (0, 0, 0), -1)
        
        info = [
            f"Frame: {result['frame_number']}",
            f"Latencia: {result['processing_time_ms']:.1f}ms",
            f"Caidas: {result['fall_count']}",
        ]
        
        for i, text in enumerate(info):
            cv2.putText(frame, text, (10, h - panel_h + 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alerta de caída
        if state == "fall_confirmed":
            thickness = 10 if (result["frame_number"] % 10 < 5) else 5
            cv2.rectangle(frame, (0, 0), (w-1, h-1), self.config["color_danger"], thickness)
            
            alert_text = "ALERTA: TRANSICION DE CAIDA DETECTADA"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            cv2.rectangle(frame, (text_x - 20, text_y - 50),
                         (text_x + text_size[0] + 20, text_y + 20), (0, 0, 150), -1)
            cv2.putText(frame, alert_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return frame


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                              MAIN
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SAFEGUARD VISION AI - DEMO LSTM ".center(70) + "║")
    print("║" + " Detección de TRANSICIONES de caída ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # Verificar modelo
    if not os.path.exists(MODEL_FOLDER):
        print(f"\n❌ ERROR: No se encontró: {MODEL_FOLDER}")
        print("   Primero ejecuta train_lstm_detector.py")
        return
    
    # Inicializar detector
    print("\n" + "="*60)
    print("🔧 INICIALIZANDO")
    print("="*60)
    
    try:
        detector = LSTMFallDetector(
            MODEL_FOLDER, 
            SEQUENCES_FOLDER,
            CONFIG["sequence_length"]
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Inicializar visualizador
    visualizer = VideoVisualizer(CONFIG)
    
    # Abrir video
    print("\n" + "="*60)
    print("📹 ABRIENDO VIDEO")
    print("="*60)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {VIDEO_SOURCE}")
        return
    
    # Propiedades
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   FPS: {orig_fps:.1f}")
    print(f"   Frames: {total_frames}")
    
    # Video de salida
    out = None
    if CONFIG["save_output"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            CONFIG["output_path"], fourcc, orig_fps,
            (CONFIG["display_width"], CONFIG["display_height"])
        )
    
    # Loop principal
    print("\n" + "="*60)
    print("🎬 PROCESANDO")
    print("="*60)
    print("\n   Controles: Q=Salir, P=Pausar, S=Screenshot, R=Reset")
    print(f"\n   ⚠️ Los primeros {CONFIG['sequence_length']} frames son para llenar el buffer")
    
    cv2.namedWindow("SafeGuard LSTM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SafeGuard LSTM", CONFIG["display_width"], CONFIG["display_height"])
    
    paused = False
    fps_counter = deque(maxlen=30)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                detector.reset()
                continue
        
        frame_start = time.perf_counter()
        
        # Redimensionar
        frame = cv2.resize(frame, (CONFIG["display_width"], CONFIG["display_height"]))
        
        # Procesar
        result = detector.process_frame(frame)
        
        # Dibujar
        skeleton_color = CONFIG["color_danger"] if result["state"] == "fall_confirmed" else CONFIG["color_skeleton"]
        frame = visualizer.draw_skeleton(frame, result["landmarks"], skeleton_color)
        
        # FPS
        frame_time = time.perf_counter() - frame_start
        fps_counter.append(1.0 / max(frame_time, 0.001))
        current_fps = np.mean(fps_counter)
        
        # Panel
        frame = visualizer.draw_status_panel(frame, result, current_fps)
        
        # Mostrar
        cv2.imshow("SafeGuard LSTM", frame)
        
        # Guardar
        if out is not None and not paused:
            out.write(frame)
        
        # Teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            cv2.imwrite(f"lstm_screenshot_{datetime.now().strftime('%H%M%S')}.png", frame)
        elif key == ord('r'):
            detector.reset()
    
    # Limpieza
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print("\n")
    print("╔" + "═"*60 + "╗")
    print("║" + " 📊 RESUMEN ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print(f"║  Frames procesados: {detector.frame_count}".ljust(61) + "║")
    print(f"║  Caídas detectadas: {detector.fall_count}".ljust(61) + "║")
    print("╚" + "═"*60 + "╝")
    print()


if __name__ == "__main__":
    main()
