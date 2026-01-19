"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - DEMO TRANSFORMER                                ║
║                                                                              ║
║   Detecta caídas usando mecanismo de ATENCIÓN sobre secuencias              ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

CONTROLES:
==========
    Q: Salir
    P: Pausar/Reanudar
    S: Screenshot
    R: Reset contador de caídas
    ESPACIO: Avanzar 1 frame (cuando está pausado)
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
from tensorflow import keras
from tensorflow.keras import layers

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CAPAS PERSONALIZADAS (necesarias para cargar modelo)
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PositionalEncoding(layers.Layer):
    """Añade información de posición temporal a la secuencia."""
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pe = np.zeros((sequence_length, embed_dim))
        pe[:, 0::2] = np.sin(position * div_term[:embed_dim//2 + embed_dim%2])
        pe[:, 1::2] = np.cos(position * div_term[:embed_dim//2])
        
        self.positional_encoding = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        return x + self.positional_encoding[:tf.shape(x)[1], :tf.shape(x)[2]]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim
        })
        return config


class TransformerBlock(layers.Layer):
    """Un bloque del Transformer Encoder."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads
        )
        
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CONFIGURACIÓN
# ╚══════════════════════════════════════════════════════════════════════════════╝

# FUENTE DE VIDEO
VIDEO_SOURCE = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\video_prueba_9.mp4"

# MODELO TRANSFORMER
MODEL_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_model_transformer"
SEQUENCES_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_sequences"

# PARÁMETROS
CONFIG = {
    "sequence_length": 30,
    "fall_threshold": 0.5,
    "confirmation_frames": 2,
    
    "display_width": 1280,
    "display_height": 720,
    "save_output": True,
    "output_path": r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_transformer_output.mp4",
    
    # Colores (BGR) - Púrpura para Transformer
    "color_safe": (0, 255, 0),
    "color_warning": (0, 255, 255),
    "color_danger": (0, 0, 255),
    "color_skeleton": (255, 0, 128),  # Púrpura/Magenta para Transformer
}

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         DETECTOR
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TransformerFallDetector:
    """
    Detector de caídas basado en Transformer.
    Usa mecanismo de atención para analizar secuencias.
    """
    
    def __init__(self, model_folder, sequences_folder, sequence_length=30):
        self.model_folder = model_folder
        self.sequences_folder = sequences_folder
        self.sequence_length = sequence_length
        
        self.frame_buffer = deque(maxlen=sequence_length)
        self.fall_history = deque(maxlen=CONFIG['confirmation_frames'])
        self.last_state = "initializing"
        self.fall_count = 0
        self.frame_count = 0
        
        self._load_model()
        self._init_blazepose()
        self._load_normalization()
        
    def _load_model(self):
        """Carga el modelo Transformer."""
        print("📂 Cargando modelo Transformer...")
        
        # Registrar capas personalizadas
        custom_objects = {
            'PositionalEncoding': PositionalEncoding,
            'TransformerBlock': TransformerBlock
        }
        
        possible_names = [
            "modelo_transformer.h5",
            "best_model.h5",
            "modelo_transformer.keras",
            "best_model.keras"
        ]
        
        model_path = None
        for name in possible_names:
            path = os.path.join(self.model_folder, name)
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No se encontró modelo en: {self.model_folder}")
        
        print(f"   📄 Archivo: {os.path.basename(model_path)}")
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"   ✅ Modelo Transformer cargado")
        
    def _init_blazepose(self):
        """Inicializa BlazePose."""
        print("🔧 Inicializando BlazePose...")
        
        model_filename = "pose_landmarker_full.task"
        
        if not os.path.exists(model_filename):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            print(f"   ⬇️  Descargando modelo BlazePose...")
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
            print("   ⚠️  No se encontraron parámetros de normalización")
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
        
        features = []
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return np.array(features, dtype=np.float32), landmarks
    
    def add_temporal_features(self, sequence):
        """Añade velocidad y aceleración a la secuencia."""
        velocities = np.zeros_like(sequence)
        velocities[1:] = sequence[1:] - sequence[:-1]
        
        accelerations = np.zeros_like(sequence)
        accelerations[1:] = velocities[1:] - velocities[:-1]
        
        enhanced = np.concatenate([sequence, velocities, accelerations], axis=1)
        
        return enhanced
    
    def predict_sequence(self):
        """Predice si la secuencia actual contiene una caída."""
        if len(self.frame_buffer) < self.sequence_length:
            return 0, 0.0, "buffering"
        
        sequence = np.array(list(self.frame_buffer))
        sequence = self.add_temporal_features(sequence)
        
        if self.norm_mean is not None:
            sequence = (sequence - self.norm_mean) / (self.norm_std + 1e-8)
        
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        probability = self.model.predict(sequence, verbose=0)[0][0]
        
        prediction = 1 if probability >= CONFIG['fall_threshold'] else 0
        
        return prediction, probability, "ready"
    
    def process_frame(self, frame):
        """Procesa un frame."""
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
        
        keypoints, landmarks = self.extract_keypoints(frame)
        
        if keypoints is None:
            result["state"] = "no_person"
            result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
            return result
        
        result["person_detected"] = True
        result["landmarks"] = landmarks
        
        self.frame_buffer.append(keypoints)
        result["buffer_status"] = f"{len(self.frame_buffer)}/{self.sequence_length}"
        
        prediction, probability, status = self.predict_sequence()
        
        result["prediction"] = prediction
        result["probability"] = probability
        
        if status == "buffering":
            result["state"] = "buffering"
        else:
            self.fall_history.append(prediction)
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         VISUALIZADOR
# ╚══════════════════════════════════════════════════════════════════════════════╝

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
    
    def draw_status_panel(self, frame, result, fps, paused=False):
        """Dibuja el panel de estado."""
        h, w = frame.shape[:2]
        
        state = result["state"]
        
        if state == "fall_confirmed":
            bg_color = (128, 0, 128)  # Púrpura oscuro
            status_color = self.config["color_danger"]
            status_text = "CAIDA DETECTADA (TRANSFORMER)"
        elif state == "fall_possible":
            bg_color = (100, 50, 100)
            status_color = self.config["color_warning"]
            status_text = "POSIBLE CAIDA"
        elif state == "buffering":
            bg_color = (80, 0, 80)
            status_color = (255, 0, 255)
            status_text = f"BUFFERING {result['buffer_status']}"
        elif state == "normal":
            bg_color = (0, 100, 0)
            status_color = self.config["color_safe"]
            status_text = "NORMAL"
        else:
            bg_color = (50, 50, 50)
            status_color = (128, 128, 128)
            status_text = "SIN PERSONA"
        
        # Panel superior
        cv2.rectangle(frame, (0, 0), (w, 90), bg_color, -1)
        cv2.rectangle(frame, (0, 0), (w, 90), status_color, 3)
        
        # Título
        cv2.putText(frame, "SAFEGUARD VISION AI - TRANSFORMER", (10, 25),
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
        
        # Indicador de PAUSA
        if paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2 - 100, 100), (w//2 + 100, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "PAUSADO", (w//2 - 70, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
            
            cv2.putText(frame, "P: Reanudar | ESPACIO: Avanzar 1 frame | S: Screenshot", 
                       (w//2 - 280, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Panel inferior
        panel_h = 100
        cv2.rectangle(frame, (0, h - panel_h), (300, h), (0, 0, 0), -1)
        
        info = [
            f"Frame: {result['frame_number']}",
            f"Latencia: {result['processing_time_ms']:.1f}ms",
            f"Caidas totales: {result['fall_count']}",
            f"Persona: {'Si' if result['person_detected'] else 'No'}",
        ]
        
        for i, text in enumerate(info):
            cv2.putText(frame, text, (10, h - panel_h + 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Panel de controles
        controls_x = w - 180
        cv2.rectangle(frame, (controls_x - 10, h - panel_h), (w, h), (0, 0, 0), -1)
        
        controls = [
            "Q: Salir",
            "P: Pausar",
            "S: Screenshot",
            "R: Reset",
        ]
        
        for i, text in enumerate(controls):
            cv2.putText(frame, text, (controls_x, h - panel_h + 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Alerta de caída
        if state == "fall_confirmed" and not paused:
            thickness = 10 if (result["frame_number"] % 10 < 5) else 5
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (255, 0, 255), thickness)  # Púrpura
            
            alert_text = "ALERTA: CAIDA DETECTADA (ATTENTION)"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            cv2.rectangle(frame, (text_x - 20, text_y - 50),
                         (text_x + text_size[0] + 20, text_y + 20), (128, 0, 128), -1)
            cv2.putText(frame, alert_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return frame


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                              MAIN
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SAFEGUARD VISION AI - DEMO TRANSFORMER ".center(70) + "║")
    print("║" + " Detección con Mecanismo de Atención ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # Verificar modelo
    if not os.path.exists(MODEL_FOLDER):
        print(f"\n❌ ERROR: No se encontró: {MODEL_FOLDER}")
        print("   Primero ejecuta train_transformer_detector.py")
        return
    
    # Inicializar
    print("\n" + "="*60)
    print("🔧 INICIALIZANDO")
    print("="*60)
    
    try:
        detector = TransformerFallDetector(
            MODEL_FOLDER, 
            SEQUENCES_FOLDER,
            CONFIG["sequence_length"]
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    visualizer = VideoVisualizer(CONFIG)
    
    # Abrir video
    print("\n" + "="*60)
    print("📹 ABRIENDO VIDEO")
    print("="*60)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {VIDEO_SOURCE}")
        return
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   FPS original: {orig_fps:.1f}")
    print(f"   Total frames: {total_frames}")
    
    # Video de salida
    out = None
    if CONFIG["save_output"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            CONFIG["output_path"], fourcc, orig_fps,
            (CONFIG["display_width"], CONFIG["display_height"])
        )
    
    # Loop
    print("\n" + "="*60)
    print("🎬 PROCESANDO")
    print("="*60)
    print("\n   Controles:")
    print("   ├── Q: Salir")
    print("   ├── P: Pausar/Reanudar")
    print("   ├── S: Screenshot")
    print("   ├── R: Reset contador")
    print("   └── ESPACIO: Avanzar 1 frame (en pausa)")
    
    cv2.namedWindow("SafeGuard Transformer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SafeGuard Transformer", CONFIG["display_width"], CONFIG["display_height"])
    
    paused = False
    fps_counter = deque(maxlen=30)
    
    current_frame = None
    current_result = None
    display_frame = None
    current_fps = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                detector.reset()
                continue
            
            frame_start = time.perf_counter()
            
            current_frame = cv2.resize(frame, (CONFIG["display_width"], CONFIG["display_height"]))
            current_result = detector.process_frame(current_frame)
            
            display_frame = current_frame.copy()
            
            skeleton_color = (255, 0, 255) if current_result["state"] == "fall_confirmed" else CONFIG["color_skeleton"]
            display_frame = visualizer.draw_skeleton(display_frame, current_result["landmarks"], skeleton_color)
            
            frame_time = time.perf_counter() - frame_start
            fps_counter.append(1.0 / max(frame_time, 0.001))
            current_fps = np.mean(fps_counter)
            
            if out is not None:
                save_frame = display_frame.copy()
                save_frame = visualizer.draw_status_panel(save_frame, current_result, current_fps, paused=False)
                out.write(save_frame)
        
        if display_frame is not None and current_result is not None:
            show_frame = display_frame.copy()
            show_frame = visualizer.draw_status_panel(show_frame, current_result, current_fps, paused=paused)
            cv2.imshow("SafeGuard Transformer", show_frame)
        
        wait_time = 1 if not paused else 50
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            print("\n   🛑 Saliendo...")
            break
            
        elif key == ord('p'):
            paused = not paused
            if paused:
                print(f"\n   ⏸️  PAUSADO en frame {current_result['frame_number'] if current_result else 0}")
            else:
                print("   ▶️  REANUDADO")
                
        elif key == ord('s'):
            if display_frame is not None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = f"transformer_screenshot_{timestamp}.png"
                
                screenshot_frame = display_frame.copy()
                if current_result:
                    screenshot_frame = visualizer.draw_status_panel(screenshot_frame, current_result, current_fps, paused=False)
                
                cv2.imwrite(screenshot_path, screenshot_frame)
                print(f"   📸 Screenshot guardado: {screenshot_path}")
                
        elif key == ord('r'):
            detector.reset()
            print("   🔄 Contador de caídas reseteado")
            
        elif key == ord(' ') and paused:
            ret, frame = cap.read()
            if ret:
                current_frame = cv2.resize(frame, (CONFIG["display_width"], CONFIG["display_height"]))
                current_result = detector.process_frame(current_frame)
                
                display_frame = current_frame.copy()
                skeleton_color = (255, 0, 255) if current_result["state"] == "fall_confirmed" else CONFIG["color_skeleton"]
                display_frame = visualizer.draw_skeleton(display_frame, current_result["landmarks"], skeleton_color)
                
                print(f"   ⏩ Frame {current_result['frame_number']} | Estado: {current_result['state']} | Prob: {current_result['probability']:.2%}")
    
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
