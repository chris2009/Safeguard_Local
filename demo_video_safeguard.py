"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - DEMO EN TIEMPO REAL                             ║
║                                                                              ║
║   Detecta caídas en video usando BlazePose + Random Forest                   ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║   Fecha: Enero 2026                                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

DESCRIPCIÓN:
============
Este script procesa un video y detecta caídas en tiempo real usando:
    1. BlazePose (FULL) para extraer keypoints del cuerpo
    2. Random Forest para clasificar: ADL vs Caída

ENTRADA:
========
    - Video: archivo .mp4, .avi, etc. o webcam (0) o RTSP URL
    - Modelo: modelo_caidas.pkl (entrenado previamente)
    - Scaler: scaler_caidas.pkl

SALIDA:
=======
    - Ventana con video anotado en tiempo real
    - Video guardado con anotaciones (opcional)
    - Log de detecciones

CONTROLES:
==========
    - Q: Salir
    - P: Pausar/Reanudar
    - S: Guardar screenshot
    - R: Resetear contador de caídas

USO:
====
    python demo_video_safeguard.py

REQUISITOS:
===========
    pip install mediapipe opencv-python pandas numpy joblib
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from collections import deque
import joblib
import warnings
warnings.filterwarnings('ignore')

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURACIÓN                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════════════════════════
# FUENTE DE VIDEO
# ═══════════════════════════════════════════════════════════════════════════════
# Opciones:
#   - Archivo de video: r"C:\ruta\al\video.mp4"
#   - Webcam: 0 (default), 1 (segunda cámara)
#   - RTSP: "rtsp://usuario:password@192.168.1.100:554/stream"

VIDEO_SOURCE = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\video_prueba_7.mp4"  # ← CAMBIA ESTO

# ═══════════════════════════════════════════════════════════════════════════════
# MODELO ENTRENADO
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\train_model_V2"  # ← CAMBIA ESTO

# ═══════════════════════════════════════════════════════════════════════════════
# BLAZEPOSE
# ═══════════════════════════════════════════════════════════════════════════════

# Modelo de BlazePose a usar (LITE, FULL, HEAVY)
BLAZEPOSE_MODEL = "HEAVY"

BLAZEPOSE_URLS = {
    "LITE": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "FULL": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "HEAVY": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Umbral de probabilidad para detectar caída
    "fall_threshold": 0.5,
    
    # Frames consecutivos con caída para confirmar alerta
    "confirmation_frames": 3,
    
    # Tamaño de ventana de visualización (None = tamaño original)
    "display_width": 1280,
    "display_height": 720,
    
    # Guardar video de salida
    "save_output": True,
    "output_path": r"G:\Mi unidad\safeguard_output.mp4",
    
    # Mostrar información de debug
    "show_debug": True,
    
    # Colores (BGR)
    "color_safe": (0, 255, 0),       # Verde - Normal
    "color_warning": (0, 255, 255),  # Amarillo - Posible caída
    "color_danger": (0, 0, 255),     # Rojo - Caída confirmada
    "color_skeleton": (255, 128, 0), # Naranja - Esqueleto
    "color_keypoints": (0, 255, 0),  # Verde - Puntos
}

# ═══════════════════════════════════════════════════════════════════════════════
# NOMBRES DE KEYPOINTS (33 puntos de BlazePose)
# ═══════════════════════════════════════════════════════════════════════════════

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

# Conexiones del esqueleto para dibujar
POSE_CONNECTIONS = [
    # Torso
    (11, 12),  # Hombros
    (11, 23), (12, 24),  # Hombros a caderas
    (23, 24),  # Caderas
    # Brazo izquierdo
    (11, 13), (13, 15),
    # Brazo derecho
    (12, 14), (14, 16),
    # Pierna izquierda
    (23, 25), (25, 27), (27, 29), (29, 31),
    # Pierna derecha
    (24, 26), (26, 28), (28, 30), (30, 32),
    # Cara
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         CLASES Y FUNCIONES                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class FallDetector:
    """
    Clase principal para detección de caídas.
    Combina BlazePose + Random Forest en un pipeline unificado.
    """
    
    def __init__(self, model_folder, blazepose_model="FULL"):
        """
        Inicializa el detector de caídas.
        
        Args:
            model_folder: Carpeta con modelo_caidas.pkl y scaler_caidas.pkl
            blazepose_model: "LITE", "FULL", o "HEAVY"
        """
        self.model_folder = model_folder
        self.blazepose_model = blazepose_model
        
        # Cargar modelo de clasificación
        self._load_classifier()
        
        # Inicializar BlazePose
        self._init_blazepose()
        
        # Estado
        self.fall_history = deque(maxlen=CONFIG['confirmation_frames'])
        self.last_state = "normal"
        self.fall_count = 0
        self.frame_count = 0
        
    def _load_classifier(self):
        """Carga el modelo Random Forest y el scaler."""
        print("📂 Cargando modelo de clasificación...")
        
        model_path = os.path.join(self.model_folder, "modelo_caidas_v2.pkl")
        scaler_path = os.path.join(self.model_folder, "scaler_caidas_v2.pkl")
        features_path = os.path.join(self.model_folder, "feature_names_v2.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"   ✅ Modelo cargado: Random Forest")
        print(f"   ✅ Features: {len(self.feature_names)}")
        
    def _init_blazepose(self):
        """Inicializa el detector de pose BlazePose."""
        print(f"🔧 Inicializando BlazePose ({self.blazepose_model})...")
        
        # Descargar modelo si no existe
        model_filename = f"pose_landmarker_{self.blazepose_model.lower()}.task"
        
        if not os.path.exists(model_filename):
            import urllib.request
            print(f"   📥 Descargando modelo {self.blazepose_model}...")
            url = BLAZEPOSE_URLS[self.blazepose_model]
            urllib.request.urlretrieve(url, model_filename)
        
        # Configurar detector
        base_options = python.BaseOptions(model_asset_path=model_filename)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)
        print(f"   ✅ BlazePose {self.blazepose_model} inicializado")
        
    def extract_keypoints(self, frame):
        """
        Extrae keypoints del frame usando BlazePose.
        
        Args:
            frame: Frame de video (BGR)
            
        Returns:
            keypoints: Dict con coordenadas o None si no detecta persona
            landmarks: Lista de landmarks para dibujar
        """
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crear imagen de MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detectar pose
        detection_result = self.pose_detector.detect(mp_image)
        
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return None, None
        
        landmarks = detection_result.pose_landmarks[0]
        
        # Extraer keypoints como dict
        keypoints = {}
        for idx, landmark in enumerate(landmarks):
            name = KEYPOINT_NAMES[idx]
            keypoints[f"{name}_x"] = landmark.x
            keypoints[f"{name}_y"] = landmark.y
            keypoints[f"{name}_z"] = landmark.z
            keypoints[f"{name}_vis"] = landmark.visibility
        
        return keypoints, landmarks
    
    def create_features(self, keypoints):
        """
        Crea el vector de features incluyendo features derivadas.
        
        Args:
            keypoints: Dict con coordenadas de keypoints
            
        Returns:
            features: Array numpy con todas las features
        """
        # Features originales (132)
        original_features = []
        for name in KEYPOINT_NAMES:
            original_features.extend([
                keypoints.get(f"{name}_x", 0),
                keypoints.get(f"{name}_y", 0),
                keypoints.get(f"{name}_z", 0),
                keypoints.get(f"{name}_vis", 0)
            ])
        
        # Features derivadas (las mismas que en entrenamiento)
        # 1. Ángulo del torso
        torso_angle = keypoints['nose_y'] - (keypoints['left_hip_y'] + keypoints['right_hip_y']) / 2
        
        # 2. Altura del cuerpo
        y_values = [keypoints[f"{name}_y"] for name in KEYPOINT_NAMES]
        body_height = max(y_values) - min(y_values)
        
        # 3. Ancho del cuerpo
        x_values = [keypoints[f"{name}_x"] for name in KEYPOINT_NAMES]
        body_width = max(x_values) - min(x_values)
        
        # 4. Ratio altura/ancho
        aspect_ratio = body_height / (body_width + 0.001)
        
        # 5. Centro de masa Y
        center_y = (keypoints['nose_y'] + keypoints['left_hip_y'] + 
                    keypoints['right_hip_y'] + keypoints['left_shoulder_y'] + 
                    keypoints['right_shoulder_y']) / 5
        
        # 6. Distancia hombros
        shoulder_dist = np.sqrt((keypoints['left_shoulder_x'] - keypoints['right_shoulder_x'])**2 +
                               (keypoints['left_shoulder_y'] - keypoints['right_shoulder_y'])**2)
        
        # 7. Distancia caderas
        hip_dist = np.sqrt((keypoints['left_hip_x'] - keypoints['right_hip_x'])**2 +
                           (keypoints['left_hip_y'] - keypoints['right_hip_y'])**2)
        
        # 8. Ángulo de piernas
        leg_angle = ((keypoints['left_hip_y'] + keypoints['right_hip_y']) / 2 - 
                     (keypoints['left_ankle_y'] + keypoints['right_ankle_y']) / 2)
        
        # Combinar todas las features
        derived_features = [torso_angle, body_height, body_width, aspect_ratio,
                          center_y, shoulder_dist, hip_dist, leg_angle]
        
        all_features = original_features + derived_features
        
        return np.array(all_features).reshape(1, -1)
    
    def predict(self, features):
        """
        Predice si hay caída usando el modelo.
        
        Args:
            features: Array de features
            
        Returns:
            prediction: 0=ADL, 1=Fall
            probability: Probabilidad de caída (0-1)
        """
        # Normalizar
        features_scaled = self.scaler.transform(features)
        
        # Predecir
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        return prediction, probability
    
    def process_frame(self, frame):
        """
        Procesa un frame completo: extrae pose, clasifica, actualiza estado.
        
        Args:
            frame: Frame de video (BGR)
            
        Returns:
            result: Dict con toda la información del frame
        """
        self.frame_count += 1
        start_time = time.perf_counter()
        
        result = {
            "frame_number": self.frame_count,
            "person_detected": False,
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
            # No se detectó persona
            self.fall_history.append(0)
            result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
            return result
        
        result["person_detected"] = True
        result["landmarks"] = landmarks
        
        # Crear features y predecir
        features = self.create_features(keypoints)
        prediction, probability = self.predict(features)
        
        result["prediction"] = prediction
        result["probability"] = probability
        
        # Actualizar historial
        self.fall_history.append(prediction)
        
        # Determinar estado basado en frames consecutivos
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
        """Resetea el contador de caídas y el historial."""
        self.fall_history.clear()
        self.fall_count = 0
        self.last_state = "normal"
        print("🔄 Detector reseteado")


class VideoVisualizer:
    """
    Clase para visualizar el video con anotaciones.
    """
    
    def __init__(self, config):
        self.config = config
        
    def draw_skeleton(self, frame, landmarks, color=None):
        """
        Dibuja el esqueleto sobre el frame.
        
        Args:
            frame: Frame de video
            landmarks: Lista de landmarks de BlazePose
            color: Color del esqueleto (BGR)
        """
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        
        if color is None:
            color = self.config["color_skeleton"]
        
        # Dibujar conexiones
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar keypoints
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, self.config["color_keypoints"], -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
        
        return frame
    
    def draw_status_panel(self, frame, result, fps):
        """
        Dibuja el panel de estado en el frame.
        
        Args:
            frame: Frame de video
            result: Resultado del detector
            fps: FPS actual
        """
        h, w = frame.shape[:2]
        
        # Determinar color según estado
        if result["state"] == "fall_confirmed":
            status_color = self.config["color_danger"]
            status_text = "🚨 CAÍDA DETECTADA"
            bg_color = (0, 0, 100)
        elif result["state"] == "fall_possible":
            status_color = self.config["color_warning"]
            status_text = "⚠️ POSIBLE CAÍDA"
            bg_color = (0, 100, 100)
        elif result["state"] == "normal":
            status_color = self.config["color_safe"]
            status_text = "✅ NORMAL"
            bg_color = (0, 100, 0)
        else:
            status_color = (128, 128, 128)
            status_text = "👤 SIN PERSONA"
            bg_color = (50, 50, 50)
        
        # Panel superior
        cv2.rectangle(frame, (0, 0), (w, 80), bg_color, -1)
        cv2.rectangle(frame, (0, 0), (w, 80), status_color, 3)
        
        # Título
        cv2.putText(frame, "SAFEGUARD VISION AI", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Estado
        cv2.putText(frame, status_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Probabilidad
        prob_text = f"Prob: {result['probability']:.1%}"
        cv2.putText(frame, prob_text, (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 150, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Panel inferior con estadísticas
        if self.config["show_debug"]:
            panel_h = 100
            cv2.rectangle(frame, (0, h - panel_h), (300, h), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, h - panel_h), (300, h), (100, 100, 100), 1)
            
            y_offset = h - panel_h + 20
            debug_info = [
                f"Frame: {result['frame_number']}",
                f"Latencia: {result['processing_time_ms']:.1f}ms",
                f"Caidas totales: {result['fall_count']}",
                f"Persona: {'Si' if result['person_detected'] else 'No'}"
            ]
            
            for i, text in enumerate(debug_info):
                cv2.putText(frame, text, (10, y_offset + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alerta de caída con borde parpadeante
        if result["state"] == "fall_confirmed":
            # Borde rojo parpadeante
            thickness = 10 if (result["frame_number"] % 10 < 5) else 5
            cv2.rectangle(frame, (0, 0), (w-1, h-1), self.config["color_danger"], thickness)
            
            # Texto grande de alerta
            alert_text = "ALERTA: CAIDA DETECTADA"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            # Fondo del texto
            cv2.rectangle(frame, (text_x - 20, text_y - 50), 
                         (text_x + text_size[0] + 20, text_y + 20), (0, 0, 150), -1)
            cv2.putText(frame, alert_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        return frame
    
    def draw_controls_help(self, frame):
        """Dibuja ayuda de controles."""
        h, w = frame.shape[:2]
        
        controls = [
            "Q: Salir",
            "P: Pausar",
            "S: Screenshot",
            "R: Reset"
        ]
        
        cv2.rectangle(frame, (w - 120, h - 100), (w, h), (0, 0, 0), -1)
        
        for i, text in enumerate(controls):
            cv2.putText(frame, text, (w - 110, h - 80 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame


def download_sample_video():
    """Descarga un video de prueba si no hay uno disponible."""
    # Crear un video de prueba simple si no existe
    print("⚠️ No se encontró video de prueba")
    print("   Usa tu propio video o webcam (cambia VIDEO_SOURCE = 0)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """Función principal de la demo."""
    
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SAFEGUARD VISION AI - DEMO EN TIEMPO REAL ".center(70) + "║")
    print("║" + " Detección de Caídas con BlazePose + Random Forest ".center(70) + "║")
    print("║" + " MIT Global Teaching Labs ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INICIALIZACIÓN
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Verificar modelo
    if not os.path.exists(MODEL_FOLDER):
        print(f"\n❌ ERROR: No se encontró la carpeta del modelo:")
        print(f"   {MODEL_FOLDER}")
        print("\n   Primero ejecuta train_fall_detector.py")
        return
    
    # Inicializar detector
    print("\n" + "="*60)
    print("🔧 INICIALIZANDO COMPONENTES")
    print("="*60)
    
    try:
        detector = FallDetector(MODEL_FOLDER, BLAZEPOSE_MODEL)
    except Exception as e:
        print(f"\n❌ Error inicializando detector: {e}")
        return
    
    # Inicializar visualizador
    visualizer = VideoVisualizer(CONFIG)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ABRIR VIDEO
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*60)
    print("📹 ABRIENDO FUENTE DE VIDEO")
    print("="*60)
    
    # Determinar tipo de fuente
    if isinstance(VIDEO_SOURCE, int):
        source_type = "webcam"
        print(f"   Fuente: Webcam {VIDEO_SOURCE}")
    elif VIDEO_SOURCE.startswith("rtsp://"):
        source_type = "rtsp"
        print(f"   Fuente: RTSP Stream")
    else:
        source_type = "file"
        print(f"   Fuente: Archivo de video")
        print(f"   Ruta: {VIDEO_SOURCE}")
        
        if not os.path.exists(VIDEO_SOURCE):
            print(f"\n❌ ERROR: No se encontró el video:")
            print(f"   {VIDEO_SOURCE}")
            download_sample_video()
            return
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"\n❌ ERROR: No se pudo abrir la fuente de video")
        return
    
    # Obtener propiedades del video
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Resolución: {orig_width}x{orig_height}")
    print(f"   FPS original: {orig_fps:.1f}")
    if source_type == "file":
        print(f"   Total frames: {total_frames}")
    
    # Configurar tamaño de display
    display_width = CONFIG["display_width"] or orig_width
    display_height = CONFIG["display_height"] or orig_height
    
    # Configurar video de salida
    out = None
    if CONFIG["save_output"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(CONFIG["output_path"], fourcc, orig_fps, 
                             (display_width, display_height))
        print(f"   💾 Guardando en: {CONFIG['output_path']}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOOP PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*60)
    print("🎬 INICIANDO PROCESAMIENTO")
    print("="*60)
    print("\n   Controles:")
    print("   - Q: Salir")
    print("   - P: Pausar/Reanudar")
    print("   - S: Guardar screenshot")
    print("   - R: Resetear contador de caídas")
    print("\n   Presiona cualquier tecla para comenzar...")
    
    cv2.namedWindow("SafeGuard Vision AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SafeGuard Vision AI", display_width, display_height)
    
    paused = False
    fps_counter = deque(maxlen=30)
    
    while True:
        # Manejar pausa
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                if source_type == "file":
                    print("\n📼 Fin del video")
                    # Reiniciar video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("\n❌ Error leyendo frame")
                    break
        
        frame_start = time.perf_counter()
        
        # Redimensionar si es necesario
        if frame.shape[1] != display_width or frame.shape[0] != display_height:
            frame = cv2.resize(frame, (display_width, display_height))
        
        # Procesar frame
        result = detector.process_frame(frame)
        
        # Dibujar esqueleto
        skeleton_color = CONFIG["color_danger"] if result["state"] == "fall_confirmed" else CONFIG["color_skeleton"]
        frame = visualizer.draw_skeleton(frame, result["landmarks"], skeleton_color)
        
        # Calcular FPS
        frame_time = time.perf_counter() - frame_start
        fps_counter.append(1.0 / max(frame_time, 0.001))
        current_fps = np.mean(fps_counter)
        
        # Dibujar panel de estado
        frame = visualizer.draw_status_panel(frame, result, current_fps)
        frame = visualizer.draw_controls_help(frame)
        
        # Mostrar frame
        cv2.imshow("SafeGuard Vision AI", frame)
        
        # Guardar frame si está habilitado
        if out is not None and not paused:
            out.write(frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n👋 Saliendo...")
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print(f"   {'⏸️ Pausado' if paused else '▶️ Reanudado'}")
        elif key == ord('s') or key == ord('S'):
            screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(screenshot_path, frame)
            print(f"   📸 Screenshot guardado: {screenshot_path}")
        elif key == ord('r') or key == ord('R'):
            detector.reset()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIMPIEZA
    # ═══════════════════════════════════════════════════════════════════════════
    
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Resumen final
    print("\n")
    print("╔" + "═"*60 + "╗")
    print("║" + " 📊 RESUMEN DE LA SESIÓN ".center(60) + "║")
    print("╠" + "═"*60 + "╣")
    print(f"║  Frames procesados:  {detector.frame_count:>10}".ljust(61) + "║")
    print(f"║  Caídas detectadas:  {detector.fall_count:>10}".ljust(61) + "║")
    print("╚" + "═"*60 + "╝")
    
    if CONFIG["save_output"]:
        print(f"\n💾 Video guardado: {CONFIG['output_path']}")
    
    print("\n✅ Demo finalizada\n")


if __name__ == "__main__":
    main()
