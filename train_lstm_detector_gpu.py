"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - ENTRENAMIENTO LSTM (GPU)                        ║
║                                                                              ║
║   Versión con soporte para GPU usando DirectML                              ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUISITOS GPU:
===============
    pip install tensorflow-directml
    
    DirectML funciona con:
    - NVIDIA (tu RTX 4070)
    - AMD
    - Intel

USO:
====
    python train_lstm_detector_gpu.py
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CONFIGURACIÓN DE GPU
# ╚══════════════════════════════════════════════════════════════════════════════╝

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Mostrar info de GPU

import tensorflow as tf

def setup_gpu():
    """
    Configura y detecta GPU disponible.
    """
    print("\n" + "="*70)
    print("🔧 CONFIGURACIÓN DE GPU")
    print("="*70)
    
    print(f"\n   TensorFlow version: {tf.__version__}")
    
    # Listar todos los dispositivos
    all_devices = tf.config.list_physical_devices()
    print(f"\n   📋 Dispositivos detectados:")
    for device in all_devices:
        print(f"      • {device.device_type}: {device.name}")
    
    # Buscar GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    # También buscar dispositivos DML (DirectML)
    dml_devices = [d for d in all_devices if 'DML' in d.name.upper()]
    
    gpu_available = len(gpus) > 0 or len(dml_devices) > 0
    
    if gpus:
        print(f"\n   ✅ GPU NVIDIA/AMD detectada: {len(gpus)}")
        for gpu in gpus:
            print(f"      • {gpu.name}")
        
        # Configurar crecimiento de memoria
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("   ✅ Memory growth habilitado")
        except RuntimeError as e:
            print(f"   ⚠️  No se pudo configurar memory growth: {e}")
            
    elif dml_devices:
        print(f"\n   ✅ DirectML GPU detectada: {len(dml_devices)}")
        for dml in dml_devices:
            print(f"      • {dml.name}")
    else:
        print("\n   ⚠️  No se detectó GPU")
        print("   📋 Usando CPU (será más lento)")
        print("\n   💡 Para habilitar GPU:")
        print("      pip uninstall tensorflow tensorflow-intel -y")
        print("      pip install tensorflow-directml")
    
    return gpu_available


# Ejecutar configuración de GPU al inicio
GPU_AVAILABLE = setup_gpu()


# Importar Keras después de configurar GPU
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CONFIGURACIÓN
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Carpeta con las secuencias (creada por create_sequences.py)
SEQUENCES_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_sequences"

# Carpeta de salida para el modelo
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_model_lstm_GPU"

# ═══════════════════════════════════════════════════════════════════════════════
# HIPERPARÁMETROS DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    # Arquitectura
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dense_units": 32,
    "dropout_rate": 0.3,
    "bidirectional": True,
    
    # Entrenamiento
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 5,
    
    # Pesos de clase
    "use_class_weights": True,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CALLBACK PERSONALIZADO
# ╚══════════════════════════════════════════════════════════════════════════════╝

class GPUMonitorCallback(Callback):
    """
    Callback para monitorear tiempo de entrenamiento y mostrar progreso.
    """
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.start_time = None
        self.total_start = None
        
    def on_train_begin(self, logs=None):
        self.total_start = time.time()
        print(f"\n   🚀 Entrenamiento iniciado {'con GPU' if GPU_AVAILABLE else 'con CPU'}...")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        # Mostrar cada 5 épocas o en las primeras
        if epoch < 3 or (epoch + 1) % 5 == 0:
            avg_time = np.mean(self.epoch_times)
            print(f"      Epoch {epoch+1}: {epoch_time:.2f}s (promedio: {avg_time:.2f}s)")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.total_start
        print(f"\n   ⏱️  Tiempo total de entrenamiento: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   📊 Promedio por época: {np.mean(self.epoch_times):.2f}s")


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         FUNCIONES
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_sequences(folder):
    """
    Carga las secuencias preparadas.
    """
    print("\n" + "="*70)
    print("📂 CARGANDO SECUENCIAS")
    print("="*70)
    
    X_train = np.load(os.path.join(folder, "X_train.npy"))
    X_test = np.load(os.path.join(folder, "X_test.npy"))
    y_train = np.load(os.path.join(folder, "y_train.npy"))
    y_test = np.load(os.path.join(folder, "y_test.npy"))
    
    print(f"\n   📊 Datos cargados:")
    print(f"      X_train: {X_train.shape}")
    print(f"      X_test:  {X_test.shape}")
    print(f"      y_train: {y_train.shape} ({sum(y_train)} caídas)")
    print(f"      y_test:  {y_test.shape} ({sum(y_test)} caídas)")
    
    # Normalizar
    print("\n   ⚖️  Normalizando datos...")
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # Guardar estadísticas de normalización
    np.save(os.path.join(folder, "norm_mean.npy"), mean)
    np.save(os.path.join(folder, "norm_std.npy"), std)
    
    print("   ✅ Datos normalizados")
    
    return X_train, X_test, y_train, y_test


def calculate_class_weights(y_train):
    """
    Calcula pesos de clase para manejar desbalance.
    """
    n_samples = len(y_train)
    n_positive = sum(y_train)
    n_negative = n_samples - n_positive
    
    weight_positive = n_samples / (2 * n_positive) if n_positive > 0 else 1
    weight_negative = n_samples / (2 * n_negative) if n_negative > 0 else 1
    
    class_weights = {0: weight_negative, 1: weight_positive}
    
    print(f"\n   ⚖️  Pesos de clase:")
    print(f"      Clase 0 (ADL):   {weight_negative:.3f}")
    print(f"      Clase 1 (Caída): {weight_positive:.3f}")
    
    return class_weights


def build_lstm_model(input_shape, config):
    """
    Construye el modelo LSTM optimizado para GPU.
    """
    print("\n" + "="*70)
    print("🏗️  CONSTRUYENDO MODELO LSTM")
    print("="*70)
    
    # Usar CuDNNLSTM si hay GPU NVIDIA (más rápido)
    # Para DirectML usamos LSTM normal
    
    model = Sequential()
    
    # Primera capa LSTM
    if config["bidirectional"]:
        model.add(Bidirectional(
            LSTM(config["lstm_units_1"], return_sequences=True),
            input_shape=input_shape
        ))
    else:
        model.add(LSTM(
            config["lstm_units_1"], 
            return_sequences=True,
            input_shape=input_shape
        ))
    
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout_rate"]))
    
    # Segunda capa LSTM
    if config["bidirectional"]:
        model.add(Bidirectional(LSTM(config["lstm_units_2"])))
    else:
        model.add(LSTM(config["lstm_units_2"]))
    
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout_rate"]))
    
    # Capas densas
    model.add(Dense(config["dense_units"], activation='relu'))
    model.add(Dropout(config["dropout_rate"]))
    
    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilar
    optimizer = Adam(learning_rate=config["learning_rate"])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    # Resumen
    print(f"\n   📋 Arquitectura:")
    model.summary()
    
    # Mostrar dispositivo
    print(f"\n   🖥️  Dispositivo de entrenamiento: {'GPU' if GPU_AVAILABLE else 'CPU'}")
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, config, output_folder):
    """
    Entrena el modelo con early stopping y callbacks.
    """
    print("\n" + "="*70)
    print("🎯 ENTRENANDO MODELO")
    print("="*70)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Monitor de GPU/tiempo
        GPUMonitorCallback(),
        
        # Early stopping
        EarlyStopping(
            monitor='val_recall',
            patience=config["early_stopping_patience"],
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reducir learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config["reduce_lr_patience"],
            min_lr=1e-6,
            verbose=1
        ),
        
        # Guardar mejor modelo
        ModelCheckpoint(
            os.path.join(output_folder, "best_model.h5"),
            monitor='val_recall',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    # Class weights
    class_weights = None
    if config["use_class_weights"]:
        class_weights = calculate_class_weights(y_train)
    
    print(f"\n   ⚙️  Configuración:")
    print(f"      Dispositivo: {'🎮 GPU' if GPU_AVAILABLE else '💻 CPU'}")
    print(f"      Batch size: {config['batch_size']}")
    print(f"      Max epochs: {config['epochs']}")
    print(f"      Early stopping: {config['early_stopping_patience']} epochs")
    
    # Entrenar
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print("\n   ✅ Entrenamiento completado")
    
    return model, history


def evaluate_model(model, X_test, y_test, output_folder):
    """
    Evalúa el modelo y genera métricas.
    """
    print("\n" + "="*70)
    print("📊 EVALUANDO MODELO")
    print("="*70)
    
    # Predicciones
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Buscar threshold óptimo para recall
    best_threshold = 0.5
    best_recall = 0
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_temp = (y_prob >= thresh).astype(int)
        recall = sum((y_pred_temp == 1) & (y_test == 1)) / sum(y_test) if sum(y_test) > 0 else 0
        if recall >= 0.95 and thresh > best_threshold * 0.5:
            best_threshold = thresh
            best_recall = recall
            break
    
    # Métricas con threshold 0.5
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        "threshold_default": {
            "threshold": 0.5,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0
        },
        "threshold_optimized": {
            "threshold": float(best_threshold),
            "accuracy": float(accuracy_score(y_test, (y_prob >= best_threshold).astype(int))),
            "precision": float(precision_score(y_test, (y_prob >= best_threshold).astype(int), zero_division=0)),
            "recall": float(recall_score(y_test, (y_prob >= best_threshold).astype(int), zero_division=0)),
            "f1": float(f1_score(y_test, (y_prob >= best_threshold).astype(int), zero_division=0))
        },
        "gpu_used": GPU_AVAILABLE
    }
    
    # Imprimir resultados
    print("\n   ╔════════════════════════════════════════════════════════════╗")
    print("   ║              MÉTRICAS DEL MODELO LSTM (GPU)                ║")
    print("   ╠════════════════════════════════════════════════════════════╣")
    print(f"   ║  Accuracy:   {metrics['threshold_default']['accuracy']*100:>6.2f}%                                ║")
    print(f"   ║  Precision:  {metrics['threshold_default']['precision']*100:>6.2f}%                                ║")
    print(f"   ║  Recall:     {metrics['threshold_default']['recall']*100:>6.2f}%  ← Detecta caídas              ║")
    print(f"   ║  F1-Score:   {metrics['threshold_default']['f1']*100:>6.2f}%                                ║")
    print(f"   ║  AUC-ROC:    {metrics['threshold_default']['auc_roc']*100:>6.2f}%                                ║")
    print("   ╚════════════════════════════════════════════════════════════╝")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   📋 Matriz de Confusión:")
    print(f"                  Predicho")
    print(f"                 ADL    Caída")
    print(f"      Real ADL   {cm[0,0]:>4}   {cm[0,1]:>4}")
    print(f"      Real Caída {cm[1,0]:>4}   {cm[1,1]:>4}")
    
    # Reporte de clasificación
    print(f"\n   📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['ADL', 'Caída'], zero_division=0))
    
    return metrics, y_prob


def plot_training_history(history, metrics, y_test, y_prob, output_folder):
    """
    Genera gráficos del entrenamiento.
    """
    print("\n📊 Generando gráficos...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    gpu_text = "🎮 GPU" if GPU_AVAILABLE else "💻 CPU"
    fig.suptitle(f'🛡️ SafeGuard Vision AI - Entrenamiento LSTM ({gpu_text})\nDetección de Caídas Temporal', 
                 fontsize=14, fontweight='bold')
    
    # 1. Loss
    ax1 = axes[0, 0]
    ax1.plot(history.history['loss'], label='Train', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Función de Pérdida', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Accuracy
    ax2 = axes[0, 1]
    ax2.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Precisión', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Recall
    ax3 = axes[0, 2]
    ax3.plot(history.history['recall'], label='Train', linewidth=2)
    ax3.plot(history.history['val_recall'], label='Validation', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Recall')
    ax3.set_title('Recall (Detección de Caídas)', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Curva ROC
    ax4 = axes[1, 0]
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax4.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        ax4.fill_between(fpr, tpr, alpha=0.3)
    ax4.plot([0, 1], [0, 1], 'r--')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('Curva ROC', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Distribución de probabilidades
    ax5 = axes[1, 1]
    ax5.hist(y_prob[y_test==0], bins=30, alpha=0.7, label='ADL', color='green')
    ax5.hist(y_prob[y_test==1], bins=30, alpha=0.7, label='Caída', color='red')
    ax5.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    ax5.set_xlabel('Probabilidad')
    ax5.set_ylabel('Frecuencia')
    ax5.set_title('Distribución de Predicciones', fontweight='bold')
    ax5.legend()
    
    # 6. Matriz de confusión
    ax6 = axes[1, 2]
    cm = confusion_matrix(y_test, (y_prob >= 0.5).astype(int))
    im = ax6.imshow(cm, interpolation='nearest', cmap='Blues')
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['ADL', 'Caída'])
    ax6.set_yticklabels(['ADL', 'Caída'])
    ax6.set_xlabel('Predicción')
    ax6.set_ylabel('Real')
    ax6.set_title('Matriz de Confusión', fontweight='bold')
    
    # Añadir números
    for i in range(2):
        for j in range(2):
            ax6.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_folder, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Gráficos guardados: {plot_path}")


def save_model_and_report(model, metrics, config, output_folder, training_time=0):
    """
    Guarda el modelo y el reporte.
    """
    print("\n💾 Guardando modelo...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(output_folder, "modelo_lstm.h5")
    model.save(model_path)
    print(f"   ✅ Modelo: {model_path}")
    
    # Guardar reporte
    report = {
        "project": "SafeGuard Vision AI",
        "model_type": "LSTM (Bidirectional)",
        "created": datetime.now().isoformat(),
        "training_device": "GPU" if GPU_AVAILABLE else "CPU",
        "tensorflow_version": tf.__version__,
        "config": config,
        "metrics": metrics,
        "files": [
            "modelo_lstm.h5",
            "best_model.h5",
            "training_history.png"
        ],
        "usage": {
            "sequence_length": 30,
            "features": "keypoints + velocidad + aceleración",
            "threshold": 0.5
        }
    }
    
    report_path = os.path.join(output_folder, "lstm_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"   ✅ Reporte: {report_path}")
    
    return model_path


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                              MAIN
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SAFEGUARD VISION AI - ENTRENAMIENTO LSTM (GPU) ".center(70) + "║")
    print("║" + " Detectando TRANSICIONES de caída ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # Mostrar estado de GPU
    if GPU_AVAILABLE:
        print("\n   🎮 MODO GPU ACTIVADO")
    else:
        print("\n   💻 MODO CPU (GPU no detectada)")
        print("\n   💡 Para usar GPU, ejecuta:")
        print("      pip uninstall tensorflow tensorflow-intel -y")
        print("      pip install tensorflow-directml")
    
    # Verificar carpeta de secuencias
    if not os.path.exists(SEQUENCES_FOLDER):
        print(f"\n❌ ERROR: No se encontró: {SEQUENCES_FOLDER}")
        print("   Primero ejecuta create_sequences.py")
        return
    
    # Cargar datos
    X_train, X_test, y_train, y_test = load_sequences(SEQUENCES_FOLDER)
    
    # Input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\n   📐 Input shape: {input_shape}")
    
    # Construir modelo
    model = build_lstm_model(input_shape, MODEL_CONFIG)
    
    # Entrenar
    start_time = time.time()
    model, history = train_model(
        model, X_train, y_train, X_test, y_test,
        MODEL_CONFIG, OUTPUT_FOLDER
    )
    training_time = time.time() - start_time
    
    # Evaluar
    metrics, y_prob = evaluate_model(model, X_test, y_test, OUTPUT_FOLDER)
    
    # Gráficos
    plot_training_history(history, metrics, y_test, y_prob, OUTPUT_FOLDER)
    
    # Guardar
    model_path = save_model_and_report(model, metrics, MODEL_CONFIG, OUTPUT_FOLDER, training_time)
    
    # Resumen final
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " ✅ MODELO LSTM ENTRENADO ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    device = "🎮 GPU" if GPU_AVAILABLE else "💻 CPU"
    print(f"║  {device} - Tiempo: {training_time/60:.1f} minutos".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    recall = metrics['threshold_default']['recall']
    precision = metrics['threshold_default']['precision']
    auc = metrics['threshold_default']['auc_roc']
    print(f"║  🎯 Recall:    {recall*100:.1f}%  (detecta caídas)".ljust(71) + "║")
    print(f"║  📊 Precision: {precision*100:.1f}%  (evita falsas alarmas)".ljust(71) + "║")
    print(f"║  📈 AUC-ROC:   {auc*100:.1f}%".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📂 Modelo guardado en: {OUTPUT_FOLDER}".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print("║  📋 SIGUIENTE PASO:".ljust(71) + "║")
    print("║     python demo_video_lstm_v2.py".ljust(71) + "║")
    print("╚" + "═"*70 + "╝")
    print()


if __name__ == "__main__":
    main()
