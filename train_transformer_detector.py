"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - TRANSFORMER TEMPORAL                            ║
║                                                                              ║
║   Usa mecanismo de ATENCIÓN para detectar transiciones de caída             ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

¿POR QUÉ TRANSFORMER?
=====================
    LSTM: Procesa frame por frame → Puede "olvidar" el inicio
    
    Transformer: Ve TODA la secuencia a la vez con ATENCIÓN
                 → Compara frame 1 (parado) con frame 30 (en suelo) directamente

ARQUITECTURA:
=============
    Input: (batch, 30 frames, 396 features)
           ↓
    Positional Encoding (añade información de posición temporal)
           ↓
    Transformer Encoder (Multi-Head Self-Attention)
           ↓
    Global Average Pooling
           ↓
    Dense layers
           ↓
    Output: Probabilidad de caída

VENTAJAS:
=========
    - Attention puede enfocarse en los frames CRÍTICOS (momento de caída)
    - No tiene problema de "olvido" como LSTM
    - Paralelizable (más rápido en GPU)
    - State-of-the-art en muchas tareas de secuencias
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Semillas para reproducibilidad
SEED = 42
np.random.seed(SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

print(f"TensorFlow version: {tf.__version__}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CONFIGURACIÓN
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Carpeta con las secuencias (creada por create_sequences.py)
SEQUENCES_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_sequences"

# Carpeta de salida para el modelo
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_model_transformer"

# ═══════════════════════════════════════════════════════════════════════════════
# HIPERPARÁMETROS DEL TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    # Arquitectura Transformer
    "num_heads": 8,              # Número de cabezas de atención
    "head_size": 64,             # Dimensión por cabeza
    "ff_dim": 256,               # Dimensión de feed-forward
    "num_transformer_blocks": 3, # Número de bloques transformer
    "mlp_units": [128, 64],      # Unidades en MLP final
    "dropout_rate": 0.3,
    "mlp_dropout": 0.4,
    
    # Entrenamiento
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.0005,     # Learning rate más bajo para transformers
    "early_stopping_patience": 15,
    "reduce_lr_patience": 5,
    
    # Pesos de clase
    "use_class_weights": True,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CAPAS PERSONALIZADAS
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PositionalEncoding(layers.Layer):
    """
    Añade información de posición temporal a la secuencia.
    Sin esto, el Transformer no sabría el ORDEN de los frames.
    """
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
        # Crear encodings posicionales
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pe = np.zeros((sequence_length, embed_dim))
        pe[:, 0::2] = np.sin(position * div_term[:embed_dim//2 + embed_dim%2])
        pe[:, 1::2] = np.cos(position * div_term[:embed_dim//2])
        
        self.positional_encoding = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        # x shape: (batch, seq_len, features)
        return x + self.positional_encoding[:tf.shape(x)[1], :tf.shape(x)[2]]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim
        })
        return config


class TransformerBlock(layers.Layer):
    """
    Un bloque del Transformer Encoder.
    
    Componentes:
    1. Multi-Head Self-Attention
    2. Feed-Forward Network
    3. Layer Normalization
    4. Residual connections
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-Head Attention
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads
        )
        
        # Feed-Forward Network
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        
        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # Self-Attention con residual connection
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-Forward con residual connection
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
#                         CALLBACK DE MONITOREO
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TrainingMonitorCallback(Callback):
    """Monitorea el progreso del entrenamiento."""
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.start_time = None
        self.total_start = None
        
    def on_train_begin(self, logs=None):
        self.total_start = time.time()
        print(f"\n   🚀 Entrenamiento Transformer iniciado...")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        if epoch < 3 or (epoch + 1) % 5 == 0:
            avg_time = np.mean(self.epoch_times)
            print(f"      Epoch {epoch+1}: {epoch_time:.2f}s (promedio: {avg_time:.2f}s)")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.total_start
        print(f"\n   ⏱️  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         FUNCIONES
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_sequences(folder):
    """Carga las secuencias preparadas."""
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
    
    # Guardar estadísticas
    np.save(os.path.join(folder, "norm_mean.npy"), mean)
    np.save(os.path.join(folder, "norm_std.npy"), std)
    
    print("   ✅ Datos normalizados")
    
    return X_train, X_test, y_train, y_test


def calculate_class_weights(y_train):
    """Calcula pesos de clase."""
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


def build_transformer_model(input_shape, config):
    """
    Construye el modelo Transformer para clasificación de secuencias.
    
    Args:
        input_shape: (sequence_length, n_features)
        config: Diccionario de configuración
        
    Returns:
        model: Modelo Keras compilado
    """
    print("\n" + "="*70)
    print("🏗️  CONSTRUYENDO MODELO TRANSFORMER")
    print("="*70)
    
    sequence_length, n_features = input_shape
    
    # Input
    inputs = layers.Input(shape=input_shape)
    
    # Proyección a dimensión del modelo
    embed_dim = config["num_heads"] * config["head_size"]
    x = layers.Dense(embed_dim)(inputs)
    
    # Positional Encoding
    x = PositionalEncoding(sequence_length, embed_dim)(x)
    
    # Transformer Blocks
    for i in range(config["num_transformer_blocks"]):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            dropout_rate=config["dropout_rate"]
        )(x)
    
    # Global pooling
    # Opción 1: Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # MLP Head
    for units in config["mlp_units"]:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(config["mlp_dropout"])(x)
    
    # Output
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    # Crear modelo
    model = Model(inputs, outputs)
    
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
    print(f"\n   📋 Arquitectura Transformer:")
    print(f"      • Embedding dim: {embed_dim}")
    print(f"      • Attention heads: {config['num_heads']}")
    print(f"      • Transformer blocks: {config['num_transformer_blocks']}")
    print(f"      • Feed-forward dim: {config['ff_dim']}")
    print(f"      • MLP units: {config['mlp_units']}")
    
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, config, output_folder):
    """Entrena el modelo."""
    print("\n" + "="*70)
    print("🎯 ENTRENANDO TRANSFORMER")
    print("="*70)
    
    os.makedirs(output_folder, exist_ok=True)
    
    callbacks = [
        TrainingMonitorCallback(),
        
        EarlyStopping(
            monitor='val_recall',
            patience=config["early_stopping_patience"],
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config["reduce_lr_patience"],
            min_lr=1e-7,
            verbose=1
        ),
        
        ModelCheckpoint(
            os.path.join(output_folder, "best_model.h5"),
            monitor='val_recall',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    class_weights = None
    if config["use_class_weights"]:
        class_weights = calculate_class_weights(y_train)
    
    print(f"\n   ⚙️  Configuración:")
    print(f"      Batch size: {config['batch_size']}")
    print(f"      Max epochs: {config['epochs']}")
    print(f"      Learning rate: {config['learning_rate']}")
    
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
    """Evalúa el modelo."""
    print("\n" + "="*70)
    print("📊 EVALUANDO MODELO TRANSFORMER")
    print("="*70)
    
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        "threshold_default": {
            "threshold": 0.5,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0
        }
    }
    
    print("\n   ╔════════════════════════════════════════════════════════════╗")
    print("   ║            MÉTRICAS DEL TRANSFORMER                        ║")
    print("   ╠════════════════════════════════════════════════════════════╣")
    print(f"   ║  Accuracy:   {metrics['threshold_default']['accuracy']*100:>6.2f}%                                ║")
    print(f"   ║  Precision:  {metrics['threshold_default']['precision']*100:>6.2f}%                                ║")
    print(f"   ║  Recall:     {metrics['threshold_default']['recall']*100:>6.2f}%  ← Detecta caídas              ║")
    print(f"   ║  F1-Score:   {metrics['threshold_default']['f1']*100:>6.2f}%                                ║")
    print(f"   ║  AUC-ROC:    {metrics['threshold_default']['auc_roc']*100:>6.2f}%                                ║")
    print("   ╚════════════════════════════════════════════════════════════╝")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   📋 Matriz de Confusión:")
    print(f"                  Predicho")
    print(f"                 ADL    Caída")
    print(f"      Real ADL   {cm[0,0]:>4}   {cm[0,1]:>4}")
    print(f"      Real Caída {cm[1,0]:>4}   {cm[1,1]:>4}")
    
    print(f"\n   📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['ADL', 'Caída'], zero_division=0))
    
    return metrics, y_prob


def plot_training_history(history, metrics, y_test, y_prob, output_folder):
    """Genera gráficos."""
    print("\n📊 Generando gráficos...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('🛡️ SafeGuard Vision AI - Transformer\nDetección de Caídas con Atención', 
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
    
    # 5. Distribución
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
    im = ax6.imshow(cm, interpolation='nearest', cmap='Purples')  # Púrpura para Transformer
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['ADL', 'Caída'])
    ax6.set_yticklabels(['ADL', 'Caída'])
    ax6.set_xlabel('Predicción')
    ax6.set_ylabel('Real')
    ax6.set_title('Matriz de Confusión', fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            ax6.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_folder, "training_history_transformer.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Gráficos guardados: {plot_path}")


def save_model_and_report(model, metrics, config, output_folder):
    """Guarda el modelo y reporte."""
    print("\n💾 Guardando modelo Transformer...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    model_path = os.path.join(output_folder, "modelo_transformer.h5")
    model.save(model_path)
    print(f"   ✅ Modelo: {model_path}")
    
    report = {
        "project": "SafeGuard Vision AI",
        "model_type": "Transformer (Self-Attention)",
        "created": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "architecture": {
            "type": "Temporal Transformer",
            "attention_heads": config["num_heads"],
            "transformer_blocks": config["num_transformer_blocks"],
            "embedding_dim": config["num_heads"] * config["head_size"]
        },
        "files": [
            "modelo_transformer.h5",
            "best_model.h5",
            "training_history_transformer.png"
        ],
        "usage": {
            "sequence_length": 30,
            "features": "keypoints + velocidad + aceleración",
            "threshold": 0.5
        }
    }
    
    report_path = os.path.join(output_folder, "transformer_report.json")
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
    print("║" + " 🛡️  SAFEGUARD VISION AI - TRANSFORMER ".center(70) + "║")
    print("║" + " Detección de Caídas con Mecanismo de Atención ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # Verificar carpeta
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
    model = build_transformer_model(input_shape, MODEL_CONFIG)
    
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
    model_path = save_model_and_report(model, metrics, MODEL_CONFIG, OUTPUT_FOLDER)
    
    # Resumen final
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " ✅ TRANSFORMER ENTRENADO ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  ⏱️  Tiempo: {training_time/60:.1f} minutos".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    recall = metrics['threshold_default']['recall']
    precision = metrics['threshold_default']['precision']
    auc = metrics['threshold_default']['auc_roc']
    print(f"║  🎯 Recall:    {recall*100:.1f}%  (detecta caídas)".ljust(71) + "║")
    print(f"║  📊 Precision: {precision*100:.1f}%  (evita falsas alarmas)".ljust(71) + "║")
    print(f"║  📈 AUC-ROC:   {auc*100:.1f}%".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📂 Guardado en: {OUTPUT_FOLDER}".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print("║  📋 SIGUIENTE: Crear demo_video_transformer.py".ljust(71) + "║")
    print("╚" + "═"*70 + "╝")
    print()


if __name__ == "__main__":
    main()
