"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - ENTRENAMIENTO v2 (MEJORADO)                     ║
║                                                                              ║
║   PASO 3: Entrena con dataset balanceado + anti-overfitting                 ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

MEJORAS vs v1:
==============
    1. Usa dataset BALANCEADO (ratio 1:1)
    2. Parámetros anti-overfitting (max_depth=10, min_samples_leaf=10)
    3. Threshold optimizado para recall (0.35)
    4. Validación más robusta
    5. Optimiza para RECALL (detectar todas las caídas)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         CONFIGURACIÓN
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════════════════════════
# RUTAS - ACTUALIZA ESTAS CON TUS RUTAS
# ═══════════════════════════════════════════════════════════════════════════════

# CSV BALANCEADO (generado por balance_dataset.py)
CSV_PATH = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\train_model_v2\keypoints_balanced.csv"

# Carpeta de salida para el modelo
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\train_model_v2"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN ANTI-OVERFITTING
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cross_validation_folds": 5,
}

# Parámetros ANTI-OVERFITTING (más restrictivos que v1)
MODEL_PARAMS = {
    'n_estimators': 200,        # Número de árboles
    'max_depth': 10,            # ← REDUCIDO de 20 a 10 (evita memorización)
    'min_samples_split': 10,    # ← AUMENTADO de 2 a 10
    'min_samples_leaf': 10,     # ← AUMENTADO de 1 a 10
    'max_features': 'sqrt',     # ← AÑADIDO: usa sqrt de features
    'class_weight': 'balanced', # ← AÑADIDO: pondera clases automáticamente
    'random_state': 42,
    'n_jobs': -1
}

# Threshold para clasificación (optimizado para recall)
CLASSIFICATION_THRESHOLD = 0.35  # ← REDUCIDO de 0.5 a 0.35


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                         FUNCIONES
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_and_prepare_data(csv_path):
    """Carga el dataset balanceado."""
    print("\n📂 Cargando dataset BALANCEADO...")
    
    df = pd.read_csv(csv_path)
    
    total = len(df)
    falls = len(df[df['label'] == 1])
    adl = len(df[df['label'] == 0])
    ratio = adl / falls if falls > 0 else 0
    
    print(f"   ✅ Cargadas {total:,} muestras")
    print(f"   📊 Caídas: {falls:,} | ADL: {adl:,} | Ratio: {ratio:.2f}:1")
    
    if ratio > 2:
        print(f"   ⚠️  ADVERTENCIA: Dataset aún desbalanceado (ratio > 2)")
    
    # Separar features y labels
    metadata_columns = ['filename', 'folder', 'dataset', 'label', 'label_name']
    feature_columns = [col for col in df.columns if col not in metadata_columns]
    
    X = df[feature_columns].values
    y = df['label'].values
    
    # Limpiar valores problemáticos
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return X, y, feature_columns, df


def create_additional_features(X, feature_names):
    """Crea features derivadas (igual que v1)."""
    print("\n🔧 Creando features adicionales...")
    
    df_features = pd.DataFrame(X, columns=feature_names)
    
    new_features = []
    new_names = []
    
    # 1. Ángulo del torso
    torso_angle = df_features['nose_y'] - (df_features['left_hip_y'] + df_features['right_hip_y']) / 2
    new_features.append(torso_angle.values)
    new_names.append('torso_angle')
    
    # 2. Altura del cuerpo
    y_columns = [col for col in feature_names if col.endswith('_y')]
    y_values = df_features[y_columns]
    body_height = y_values.max(axis=1) - y_values.min(axis=1)
    new_features.append(body_height.values)
    new_names.append('body_height')
    
    # 3. Ancho del cuerpo
    x_columns = [col for col in feature_names if col.endswith('_x')]
    x_values = df_features[x_columns]
    body_width = x_values.max(axis=1) - x_values.min(axis=1)
    new_features.append(body_width.values)
    new_names.append('body_width')
    
    # 4. Ratio altura/ancho
    aspect_ratio = body_height / (body_width + 0.001)
    new_features.append(aspect_ratio.values)
    new_names.append('aspect_ratio')
    
    # 5. Centro de masa Y
    center_y = (df_features['nose_y'] + df_features['left_hip_y'] + 
                df_features['right_hip_y'] + df_features['left_shoulder_y'] + 
                df_features['right_shoulder_y']) / 5
    new_features.append(center_y.values)
    new_names.append('center_mass_y')
    
    # 6. Distancia hombros
    shoulder_dist = np.sqrt((df_features['left_shoulder_x'] - df_features['right_shoulder_x'])**2 +
                           (df_features['left_shoulder_y'] - df_features['right_shoulder_y'])**2)
    new_features.append(shoulder_dist.values)
    new_names.append('shoulder_distance')
    
    # 7. Distancia caderas
    hip_dist = np.sqrt((df_features['left_hip_x'] - df_features['right_hip_x'])**2 +
                       (df_features['left_hip_y'] - df_features['right_hip_y'])**2)
    new_features.append(hip_dist.values)
    new_names.append('hip_distance')
    
    # 8. Ángulo de piernas
    leg_angle = ((df_features['left_hip_y'] + df_features['right_hip_y']) / 2 - 
                 (df_features['left_ankle_y'] + df_features['right_ankle_y']) / 2)
    new_features.append(leg_angle.values)
    new_names.append('leg_angle')
    
    # Combinar
    new_features_array = np.column_stack(new_features)
    X_enhanced = np.hstack([X, new_features_array])
    enhanced_feature_names = list(feature_names) + new_names
    
    print(f"   ✅ {len(new_names)} features adicionales creadas")
    print(f"   📊 Total features: {len(enhanced_feature_names)}")
    
    return X_enhanced, enhanced_feature_names


def find_optimal_threshold(y_true, y_prob, target_recall=0.95):
    """
    Encuentra el threshold óptimo para alcanzar el recall objetivo.
    Para seguridad industrial, priorizamos detectar TODAS las caídas.
    """
    print(f"\n🎯 Buscando threshold óptimo (recall objetivo: {target_recall*100}%)...")
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Encontrar threshold que da recall >= objetivo
    optimal_threshold = 0.5
    for i, (prec, rec, thresh) in enumerate(zip(precisions[:-1], recalls[:-1], thresholds)):
        if rec >= target_recall:
            optimal_threshold = thresh
            print(f"   ✅ Threshold óptimo: {optimal_threshold:.3f}")
            print(f"      Recall: {rec*100:.1f}% | Precision: {prec*100:.1f}%")
            break
    
    return optimal_threshold


def train_model(X_train, y_train):
    """Entrena el modelo con parámetros anti-overfitting."""
    print("\n🧠 Entrenando Random Forest (anti-overfitting)...")
    print(f"   📋 Parámetros:")
    for key, value in MODEL_PARAMS.items():
        print(f"      • {key}: {value}")
    
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    print("   ✅ Modelo entrenado")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, output_folder):
    """Evalúa el modelo con múltiples métricas."""
    print("\n📊 Evaluando modelo...")
    
    # Predicciones con threshold por defecto (0.5)
    y_pred_train = model.predict(X_train)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    # Encontrar threshold óptimo para recall
    optimal_threshold = find_optimal_threshold(y_test, y_prob_test, target_recall=0.95)
    
    # Predicciones con threshold optimizado
    y_pred_test_default = (y_prob_test >= 0.5).astype(int)
    y_pred_test_optimized = (y_prob_test >= optimal_threshold).astype(int)
    
    # Métricas con threshold 0.5
    metrics_default = {
        'threshold': 0.5,
        'accuracy': accuracy_score(y_test, y_pred_test_default),
        'precision': precision_score(y_test, y_pred_test_default),
        'recall': recall_score(y_test, y_pred_test_default),
        'f1': f1_score(y_test, y_pred_test_default),
    }
    
    # Métricas con threshold optimizado
    metrics_optimized = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(y_test, y_pred_test_optimized),
        'precision': precision_score(y_test, y_pred_test_optimized),
        'recall': recall_score(y_test, y_pred_test_optimized),
        'f1': f1_score(y_test, y_pred_test_optimized),
    }
    
    # AUC-ROC
    auc_roc = roc_auc_score(y_test, y_prob_test)
    
    # Validación cruzada
    cv = StratifiedKFold(n_splits=CONFIG['cross_validation_folds'], shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, np.vstack([X_train, X_test]), 
                                np.concatenate([y_train, y_test]), 
                                cv=cv, scoring='recall')
    
    # Train accuracy (para verificar overfitting)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    
    # Imprimir comparación
    print("\n   ╔════════════════════════════════════════════════════════════╗")
    print("   ║              COMPARACIÓN DE THRESHOLDS                     ║")
    print("   ╠════════════════════════════════════════════════════════════╣")
    print(f"   ║  Métrica       │  Threshold=0.5  │  Threshold={optimal_threshold:.2f}       ║")
    print("   ╠────────────────┼─────────────────┼────────────────────────╣")
    print(f"   ║  Accuracy      │     {metrics_default['accuracy']*100:>5.1f}%      │     {metrics_optimized['accuracy']*100:>5.1f}%            ║")
    print(f"   ║  Precision     │     {metrics_default['precision']*100:>5.1f}%      │     {metrics_optimized['precision']*100:>5.1f}%            ║")
    print(f"   ║  Recall        │     {metrics_default['recall']*100:>5.1f}%      │     {metrics_optimized['recall']*100:>5.1f}%  ← MEJOR   ║")
    print(f"   ║  F1-Score      │     {metrics_default['f1']*100:>5.1f}%      │     {metrics_optimized['f1']*100:>5.1f}%            ║")
    print("   ╚════════════════════════════════════════════════════════════╝")
    
    print(f"\n   📈 AUC-ROC: {auc_roc*100:.1f}%")
    print(f"   📊 CV Recall (mean±std): {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
    
    # Verificar overfitting
    print(f"\n   🔍 Verificación de Overfitting:")
    print(f"      Train Accuracy: {train_accuracy*100:.1f}%")
    print(f"      Test Accuracy:  {metrics_default['accuracy']*100:.1f}%")
    gap = (train_accuracy - metrics_default['accuracy']) * 100
    if gap > 5:
        print(f"      ⚠️  Gap de {gap:.1f}% - Posible overfitting leve")
    else:
        print(f"      ✅ Gap de {gap:.1f}% - Sin overfitting significativo")
    
    # Crear visualizaciones
    create_evaluation_plots(model, X_test, y_test, y_prob_test, feature_names, 
                           optimal_threshold, output_folder)
    
    return {
        'train_accuracy': round(train_accuracy, 4),
        'test_accuracy_default': round(metrics_default['accuracy'], 4),
        'test_accuracy_optimized': round(metrics_optimized['accuracy'], 4),
        'precision_default': round(metrics_default['precision'], 4),
        'precision_optimized': round(metrics_optimized['precision'], 4),
        'recall_default': round(metrics_default['recall'], 4),
        'recall_optimized': round(metrics_optimized['recall'], 4),
        'f1_default': round(metrics_default['f1'], 4),
        'f1_optimized': round(metrics_optimized['f1'], 4),
        'auc_roc': round(auc_roc, 4),
        'cv_recall_mean': round(cv_scores.mean(), 4),
        'cv_recall_std': round(cv_scores.std(), 4),
        'optimal_threshold': round(optimal_threshold, 4),
        'recommended_threshold': round(optimal_threshold, 2)
    }


def create_evaluation_plots(model, X_test, y_test, y_prob, feature_names, optimal_threshold, output_folder):
    """Crea gráficos de evaluación."""
    print("\n📊 Generando gráficos de evaluación...")
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('🛡️ SafeGuard Vision AI - Evaluación del Modelo v2\n(Dataset Balanceado + Anti-Overfitting)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Matriz de Confusión (threshold optimizado)
    ax1 = fig.add_subplot(2, 3, 1)
    y_pred = (y_prob >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ADL', 'Fall'], 
                yticklabels=['ADL', 'Fall'],
                ax=ax1, annot_kws={'size': 16})
    ax1.set_xlabel('Predicción', fontweight='bold')
    ax1.set_ylabel('Real', fontweight='bold')
    ax1.set_title(f'Matriz de Confusión\n(threshold={optimal_threshold:.2f})', fontweight='bold')
    
    # 2. Curva ROC
    ax2 = fig.add_subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=1)
    ax2.fill_between(fpr, tpr, alpha=0.3)
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')
    ax2.set_title('Curva ROC', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    
    # 3. Curva Precision-Recall
    ax3 = fig.add_subplot(2, 3, 3)
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    ax3.plot(rec, prec, 'g-', linewidth=2)
    ax3.axvline(x=0.95, color='red', linestyle='--', label='Recall objetivo (95%)')
    ax3.set_xlabel('Recall', fontweight='bold')
    ax3.set_ylabel('Precision', fontweight='bold')
    ax3.set_title('Curva Precision-Recall', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Distribución de probabilidades
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(y_prob[y_test==0], bins=30, alpha=0.7, label='ADL', color='#2ecc71')
    ax4.hist(y_prob[y_test==1], bins=30, alpha=0.7, label='Fall', color='#e74c3c')
    ax4.axvline(x=0.5, color='blue', linestyle='--', label='Threshold=0.5')
    ax4.axvline(x=optimal_threshold, color='orange', linestyle='-', linewidth=2, 
                label=f'Threshold={optimal_threshold:.2f} (óptimo)')
    ax4.set_xlabel('Probabilidad de Caída', fontweight='bold')
    ax4.set_ylabel('Frecuencia', fontweight='bold')
    ax4.set_title('Distribución de Probabilidades', fontweight='bold')
    ax4.legend()
    
    # 5. Feature Importance (Top 20)
    ax5 = fig.add_subplot(2, 3, 5)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    
    colors = ['#e74c3c' if any(x in feature_names[i] for x in ['torso', 'height', 'ratio', 'angle', 'center'])
              else '#3498db' for i in indices]
    
    ax5.barh(range(len(indices)), importances[indices], color=colors)
    ax5.set_yticks(range(len(indices)))
    ax5.set_yticklabels([feature_names[i][:20] for i in indices])
    ax5.set_xlabel('Importancia', fontweight='bold')
    ax5.set_title('Top 20 Features', fontweight='bold')
    
    # 6. Threshold vs Métricas
    ax6 = fig.add_subplot(2, 3, 6)
    thresholds_range = np.linspace(0.1, 0.9, 50)
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds_range:
        y_pred_t = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_t, zero_division=0))
    
    ax6.plot(thresholds_range, precisions, 'b-', label='Precision')
    ax6.plot(thresholds_range, recalls, 'g-', label='Recall')
    ax6.plot(thresholds_range, f1s, 'r-', label='F1-Score')
    ax6.axvline(x=optimal_threshold, color='orange', linestyle='--', 
                label=f'Óptimo ({optimal_threshold:.2f})')
    ax6.set_xlabel('Threshold', fontweight='bold')
    ax6.set_ylabel('Score', fontweight='bold')
    ax6.set_title('Threshold vs Métricas', fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = os.path.join(output_folder, "model_evaluation_v2.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Gráficos guardados: {plot_path}")


def save_model(model, scaler, feature_names, metrics, output_folder):
    """Guarda el modelo y archivos asociados."""
    print("\n💾 Guardando modelo v2...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Modelo
    model_path = os.path.join(output_folder, "modelo_caidas_v2.pkl")
    joblib.dump(model, model_path)
    print(f"   ✅ Modelo: {model_path}")
    
    # Scaler
    scaler_path = os.path.join(output_folder, "scaler_caidas_v2.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Scaler: {scaler_path}")
    
    # Features
    features_path = os.path.join(output_folder, "feature_names_v2.json")
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Reporte
    report = {
        "project": "SafeGuard Vision AI",
        "version": "v2 (Balanced + Anti-Overfitting)",
        "author": "Christian Cajusol - MIT Global Teaching Labs",
        "training_date": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "improvements": [
            "Dataset balanceado (ratio 1:1)",
            "Anti-overfitting: max_depth=10, min_samples_leaf=10",
            "Threshold optimizado para recall",
            "class_weight='balanced'"
        ],
        "parameters": MODEL_PARAMS,
        "metrics": metrics,
        "recommended_threshold": metrics['recommended_threshold'],
        "files": [
            "modelo_caidas_v2.pkl",
            "scaler_caidas_v2.pkl",
            "feature_names_v2.json",
            "model_evaluation_v2.png"
        ]
    }
    
    report_path = os.path.join(output_folder, "training_report_v2.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"   ✅ Reporte: {report_path}")
    
    return model_path


# ╔══════════════════════════════════════════════════════════════════════════════╗
#                              MAIN
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SAFEGUARD VISION AI - ENTRENAMIENTO v2 ".center(70) + "║")
    print("║" + " Dataset Balanceado + Anti-Overfitting ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # Verificar CSV
    if not os.path.exists(CSV_PATH):
        print(f"\n❌ ERROR: No se encontró el CSV balanceado:")
        print(f"   {CSV_PATH}")
        print("\n   Primero ejecuta balance_dataset.py")
        return
    
    # Cargar datos
    X, y, feature_names, df = load_and_prepare_data(CSV_PATH)
    
    # Crear features adicionales
    X, feature_names = create_additional_features(X, feature_names)
    
    # Dividir datos
    print("\n📊 Dividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Normalizar
    print("\n⚖️ Normalizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar
    model = train_model(X_train_scaled, y_train)
    
    # Evaluar
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, 
                            y_train, y_test, feature_names, OUTPUT_FOLDER)
    
    # Guardar
    save_model(model, scaler, feature_names, metrics, OUTPUT_FOLDER)
    
    # Resumen final
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " ✅ ENTRENAMIENTO v2 COMPLETADO ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  🎯 Recall (threshold={metrics['recommended_threshold']}): {metrics['recall_optimized']*100:.1f}%".ljust(71) + "║")
    print(f"║  📊 Precision: {metrics['precision_optimized']*100:.1f}%".ljust(71) + "║")
    print(f"║  📈 AUC-ROC: {metrics['auc_roc']*100:.1f}%".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📂 Modelo guardado en: {OUTPUT_FOLDER}".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print("║  💡 IMPORTANTE: Usa threshold={:.2f} en la demo".format(metrics['recommended_threshold']).ljust(71) + "║")
    print("╚" + "═"*70 + "╝")
    print()


if __name__ == "__main__":
    main()
