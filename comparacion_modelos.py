"""
SAFEGUARD VISION AI
Comparación científica de modelos de detección de caídas
Nivel Maestría / MIT

Autor: Christian Cajusol
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

MODEL_PATHS = {
    "RF_v1": {
        "path": "train_model/training_report.json",
        "paradigm": "Frame-based (RF)",
    },
    "RF_v2": {
        "path": "train_model_v2/training_report_v2.json",
        "paradigm": "Frame-based + balance (RF)",
    },
    "LSTM": {
        "path": "safeguard_model_lstm/lstm_report.json",
        "paradigm": "Sequence-based (LSTM)",
    },
    "Transformer": {
        "path": "safeguard_model_transformer/transformer_report.json",
        "paradigm": "Attention-based (Transformer)",
    }
}

OUTPUT_DIR = "model_comparison_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 1. NORMALIZADOR DE MÉTRICAS (CLAVE)
# ==========================================================

def extract_metrics(report):
    """
    Unifica métricas desde distintos esquemas de evaluación
    """
    m = report.get("metrics", {})

    # LSTM / Transformer
    if "threshold_default" in m:
        td = m["threshold_default"]
        return {
            "accuracy": td["accuracy"],
            "precision": td["precision"],
            "recall": td["recall"],
            "f1": td["f1"],
            "auc": td.get("auc_roc", None),
        }

    # Random Forest v2
    if "recall_optimized" in m:
        return {
            "accuracy": m.get("test_accuracy", None),
            "precision": m["precision_optimized"],
            "recall": m["recall_optimized"],
            "f1": m["f1_optimized"],
            "auc": m.get("auc_roc", None),
        }

    # Random Forest v1
    return {
        "accuracy": m.get("test_accuracy"),
        "precision": m.get("test_precision"),
        "recall": m.get("test_recall"),
        "f1": m.get("test_f1_score"),
        "auc": m.get("test_auc_roc"),
    }

# ==========================================================
# 2. CARGA GLOBAL DE RESULTADOS
# ==========================================================

def load_all_models():
    results = {}

    for model_name, cfg in MODEL_PATHS.items():
        with open(cfg["path"], "r") as f:
            report = json.load(f)

        metrics = extract_metrics(report)

        results[model_name] = {
            "paradigm": cfg["paradigm"],
            "metrics": metrics
        }

    return results

# ==========================================================
# 3. FIGURA CENTRAL — RECALL + FN (CRÍTICO)
# ==========================================================

def plot_recall_comparison(results):
    names = list(results.keys())
    recalls = [results[n]["metrics"]["recall"] for n in names]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, recalls)
    plt.ylim(0, 1.05)
    plt.ylabel("Recall (Detección de Caídas)")
    plt.title("Comparación de Recall entre Modelos\n(Eventos Críticos)")

    for bar, r in zip(bars, recalls):
        plt.text(bar.get_x() + bar.get_width()/2, r + 0.02,
                 f"{r*100:.1f}%", ha="center", va="bottom")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "recall_comparison.png"), dpi=150)
    plt.close()

# ==========================================================
# 4. FIGURA COST-SENSITIVE (FN vs FP – conceptual)
# ==========================================================

def plot_cost_sensitive(results):
    """
    Aproximación conceptual: FN ~ (1 - recall)
    FP ~ (1 - precision)
    """
    plt.figure(figsize=(7, 6))

    for name, r in results.items():
        recall = r["metrics"]["recall"]
        precision = r["metrics"]["precision"]

        fn = 1 - recall
        fp = 1 - precision

        plt.scatter(fn, fp, s=120)
        plt.text(fn + 0.01, fp + 0.01, name)

    plt.xlabel("False Negatives (Caídas NO detectadas)")
    plt.ylabel("False Positives (Falsas alarmas)")
    plt.title("Análisis Sensible al Costo\n(FN vs FP)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cost_sensitive_analysis.png"), dpi=150)
    plt.close()

# ==========================================================
# 5. TABLA RESUMEN (PARA REPORTE)
# ==========================================================

def print_summary_table(results):
    print("\n=== RESUMEN COMPARATIVO DE MODELOS ===\n")
    print(f"{'Modelo':<12} {'Recall':>8} {'Precision':>10} {'F1':>8} {'AUC':>8}")
    print("-"*50)

    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<12} {m['recall']*100:>7.1f}% {m['precision']*100:>9.1f}% {m['f1']*100:>7.1f}% {m['auc']*100 if m['auc'] else 0:>7.1f}%")

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    results = load_all_models()
    print_summary_table(results)
    plot_recall_comparison(results)
    plot_cost_sensitive(results)

    print("\nFiguras guardadas en:", OUTPUT_DIR)
