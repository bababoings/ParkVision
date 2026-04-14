"""
CETYS Parking Detection - Model Evaluation
==========================================
Script independiente para evaluar el rendimiento del modelo clasificador binario
Genera métricas como Precision, Recall, F1-Score y gráficos como Matriz de Confusión y ROC Curve.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("CETYS Parking - Model Evaluation Metrics")
print("=" * 60)

# =============================================================================
# Configuración
# =============================================================================
IMG_WIDTH, IMG_HEIGHT = 96, 96
BATCH_SIZE = 32
MODEL_PATH = 'parking_mobilenetv2.h5'
VAL_DIR = 'train_data/test'
OUTPUT_DIR = 'metrics_output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"[!] Error: No se encontró el modelo en {MODEL_PATH}")
    print("[!] Por favor, ejecuta train_model.py primero para entrenar el modelo.")
    exit(1)

# =============================================================================
# Carga de Datos y Predicción
# =============================================================================
print("[1/4] Cargando datos y modelo...")

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

model = load_model(MODEL_PATH)
print("\n[2/4] Generando predicciones sobre el set de validación...")
validation_steps = max(1, val_generator.samples // BATCH_SIZE) + (1 if val_generator.samples % BATCH_SIZE != 0 else 0)

y_pred_probs = model.predict(val_generator, steps=validation_steps)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes[:len(y_pred)]
class_labels = list(val_generator.class_indices.keys())

# =============================================================================
# Cálculo de F1-Score (PRIORIDAD)
# =============================================================================
print("\n[3/4] Generando Métricas de Rendimiento...")

f1 = f1_score(y_true, y_pred, average='weighted')

print("\n" + "*" * 60)
print(f"*** F1-SCORE GENERAL DEL MODELO: {f1:.4f} ***")
print("El F1-Score es el balance perfecto entre Precisión (falsos positivos) y Recall (falsos negativos).")
print("*" * 60 + "\n")

print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

# =============================================================================
# Gráficos: Matriz de Confusión
# =============================================================================
print("[4/4] Generando gráficas en 'metrics_output/'...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusión\n(Qué tanto se equivoca y acierta el modelo)', fontsize=14)
plt.ylabel('Etiqueta Verdadera (Realidad)')
plt.xlabel('Etiqueta Predicha (Lo que dijo la IA)')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f" [+] Matriz de Confusión guardada en: {cm_path}")

# =============================================================================
# Gráficos: Curva ROC y AUC
# =============================================================================
# Asumiendo clasificación binaria para curva ROC simplificada
if len(class_labels) == 2:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (False Positive Rate)')
    plt.ylabel('Tasa de Verdaderos Positivos (True Positive Rate)')
    plt.title('Receiver Operating Characteristic (Curva ROC)', fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f" [+] Curva ROC guardada en:          {roc_path}")

print("\n" + "=" * 60)
print("¡Evaluación y Generación de Gráficas completada!")
print("=" * 60)
