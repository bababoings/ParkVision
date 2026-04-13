"""
CETYS Parking Detection - Model Training
==========================================
Transfer Learning con MobileNetV2 para clasificación binaria:
  Clase 0: "empty" (Disponible)
  Clase 1: "occupied" (Ocupado)

Genera: parking_mobilenetv2.h5
Calcula F1-score al final del entrenamiento.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

print("=" * 60)
print("CETYS Parking - MobileNetV2 Training")
print("=" * 60)

# =============================================================================
# Configuration
# =============================================================================
IMG_WIDTH, IMG_HEIGHT = 96, 96
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2
LEARNING_RATE = 0.0001

TRAIN_DIR = 'train_data/train'
VAL_DIR = 'train_data/test'
MODEL_OUTPUT = 'parking_mobilenetv2.h5'

# Count files
files_train = 0
files_val = 0
for sub in os.listdir(TRAIN_DIR):
    path = os.path.join(TRAIN_DIR, sub)
    if os.path.isdir(path):
        files_train += len(os.listdir(path))
for sub in os.listdir(VAL_DIR):
    path = os.path.join(VAL_DIR, sub)
    if os.path.isdir(path):
        files_val += len(os.listdir(path))

print(f"Training images:   {files_train}")
print(f"Validation images: {files_val}")
print(f"Image size:        {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Max epochs:        {EPOCHS}")
print()

# =============================================================================
# Data Generators with Augmentation
# =============================================================================
print("[1/4] Preparing data generators...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,           # Smart Sampling: simulates different camera angles
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255  # No augmentation for validation
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Class indices: {train_generator.class_indices}")
print()

# =============================================================================
# Build Model - MobileNetV2 with Transfer Learning
# =============================================================================
print("[2/4] Building MobileNetV2 model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Deep Fine-Tuning: Unfreeze the top layers of the base model
base_model.trainable = True

# Freeze the early layers (e.g. up to layer 100) to retain general features
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print()

# =============================================================================
# Callbacks
# =============================================================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_OUTPUT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# =============================================================================
# Training
# =============================================================================
print("[3/4] Starting training...")
print()

steps_per_epoch = max(1, files_train // BATCH_SIZE)
validation_steps = max(1, files_val // BATCH_SIZE)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# Evaluation & F1-Score
# =============================================================================
print()
print("[4/4] Evaluating model...")

# Reset validation generator for full evaluation
val_generator.reset()

# Get predictions
y_pred_probs = model.predict(val_generator, steps=validation_steps)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes[:len(y_pred)]

# Calculate F1-score
try:
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\n{'='*60}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"{'='*60}")
    print()
    print("Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['empty (Disponible)', 'occupied (Ocupado)']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
except ImportError:
    print("[WARNING] scikit-learn not installed. Cannot calculate F1-score.")
    print("Install with: pip install scikit-learn")

# Final summary
print()
print("=" * 60)
print(f"Model saved to: {MODEL_OUTPUT}")
print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final val accuracy:   {history.history['val_accuracy'][-1]:.4f}")
print("=" * 60)
print("Training complete!")
