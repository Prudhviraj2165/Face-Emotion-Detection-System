"""
train_model.py — Best-practice emotion recognition model trainer
================================================================
Uses MobileNetV2 as a pretrained backbone fine-tuned on FER-2013 data.
Expected FER-2013 directory structure:
  data/
    train/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
    test/
      angry/ ... (same classes)

Usage:
  1. Download FER-2013 from Kaggle:  https://www.kaggle.com/datasets/msambare/fer2013
  2. Extract to ./data/  (the folder above)
  3. Run:  python train_model.py
  4. The best model is saved as emotion_model.keras in the project root.
"""

import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR      = "data"
TRAIN_DIR     = os.path.join(DATA_DIR, "train")
TEST_DIR      = os.path.join(DATA_DIR, "test")
MODEL_OUT     = "emotion_model.keras"
IMG_SIZE      = (48, 48)          # FER-2013 native resolution
BATCH_SIZE    = 64
EPOCHS_HEAD   = 10                # Phase 1: train only the new head
EPOCHS_FINE   = 20                # Phase 2: fine-tune the top backbone layers
LR_HEAD       = 1e-3
LR_FINE       = 1e-5
CLASSES       = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES   = len(CLASSES)

# ── Verify data exists ────────────────────────────────────────────────────────
for d in [TRAIN_DIR, TEST_DIR]:
    if not os.path.isdir(d):
        print(f"[ERROR] Directory not found: {d}")
        print("  Please download FER-2013 and extract to ./data/  (see instructions above)")
        sys.exit(1)

print(f"[INFO] Using GPU: {len(tf.config.list_physical_devices('GPU'))} device(s)")
print(f"[INFO] TensorFlow version: {tf.__version__}")

# ── Data generators ───────────────────────────────────────────────────────────
# FER-2013 images are 48x48 greyscale; MobileNetV2 expects 96x96+ RGB.
# We upscale to 96x96 and convert to pseudo-RGB (triplicate channel).
MOBILENET_SIZE = (96, 96)

train_datagen = ImageDataGenerator(
    preprocessing_function=None,
    rescale=1.0 / 255.0,
    # Spatial augmentation
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.1,
    # Colour jitter (applied even on replicated channels — still helps)
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    # Reserve 10% of training images for validation
    validation_split=0.10,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=MOBILENET_SIZE,
    color_mode='rgb',           # MobileNetV2 needs 3 channels
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    subset='training',
    seed=SEED,
    shuffle=True,
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=MOBILENET_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    subset='validation',
    seed=SEED,
    shuffle=False,
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=MOBILENET_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False,
)

print(f"[INFO] Training samples   : {train_gen.n}")
print(f"[INFO] Validation samples : {val_gen.n}")
print(f"[INFO] Test samples       : {test_gen.n}")
print(f"[INFO] Class indices      : {train_gen.class_indices}")

# ── Class weights (handles class imbalance in FER-2013) ──────────────────────
labels = train_gen.classes
class_counts = np.bincount(labels)
total = sum(class_counts)
class_weights = {i: total / (NUM_CLASSES * count) for i, count in enumerate(class_counts)}
print(f"[INFO] Class weights: {class_weights}")

# ── Model architecture ────────────────────────────────────────────────────────
def build_model(trainable_backbone=False):
    base = MobileNetV2(
        input_shape=(*MOBILENET_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg',
    )
    base.trainable = trainable_backbone

    inputs = keras.Input(shape=(*MOBILENET_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # Label smoothing via this Dense — actual smoothing done in loss function
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='EmotionMobileNetV2')
    return model, base


# ── Phase 1: Train only the new classification head ──────────────────────────
print("\n[PHASE 1] Training classification head (backbone frozen) ...")
model, base = build_model(trainable_backbone=False)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LR_HEAD, weight_decay=1e-5),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy'],
)
model.summary()

cbs_phase1 = [
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_accuracy', verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=cbs_phase1,
)

# ── Phase 2: Fine-tune the top layers of MobileNetV2 ─────────────────────────
print("\n[PHASE 2] Fine-tuning top backbone layers ...")
# Unfreeze the top 40 layers of MobileNetV2
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LR_FINE, weight_decay=1e-6),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy'],
)

cbs_phase2 = [
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_accuracy', verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1),
    callbacks.TensorBoard(log_dir='./logs', histogram_freq=0),
]

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=cbs_phase2,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n[EVAL] Evaluating on test set ...")
loss, accuracy = model.evaluate(test_gen, verbose=1)
print(f"\n{'='*50}")
print(f"  Test Accuracy : {accuracy * 100:.2f}%")
print(f"  Test Loss     : {loss:.4f}")
print(f"{'='*50}")
print(f"\n[SUCCESS] Model saved to: {os.path.abspath(MODEL_OUT)}")
print("  Place this file in the project root and restart app.py")

# ── Per-class accuracy breakdown ─────────────────────────────────────────────
print("\n[INFO] Per-class metrics:")
test_gen.reset()
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes
for i, cls in enumerate(CLASSES):
    mask = y_true == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == i).mean()
        print(f"  {cls:10s}: {acc * 100:.1f}%  ({mask.sum()} samples)")
