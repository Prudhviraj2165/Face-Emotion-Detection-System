"""
download_model.py — Reconstruct FER-2013 model from weights
==========================================================
Manually builds the CNN architecture from the gitshanks/fer2013 repo
and loads the pre-trained weights.

Usage:
    python download_model.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

H5_PATH = "fer.h5"
OUTPUT_PATH = "emotion_model.keras"

def build_gitshanks_model():
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), name='conv2d_1'),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
        BatchNormalization(name='batch_normalization_1'),
        MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'),
        Dropout(0.5, name='dropout_1'),

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3'),
        BatchNormalization(name='batch_normalization_2'),
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_4'),
        BatchNormalization(name='batch_normalization_3'),
        MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'),
        Dropout(0.5, name='dropout_2'),

        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_5'),
        BatchNormalization(name='batch_normalization_4'),
        Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_6'),
        BatchNormalization(name='batch_normalization_5'),
        MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3'),
        Dropout(0.5, name='dropout_3'),

        # Block 4
        Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_7'),
        BatchNormalization(name='batch_normalization_6'),
        Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_8'),
        BatchNormalization(name='batch_normalization_7'),
        MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_4'),
        Dropout(0.5, name='dropout_4'),

        Flatten(name='flatten_1'),

        # Dense layers
        Dense(512, activation='relu', name='dense_1'),
        Dropout(0.4, name='dropout_5'),
        Dense(256, activation='relu', name='dense_2'),
        Dropout(0.4, name='dropout_6'),
        Dense(128, activation='relu', name='dense_3'),
        Dropout(0.5, name='dropout_7'),
        Dense(7, activation='softmax', name='dense_4')
    ])
    return model

print("[1/3] Building model architecture...")
model = build_gitshanks_model()
print(f"      Model built with {len(model.layers)} layers ✓")

print("\n[2/3] Loading weights from fer.h5...")
if not os.path.exists(H5_PATH):
    print(f"[ERROR] {H5_PATH} not found. Please run the script again.")
    # Attempt to download if missing
    import urllib.request
    urllib.request.urlretrieve("https://github.com/gitshanks/fer2013/raw/master/fer.h5", H5_PATH)

try:
    model.load_weights(H5_PATH)
    print("      Weights loaded successfully ✓")
except Exception as e:
    print(f"[ERROR] Could not load weights: {e}")
    sys.exit(1)

print("\n[3/3] Saving final model...")
model.save(OUTPUT_PATH)
print(f"      Saved as {OUTPUT_PATH} ✓")

# Cleanup
if os.path.exists(H5_PATH):
    os.remove(H5_PATH)
if os.path.exists("fer.json"):
    os.remove("fer.json")

print("\n" + "="*50)
print(f"  SUCCESS! Model ready at {os.path.abspath(OUTPUT_PATH)}")
print("="*50)
print("\n  Restart the app: python app.py")
