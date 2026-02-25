import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Reshape, Permute

# --- CONFIGURATION ---
OUTPUT_DIR = 'PROCESSED_DATA'
MODEL_WEIGHTS_PATH = 'trained_rnn.weights.h5'
SPECTROGRAM_SHAPE = (128, 63, 1)
NUM_CLASSES = 1

# --- MODEL DEFINITION ---
def build_rnn_model(input_shape=SPECTROGRAM_SHAPE, num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape)
    x = Reshape((128, 63))(inputs)
    x = Permute((2, 1))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs, outputs)

# --- LOAD VALIDATION DATA ---
def load_data(output_dir):
    X_val = np.load(os.path.join(output_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(output_dir, 'y_val.npy'))
    print(f"Loaded Validation Data: X={X_val.shape}, y={y_val.shape}")
    return X_val, y_val

# --- EVALUATION ---
def evaluate_model():
    X_val, y_val = load_data(OUTPUT_DIR)
    
    model = build_rnn_model()
    model.load_weights(MODEL_WEIGHTS_PATH)
    print("✅ Model loaded successfully for evaluation.")

    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"\n📊 Validation Loss: {loss:.4f}")
    print(f"📊 Validation Accuracy: {acc * 100:.2f}%")

if __name__ == '__main__':
    evaluate_model()
