import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape, Permute
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---------------- CONFIGURATION ----------------
OUTPUT_DIR = 'PROCESSED_DATA'
MODEL_WEIGHTS_PATH = 'trained_rnn.weights.h5'  # Your RNN/LSTM weights
SPECTROGRAM_SHAPE = (128, 63, 1)
NUM_CLASSES = 1
THRESHOLD = 0.5  # Sigmoid threshold for binary classification

# ---------------- MODEL DEFINITION ----------------
def build_rnn_model(input_shape=SPECTROGRAM_SHAPE, num_classes=NUM_CLASSES):
    """RNN (LSTM) model for audio spectrogram classification"""
    inputs = Input(shape=input_shape)
    x = Reshape((128, 63))(inputs)  # Remove channel dimension
    x = Permute((2, 1))(x)          # Time steps first
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs, outputs)

# ---------------- DATA LOADING ----------------
def load_validation_data(output_dir=OUTPUT_DIR):
    """Load preprocessed validation data"""
    try:
        X_val = np.load(os.path.join(output_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(output_dir, 'y_val.npy'))
        print(f"✅ Validation data loaded: X={X_val.shape}, y={y_val.shape}")
        return X_val, y_val
    except Exception as e:
        print(f"❌ Failed to load validation data: {e}")
        return None, None

# ---------------- EVALUATION ----------------
def evaluate_rnn_model():
    # 1️⃣ Load data
    X_val, y_val = load_validation_data()
    if X_val is None:
        return

    # 2️⃣ Load model
    model = build_rnn_model()
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("✅ RNN/LSTM model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        return

    # Compile (required for .evaluate)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    # 3️⃣ Evaluate loss and accuracy
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n📊 Validation Loss: {loss:.4f}")
    print(f"📊 Validation Accuracy: {acc*100:.2f}%")

    # 4️⃣ Predict classes
    y_pred_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

    # 5️⃣ Detailed metrics
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print("\n🔬 Classification Metrics:")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print(f"\nConfusion Matrix:\n{cm}")

    # 6️⃣ Full classification report
    report = classification_report(
        y_val, y_pred, target_names=['REAL (0)', 'FAKE (1)'], digits=4
    )
    print("\n📋 Detailed Classification Report:\n")
    print(report)

# ---------------- MAIN ----------------
if __name__ == '__main__':
    evaluate_rnn_model()
