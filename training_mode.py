import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Reshape, Permute
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- SET SEED FOR REPRODUCIBILITY ---
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed(42)

# --- CONFIGURATION ---
OUTPUT_DIR = 'PROCESSED_DATA'
MODEL_WEIGHTS_PATH = 'trained_rnn.weights.h5'
SPECTROGRAM_SHAPE = (128, 63, 1)
NUM_CLASSES = 1
INITIAL_LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32

# --- MODEL DEFINITION ---
def build_rnn_model(input_shape=SPECTROGRAM_SHAPE, num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape)
    x = Reshape((128, 63))(inputs)
    x = Permute((2, 1))(x)  # (Time, Features)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs, outputs)

# --- DATA LOADING ---
def load_data(output_dir):
    try:
        X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(output_dir, 'X_val.npy'))
        y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(output_dir, 'y_val.npy'))
        print(f"Loaded Training Data: X={X_train.shape}, y={y_train.shape}")
        return X_train, X_val, y_train, y_val
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None

# --- TRAINING ---
def train_model():
    X_train, X_val, y_train, y_val = load_data(OUTPUT_DIR)
    if X_train is None:
        return

    model = build_rnn_model()
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n📦 RNN Model Summary:")
    model.summary()

    # --- CALLBACKS ---
    callbacks = [
        ModelCheckpoint(MODEL_WEIGHTS_PATH, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # --- Print Accuracy per Epoch ---
    print("\n📊 --- Accuracy per Epoch ---")
    for epoch in range(len(history.history['accuracy'])):
        train_acc = history.history['accuracy'][epoch] * 100
        val_acc = history.history['val_accuracy'][epoch] * 100
        print(f"Epoch {epoch+1:02d}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

    # --- Final Evaluation ---
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print("\n📊 Final Validation Evaluation:")
    print(f"Final Validation Loss: {loss:.4f}")
    print(f"Final Validation Accuracy: {acc * 100:.2f}%")
    print("-" * 30)


if __name__ == '__main__':
    train_model()
