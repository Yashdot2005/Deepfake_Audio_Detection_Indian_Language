from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Reshape, Permute

# --- CONFIG ---
SPECTROGRAM_SHAPE = (128, 63, 1)
SAMPLE_RATE = 16000
DURATION_SECONDS = 2
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(BASE_DIR, 'trained_rnn.weights.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- MODEL ---
MODEL = None

def build_rnn_model(input_shape=SPECTROGRAM_SHAPE, num_classes=1):
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

def load_model():
    global MODEL
    if not os.path.exists(WEIGHTS_FILE):
        print(f"❌ Weights file not found at {WEIGHTS_FILE}")
        return
    try:
        MODEL = build_rnn_model()
        MODEL.load_weights(WEIGHTS_FILE)
        MODEL.predict(np.zeros((1, 128, 63, 1)), verbose=0)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        MODEL = None

load_model()

# --- AUDIO PREPROCESSING ---
def preprocess_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION_SECONDS, mono=True)

        if np.max(np.abs(audio)) > 0:
            audio = librosa.util.normalize(audio)

        target_len = int(SAMPLE_RATE * DURATION_SECONDS)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
        else:
            audio = audio[:target_len]

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        current_width = log_mel_spec.shape[1]
        if current_width < SPECTROGRAM_SHAPE[1]:
            log_mel_spec = np.pad(
                log_mel_spec,
                ((0, 0), (0, SPECTROGRAM_SHAPE[1] - current_width)),
                'constant'
            )
        else:
            log_mel_spec = log_mel_spec[:, :SPECTROGRAM_SHAPE[1]]

        features = np.expand_dims(log_mel_spec, axis=-1)
        features = np.expand_dims(features, axis=0)
        return features

    except Exception as e:
        print(f"❌ Preprocessing Error: {e}")
        return None

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_audio():
    global MODEL
    if MODEL is None:
        load_model()
        if MODEL is None:
            return jsonify({"error": "Server Error: Model failed to load."}), 500

    if 'audio_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(file_path)

    try:
        features = preprocess_audio(file_path)
        if features is None:
            return jsonify({"error": "Processing failed"}), 500

        score = float(MODEL.predict(features, verbose=0)[0][0])

        if score > 0.5:
            result = "🚨 DEEPFAKE"
            accuracy = f"{score * 100:.2f}%"
        else:
            result = "✅ REAL"
            accuracy = f"{(1 - score) * 100:.2f}%"

        return jsonify({
            "filename": audio_file.filename,
            "result": result,
            "score": score,
            "accuracy": accuracy
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
