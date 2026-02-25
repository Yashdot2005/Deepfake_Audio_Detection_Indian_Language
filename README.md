# Deepfake_Audio_Detection_Indian_Language

This project focuses on detecting deepfake (synthetically generated) audio clips across multiple Indian languages using machine learning techniques.

The system classifies audio clips as:

✅ Real

❌ Fake (Deepfake)

---

🌍 Supported Languages

The dataset includes audio samples from the following 4 Indian languages:

Hindi

Marathi

Punjabi

Bengali



---

📊 Dataset Details

Total Audio Clips: 800

Real Audio Clips: 400

Fake Audio Clips: 400

Languages: 4

Format: .wav or .mp3


Each language contains:

100 Real audio clips

100 Fake audio clips



---

📁 Dataset Structure

dataset/
│
├── hindi/
│   ├── real/
│   └── fake/
│
├── marathi/
│   ├── real/
│   └── fake/
│
├── punjabi/
│   ├── real/
│   └── fake/
│
├── bengali/
│   ├── real/
│   └── fake/

⚠️ Note: The dataset is not included in this repository due to size limitations.
Please download the real dataset from Kaggle and generate the fake samples as described above, then organize them as shown below.


After downloading, place the dataset folder inside the project directory.
---

⚙️ Technologies Used

Python

NumPy

Librosa

TensorFlow



---

🧠 Methodology

1. Audio preprocessing


2. Feature extraction (e.g., MFCC / Spectrogram)


3. Model training


4. Evaluation on test dataset




---

🚀 How to Run

1. Clone the repository


2. Install dependencies



pip install -r requirements.txt

3. Place dataset inside the dataset/ folder


4. Run training script



python train.py


---

📈 Objective

To build a robust deepfake audio detection system that works across multiple Indian languages and helps identify synthetic voice manipulation.


---

👥 Team Members

-Yash Ghotekar:Project Lead and Backened Development
-Sarang Channe: Frontened Development (UI Integration)
-Dikshant Fulzele: Dataset Collection 
-Vinit Guglot: Dataset Organization
-Gourav Khumbhare: Training
 

RESEARCH PAPER
This project is supported by a published research paper in the International Journal of Engineering Research & Technology (IJERT).
Title: Deepfake-audio Detection for Indian Language
Authors: Yash Anand Ghotekar, Dikshant Vilas Fulzele, Sarang Vinod Channe, Gourav Pramod Kumbhare, Vinit Venkanna Guglot, Dr. Manisha Pise
Published: IJERT, Volume 14, Issue 12, December 2025
DOI: 10.17577/IJERTV14IS120304 �
ijert.org
📄 You can read/download the full paper here:
🔗 https://www.ijert.org/deepfake-audio-detection-for-indian-language⁠
