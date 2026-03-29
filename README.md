# 🎙️ Speech Emotion Recognition

A Vietnamese-language web application for real-time speech emotion analysis using a Convolutional Emotion Neural Network (CeNN). Upload a `.wav` file and get instant sentiment classification with confidence scoring.

---

## ✨ Features

- **Audio upload** — drag-and-drop or browse `.wav` files
- **Emotion detection** — classifies audio as **Positive** (happy / neutral) or **Negative** (angry / sad / fearful)
- **Confidence scoring** — returns prediction percentage
- **Audio preview** — built-in playback before analysis
- **Streamlit UI** — pastel blue-purple gradient, frosted-glass design, fully in Vietnamese

---

## 🧠 Model

### Architecture
- **CeNN** (Convolutional Emotion Neural Network)
- Input: Mel-spectrogram of shape `(N_MELS, MAX_FRAMES, 3)`
- Output: Binary classification — **Positive** vs **Negative** sentiment

### Training Datasets
| Dataset | Emotion Encoding |
|---------|-----------------|
| **EmoDB** | Filename codes: W=angry, L=boredom, E=disgust, A=fear, F=happy, T=sad, N=neutral |
| **Crema** | 3-letter codes: ANG, DIS, FEA, HAP, SAD, NEU |
| **SAVEE** | Single-letter codes: a, d, f, h, n, sa, su |
| **TESS** | Directory/filename structure |

All datasets are filtered to 5 emotions → binary labels:
- **Positive**: happy, neutral
- **Negative**: angry, sad, fear

### Training Config
| Parameter | Value |
|-----------|-------|
| Batch size | 12 |
| Max epochs | 50 (early stopping) |
| Train/Val split | 80% / 20% stratified |
| Early stopping patience | 5 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5) |

### Audio Feature Extraction (`utils.py`)
67-dimensional feature vector per audio file:

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| Zero-Crossing Rate | 1 | Signal sign-change rate |
| RMS Energy | 1 | Overall loudness |
| MFCCs | 40 | Perceptual mel-cepstral coefficients |
| Chroma | 12 | Energy across 12 pitch classes |
| Spectral Contrast | 7 | Spectrum peak/valley differences |
| Tonnetz | 6 | Tonal centroid / harmonic features |

---

## 📁 Project Structure

```
SpeechRecognition/
├── app/
│   ├── streamlit_voice_sentiment_app.py   # Main Streamlit app
│   ├── train_model.py                     # Model training pipeline
│   ├── modelcenn.py                       # CeNN architecture
│   ├── modelcnn.py                        # CNN architecture (alt)
│   ├── utils.py                           # Audio feature extraction
│   ├── scaler.pkl                         # Fitted feature scaler
│   ├── requirements.txt                   # Python dependencies
│   └── training_history.csv              # Training metrics log
└── emotion_model.keras                    # Pre-trained model weights
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/fong-AI/SpeechRecognition.git
cd SpeechRecognition/app
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run streamlit_voice_sentiment_app.py
```

Then open your browser at `http://localhost:8501`.

---

## 📦 Dependencies

```
streamlit
tensorflow==2.13.0
librosa
joblib
numpy
scikit-learn
h5py==3.8.0
```

---

## 🔄 Workflow

```
Upload .wav file
       ↓
Feature extraction (67-dim vector via librosa)
       ↓
Scale features (scaler.pkl)
       ↓
CeNN model inference (emotion_model.keras)
       ↓
Binary prediction + confidence %
       ↓
Display result (Positive / Negative)
```

---

## 📊 Model Performance

Training history is logged in `app/training_history.csv`. Best validation accuracy achieved: **81.50%** (reflected in model filename `cnn_rpla_emotion_8150_v2.keras`).

---

## 🛠️ Retrain the Model

To retrain from scratch with your own datasets:

```bash
cd app
python train_model.py
```

> Datasets (EmoDB, Crema, SAVEE, TESS) must be downloaded and paths configured in `train_model.py`.

---

## 📝 Notes

- Only `.wav` format is supported for inference
- The app patches TensorFlow's `LegacyInputLayer` for compatibility with TF 2.13
- All UI text is in Vietnamese

---

## 📄 License

This project is for personal / educational use.
