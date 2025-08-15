import numpy as np
import librosa

def extract_features_clean(data, sample_rate):
    try:
        result = []
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data), axis=1)  # 1
        rms = np.mean(librosa.feature.rms(y=data), axis=1)                # 1
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40), axis=1)  # 40
        chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate), axis=1)    # 12
        contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate), axis=1)  # 7
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate), axis=1)  # 6

        result.extend([zcr, rms, mfcc, chroma, contrast, tonnetz])
        return np.hstack(result)  # 67 chiều

    except Exception as e:
        print(f"Lỗi khi trích đặc trưng: {e}")
        return np.zeros(67)