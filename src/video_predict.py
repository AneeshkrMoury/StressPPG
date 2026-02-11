import numpy as np
from scipy.signal import find_peaks
import joblib

FS = 30    # video sampling rate we estimated earlier
WINDOW = 30 * FS
model = joblib.load("models/stress_model.pkl")

def compute_features(ppg):
    peaks, _ = find_peaks(ppg, distance=6)  # ~ 180 bpm max
    if len(peaks) < 2:
        return None

    rr = np.diff(peaks) / FS
    hr = 60 / np.mean(rr)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    sdnn = np.std(rr)
    pnn50 = np.mean((np.abs(np.diff(rr)) > 0.05).astype(float)) * 100

    return [hr, rmssd, sdnn, pnn50]


if __name__ == "__main__":
    ppg = np.load("data/video_ppg.npy")

    # Use first 30 seconds only
    ppg = ppg[:WINDOW]

    features = compute_features(ppg)
    if features is None:
        print("Could not extract features from video.")
    else:
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0]

        state = "CALM 😌" if pred == 0 else "STRESSED 😣"
        print("\nPrediction:", state)
        print("Confidence:", round(prob[pred] * 100, 2), "%")
