import numpy as np
import pandas as pd
from scipy.signal import find_peaks

WINDOW_SIZE = 30 * 64   # 30 seconds window @ 64 Hz
STEP = 15 * 64          # 50% overlap


def extract_hrv_features(ppg, label):
    rows = []
    for start in range(0, len(ppg) - WINDOW_SIZE, STEP):
        segment = ppg[start:start + WINDOW_SIZE]

        # peak detection
        peaks, _ = find_peaks(segment, distance=30)  # minimum 30 samples between peaks
        if len(peaks) < 2:
            continue

        # RR intervals (seconds)
        rr = np.diff(peaks) / 64.0

        # Heart Rate (bpm)
        hr = 60.0 / np.mean(rr)

        # HRV features
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
        sdnn = np.std(rr)
        pnn50 = np.mean((np.abs(np.diff(rr)) > 0.05).astype(float)) * 100

        rows.append([hr, rmssd, sdnn, pnn50, label])
    return rows


def process_all():
    calm = np.load("data/calm.npy")
    stress = np.load("data/stress.npy")

    rows = []
    rows += extract_hrv_features(calm, 0)      # label 0 = calm
    rows += extract_hrv_features(stress, 1)    # label 1 = stress

    df = pd.DataFrame(rows, columns=["HR", "RMSSD", "SDNN", "pNN50", "label"])
    df.to_csv("data/features.csv", index=False)

    print("Feature extraction complete ✔")
    print("Generated data/features.csv with shape:", df.shape)


if __name__ == "__main__":
    process_all()
