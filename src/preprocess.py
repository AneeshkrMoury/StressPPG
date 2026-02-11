import numpy as np
import pandas as pd
import pickle
import os

DATA_PATH = "data/WESAD"
FS = 64       # BVP sampling rate (Hz)
SEC = 60      # 1 minute = 60 seconds


def parse_csv_segments(csv_path):
    """
    Parse the Sx_quest.csv file to extract Base and TSST timestamps.
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Identify lines
    order_line = next(l for l in lines if l.startswith("# ORDER"))
    start_line = next(l for l in lines if l.startswith("# START"))
    end_line = next(l for l in lines if l.startswith("# END"))

    # Remove '#' and newline → split by ';'
    order = [x for x in order_line.replace("#", "").strip().split(";")[1:] if x.strip() != ""]
    start_vals = [x for x in start_line.replace("#", "").strip().split(";")[1:] if x.strip() != ""]
    end_vals = [x for x in end_line.replace("#", "").strip().split(";")[1:] if x.strip() != ""]

    # Convert strings → float
    starts = list(map(float, start_vals))
    ends = list(map(float, end_vals))

    # Map segments:  { "Base": (start, end), "TSST": (start, end), ... }
    segments = dict(zip(order, zip(starts, ends)))
    return segments


def load_subject(subject_dir):
    """
    Extract calm (baseline) and stress (TSST) segments for one subject.
    """
    subject_name = os.path.basename(subject_dir)

    # Load BVP from .pkl
    with open(os.path.join(subject_dir, f"{subject_name}.pkl"), "rb") as f:
        data = pickle.load(f, encoding="latin1")
    bvp = np.ravel(data["signal"]["wrist"]["BVP"])

    # Load timestamps from Sx_quest.csv
    csv_path = os.path.join(subject_dir, f"{subject_name}_quest.csv")
    segments = parse_csv_segments(csv_path)

    # Extract segment times (in minutes)
    base_start, base_end = segments["Base"]
    tsst_start, tsst_end = segments["TSST"]

    # Convert minutes → samples
    calm_start = int(base_start * SEC * FS)
    calm_end   = int(base_end * SEC * FS)
    stress_start = int(tsst_start * SEC * FS)
    stress_end   = int(tsst_end * SEC * FS)

    calm = bvp[calm_start:calm_end]
    stress = bvp[stress_start:stress_end]
    return calm, stress


def process_all():
    """
    Process all subjects and save final calm/stress arrays.
    """
    calm_all = []
    stress_all = []

    for subject in os.listdir(DATA_PATH):
        if not subject.startswith("S"):     # avoid non-subject folders
            continue

        subject_dir = os.path.join(DATA_PATH, subject)
        print(f"Processing → {subject}")

        calm, stress = load_subject(subject_dir)
        calm_all.append(calm)
        stress_all.append(stress)

    calm_final = np.concatenate(calm_all)
    stress_final = np.concatenate(stress_all)

    print("\nFinal dataset sizes:")
    print("Calm samples:", calm_final.shape)
    print("Stress samples:", stress_final.shape)

    np.save("data/calm.npy", calm_final)
    np.save("data/stress.npy", stress_final)
    print("Saved calm.npy and stress.npy ✔")


if __name__ == "__main__":
    process_all()
