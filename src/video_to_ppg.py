import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def butter_bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=30, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def extract_ppg_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fs = cap.get(cv2.CAP_PROP_FPS)
    green_signal = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame  # full frame for now
        green = np.mean(roi[:, :, 1])  # green channel intensity
        green_signal.append(green)

    cap.release()
    green_signal = np.array(green_signal)

    # filter to remove noise
    filtered = butter_bandpass_filter(green_signal, fs=int(fs))
    return filtered

if __name__ == "__main__":
    video_path = "data/sample.mp4"   # change if your path is different
    ppg = extract_ppg_from_video(video_path)
    print("PPG extracted length:", len(ppg))

    # show waveform to check pulse-like shape
    plt.plot(ppg)
    plt.title("Extracted rPPG from Video")
    plt.show()

    np.save("data/video_ppg.npy", ppg)
    print("Saved PPG waveform → data/video_ppg.npy")
