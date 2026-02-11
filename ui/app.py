import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import joblib
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from datetime import datetime
import io
import tempfile

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "models/stress_model.pkl"
FS = 64
WINDOW_SECONDS = 30
WINDOW_SIZE = WINDOW_SECONDS * FS

# -------------------------
# Helpers
# -------------------------
def load_model(path=MODEL_PATH):
    return joblib.load(path)

def synthesize_ppg(seconds=30, fs=64, hr_bpm=70, noise_level=0.02):
    """
    Produce a realistic-ish PPG waveform:
      - base sinusoid at heart rate
      - small variability in HR
      - Gaussian noise and small harmonics
    Returns 1D numpy array of samples (float)
    """
    t = np.arange(0, seconds, 1/fs)
    # vary HR slightly across time
    hr_var = hr_bpm + 1.5 * np.sin(0.1 * np.pi * t)  # slow variation
    inst_freq = hr_var / 60.0  # Hz
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    ppg = 0.6 * np.sin(phase)            # main pulse
    ppg += 0.08 * np.sin(2 * phase)      # harmonic
    ppg += 0.02 * np.random.randn(len(t)) * noise_level * 50
    # add baseline wander
    ppg += 0.02 * np.sin(0.02 * np.pi * t)
    # normalize to typical range [0,1]
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
    return ppg.astype(float)

def ppg_to_csv_bytes(ppg):
    s = "\n".join(f"{v:.6f}" for v in ppg)
    return s.encode()

def ppg_to_npy_bytes(ppg):
    buf = io.BytesIO()
    np.save(buf, ppg, allow_pickle=False)
    buf.seek(0)
    return buf.read()

# PDF generation (same polished version with footer)
def generate_pdf(state, emotion, stress_score, hr, rmssd, sdnn, pnn50, waveform_fig, hrtrend_fig):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    y = 810

    c.setFont("Helvetica-Bold", 15)
    c.drawString(55, y, "Stress Detection Using HR and HRV Features Extracted from PPG Signals")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(55, y, f"Date & Time: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    y -= 35

    c.setFont("Helvetica-Bold", 13)
    c.drawString(55, y, "Prediction Summary")
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(55, y, f"Model Result: {state}"); y -= 18
    c.drawString(55, y, f"Emotion Category: {emotion}"); y -= 18
    c.drawString(55, y, f"Stress Level Score: {stress_score:.2f}%"); y -= 30

    c.setFont("Helvetica-Bold", 13)
    c.drawString(55, y, "HRV Metrics"); y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(55, y, f"Heart Rate (HR): {hr:.2f} bpm"); y -= 18
    c.drawString(55, y, f"RMSSD: {rmssd:.4f}"); y -= 18
    c.drawString(55, y, f"SDNN: {sdnn:.4f}"); y -= 18
    c.drawString(55, y, f"pNN50: {pnn50:.2f}%"); y -= 30

    interpretation = {
        "CALM 😌": "The autonomic nervous system is balanced. Stress levels appear healthy.",
        "STRESSED 😣": "Low HRV and high HR suggest heightened sympathetic activation.",
        "🤩 Excited": "Elevated HR with high HRV — positive emotional arousal rather than stress.",
        "😔 Sad / Tired": "Lower HR & low HRV indicate reduced emotional energy.",
        "😐 Neutral / Mixed": "No strong emotional dominance — moderate physiological patterns."
    }.get(emotion, "")

    c.setFont("Helvetica-Bold", 13)
    c.drawString(55, y, "Interpretation")
    y -= 22
    c.setFont("Helvetica", 11)
    text = c.beginText(55, y)
    for line in interpretation.split(". "):
        if line.strip():
            text.textLine(line.strip())
    c.drawText(text)
    y -= 60

    # graphs (save to temp files)
    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    waveform_fig.savefig(tmp1.name, format="png", dpi=200)
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    hrtrend_fig.savefig(tmp2.name, format="png", dpi=200)

    c.drawImage(ImageReader(tmp1.name), 55, 330, width=480, height=170)
    c.drawImage(ImageReader(tmp2.name), 55, 130, width=480, height=170)

    c.setFont("Helvetica", 10)
    c.drawCentredString(300, 60, "Student: Aneesh Kumar Mourya   |   Guide: Dr. Namrata Nagpal")
    c.drawCentredString(300, 45, "Institution: Amity Institute of Information Technology, Lucknow")

    c.save()
    buffer.seek(0)
    return buffer

# -------------------------
# Load model
# -------------------------
model = load_model(MODEL_PATH)

# -------------------------
# Streamlit UI layout
# -------------------------
st.set_page_config(page_title="Stress Detection using HRV", page_icon="💓", layout="centered")
st.title("💓 Stress Detection using HR & HRV")

tab1, tab2 = st.tabs(["🎥 Video (Recommended)", "🔬 Advanced: PPG Upload"])

# -------------------------
# Tab 1: Video - recommended
# -------------------------
with tab1:
    st.markdown(
        "**Recommended:** Record a short video by covering your phone camera with your fingertip (flash ON). "
        "Hold still for 10–30 seconds."
    )

    video_file = st.file_uploader("Upload video (.mp4)", type=["mp4"], key="video_upload")
    if video_file:
        st.video(video_file)
        if st.button("Extract Pulse & Predict from Video"):
            import cv2
            from scipy.signal import butter, filtfilt

            def bandpass(sig, fs):
                # bandpass approx 0.7 - 4.0 Hz (42 - 240 bpm)
                b, a = butter(4, [0.7/(0.5*fs), 4.0/(0.5*fs)], btype="band")
                return filtfilt(b, a, sig)

            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(video_file.read())
            cap = cv2.VideoCapture(tmp.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30

            green = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # frame may be grayscale or color
                if frame.ndim == 3:
                    green_val = np.mean(frame[:, :, 1])
                else:
                    green_val = np.mean(frame)
                green.append(green_val)
            cap.release()

            if len(green) < 100:
                st.error("Video too short or frames could not be read reliably.")
            else:
                try:
                    ppg = bandpass(np.array(green), fps)
                except Exception as e:
                    st.error("Filtering failed (video noisy). Try better lighting / steady finger.")
                    st.exception(e)
                    ppg = None

                if ppg is not None:
                    peaks, _ = find_peaks(ppg, distance=max(4, int(fps/2)))
                    if len(peaks) < 2:
                        st.error("Unable to detect clear heartbeats. Try again with flash ON and stable finger.")
                    else:
                        rr = np.diff(peaks) / fps
                        hr = 60.0 / np.mean(rr)
                        rmssd = float(np.sqrt(np.mean(np.square(np.diff(rr)))))
                        sdnn = float(np.std(rr))
                        pnn50 = float(np.mean((np.abs(np.diff(rr)) > 0.05).astype(float)) * 100)

                        prob = model.predict_proba([[hr, rmssd, sdnn, pnn50]])[0]
                        pred = model.predict([[hr, rmssd, sdnn, pnn50]])[0]
                        p_calm = float(prob[0]) * 100
                        p_stress = float(prob[1]) * 100
                        state = "CALM 😌" if pred == 0 else "STRESSED 😣"
                        emotion = (
                            "😣 Stress / Anxiety" if (hr > 85 and (rmssd < 0.09 and sdnn < 0.09)) else
                            "😌 Calm / Relaxed" if (hr < 70 and (rmssd > 0.12 and sdnn > 0.12)) else
                            "🤩 Excited" if (hr > 85 and (rmssd > 0.12 and sdnn > 0.12)) else
                            "😔 Sad / Tired" if (hr < 70 and (rmssd < 0.09 and sdnn < 0.09)) else
                            "😐 Neutral / Mixed"
                        )
                        stress_score = p_stress

                        # figures
                        fig1, ax1 = plt.subplots()
                        ax1.plot(ppg); ax1.set_title("Extracted rPPG Waveform")
                        fig2, ax2 = plt.subplots()
                        ax2.plot(60/rr); ax2.set_title("Instantaneous HR (bpm)")

                        # save to session
                        st.session_state["results"] = {
                            "state": state, "emotion": emotion,
                            "p_calm": p_calm, "p_stress": p_stress, "stress_score": stress_score,
                            "hr": hr, "rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50,
                            "waveform_fig": fig1, "hrtrend_fig": fig2
                        }
                        st.success("Pulse extracted and prediction saved. Scroll down to view results / download PDF.")

# -------------------------
# Tab 2: Advanced PPG Upload
# -------------------------
with tab2:
    st.markdown("**Advanced Mode (for researchers / dataset owners).**")
    with st.expander("PPG Upload instructions & sample files (click to open)", expanded=True):
        st.markdown(
            "- The PPG file must be a **single column** of amplitude values (no header, no timestamps).\n"
            f"- Minimum length: **{WINDOW_SECONDS} seconds × {FS} Hz = {WINDOW_SIZE} samples**.\n"
            "- Accepted formats: **.csv** (single column) or **.npy** (NumPy 1D array).\n"
            "- CSV example: each line contains one float sample (e.g. `0.345678`).\n"
            "- Video mode is recommended for normal users — use PPG upload when you already have pre-collected waveforms."
        )
        # Sample generator (realistic noisy PPG)
        sample_ppg = synthesize_ppg(seconds=WINDOW_SECONDS, fs=FS, hr_bpm=72, noise_level=0.02)
        csv_bytes = ppg_to_csv_bytes(sample_ppg)
        npy_bytes = ppg_to_npy_bytes(sample_ppg)

        st.download_button("⬇ Download sample_ppg.csv", data=csv_bytes,
                           file_name="sample_ppg.csv", mime="text/csv")
        st.download_button("⬇ Download sample_ppg.npy", data=npy_bytes,
                           file_name="sample_ppg.npy", mime="application/octet-stream")

    # uploader
    ppg_file = st.file_uploader("Upload PPG (.csv or .npy)", type=["csv", "npy"], key="ppg_upload")
    if ppg_file:
        try:
            if ppg_file.name.endswith(".npy"):
                ppg = np.load(ppg_file)
            else:
                ppg = pd.read_csv(ppg_file, header=None).values.flatten()
            ppg = np.asarray(ppg, dtype=float)
        except Exception as e:
            st.error("Unable to read uploaded file. Ensure it's a single-column CSV or a NumPy .npy file.")
            st.exception(e)
            ppg = None

        if ppg is not None:
            if len(ppg) < WINDOW_SIZE:
                st.error(f"File too short — needs at least {WINDOW_SIZE} samples ({WINDOW_SECONDS}s at {FS}Hz).")
            else:
                ppg = ppg[:WINDOW_SIZE]
                peaks, _ = find_peaks(ppg, distance=30)
                if len(peaks) < 2:
                    st.error("Could not detect peaks in PPG signal — try a cleaner signal.")
                else:
                    rr = np.diff(peaks) / FS
                    hr = 60.0 / np.mean(rr)
                    rmssd = float(np.sqrt(np.mean(np.square(np.diff(rr)))))
                    sdnn = float(np.std(rr))
                    pnn50 = float(np.mean((np.abs(np.diff(rr)) > 0.05).astype(float)) * 100)

                    prob = model.predict_proba([[hr, rmssd, sdnn, pnn50]])[0]
                    pred = model.predict([[hr, rmssd, sdnn, pnn50]])[0]
                    p_calm = float(prob[0]) * 100
                    p_stress = float(prob[1]) * 100
                    state = "CALM 😌" if pred == 0 else "STRESSED 😣"
                    emotion = (
                        "😣 Stress / Anxiety" if (hr > 85 and (rmssd < 0.09 and sdnn < 0.09)) else
                        "😌 Calm / Relaxed" if (hr < 70 and (rmssd > 0.12 and sdnn > 0.12)) else
                        "🤩 Excited" if (hr > 85 and (rmssd > 0.12 and sdnn > 0.12)) else
                        "😔 Sad / Tired" if (hr < 70 and (rmssd < 0.09 and sdnn < 0.09)) else
                        "😐 Neutral / Mixed"
                    )
                    stress_score = p_stress

                    fig1, ax1 = plt.subplots(); ax1.plot(ppg); ax1.set_title("PPG Waveform")
                    fig2, ax2 = plt.subplots(); ax2.plot(60/rr); ax2.set_title("Instant HR (bpm)")

                    st.session_state["results"] = {
                        "state": state, "emotion": emotion,
                        "p_calm": p_calm, "p_stress": p_stress, "stress_score": stress_score,
                        "hr": hr, "rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50,
                        "waveform_fig": fig1, "hrtrend_fig": fig2
                    }
                    st.success("PPG processed and prediction saved. Scroll down to view results / download PDF.")

# -------------------------
# Always show results if available (survives reruns & PDF clicks)
# -------------------------
if "results" in st.session_state:
    st.markdown("---")
    r = st.session_state["results"]

    st.subheader(f"Prediction: **{r['state']}**")
    st.write(f"Calm Probability: {r['p_calm']:.2f}%")
    st.write(f"Stress Probability: {r['p_stress']:.2f}%")
    st.write(f"Emotion Category: **{r['emotion']}**")

    st.markdown("### 🧭 Physiological Stress Level")
    st.progress(int(r["stress_score"]))

    df = pd.DataFrame({
        "HR (bpm)": [r["hr"]],
        "RMSSD": [r["rmssd"]],
        "SDNN": [r["sdnn"]],
        "pNN50": [r["pnn50"]],
    })
    st.table(df)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(r["waveform_fig"])
    with col2:
        st.pyplot(r["hrtrend_fig"])

    pdf = generate_pdf(
        r["state"], r["emotion"], r["stress_score"],
        r["hr"], r["rmssd"], r["sdnn"], r["pnn50"],
        r["waveform_fig"], r["hrtrend_fig"]
    )
    st.download_button(
        "📄 Download Stress Report (PDF)",
        data=pdf,
        file_name="stress_report.pdf",
        mime="application/pdf"
    )
