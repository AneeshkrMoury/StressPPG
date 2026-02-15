# 🫀 StressPPG – PPG-Based Stress Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

StressPPG is a machine learning–based system for detecting mental stress using **Photoplethysmography (PPG)** and **Heart Rate Variability (HRV)** features.  
The project supports both **sensor-based PPG signals** and **video-based remote PPG (rPPG)** and provides an interactive **Streamlit web interface** for visualization, prediction, and report generation.



## 📌 Project Overview

Mental stress has a direct impact on the autonomic nervous system and is reflected in cardiovascular activity. Traditional stress assessment methods rely on self-reported questionnaires, which are subjective and unsuitable for real-time monitoring.

This project proposes an **objective, lightweight, and non-invasive stress detection approach** using:
- Heart Rate (HR)
- Heart Rate Variability (HRV)
- Machine Learning (Random Forest)

The system classifies physiological states into **Calm** and **Stressed** and further interprets them into higher-level emotional states.



## 🧠 What This Project Does

- Accepts **PPG signal files** (`.npy`, `.csv`)
- Extracts HR and HRV features
- Trains a **Random Forest classifier**
- Predicts stress level from unseen data
- Supports **video-based rPPG input**
- Displays results via a **Streamlit web app**
- Generates **visualizations and downloadable reports**



## 📊 Model Training Results

The model was trained using HR and HRV features extracted from PPG signals.

**Performance Summary:**

- **Accuracy:** 84.27%

**Confusion Matrix:**
-           [[205  24]
             [ 35 111]]


**Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Calm (0) | 0.854 | 0.895 | 0.874 |
| Stressed (1) | 0.822 | 0.760 | 0.790 |

The trained model is saved as:
 -   models/stress_model.pkl


## 🏗️ Project Structure
 ``` StressPPG/
│
├── data/
│   └── WESAD/
│       ├── features.csv
│       ├── stress.npy
│       └── video_ppg.npy
│
├── src/
│   ├── preprocess.py          # Signal preprocessing
│   ├── feature_extraction.py  # HR & HRV feature extraction
│   ├── train_model.py         # Model training
│   ├── video_to_ppg.py        # rPPG extraction from video
│   └── video_predict.py       # Video-based stress prediction
│
├── ui/
│   └── app.py                 # Streamlit web application
│
├── requirements.txt
├── LICENSE
└── README.md

 ```

## ▶️ How to Run the Project

### 1️ Clone the Repository
```bash
git clone https://github.com/AneeshkrMoury/StressPPG.git
cd StressPPG
```

### 2️ Create and Activate Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3️ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️ Train the Model
```bash
python src/train_model.py
```
### 5️ Run the Streamlit App
```bash
streamlit run ui/app.py
```
### The app will open in your browser at:
```bash
http://localhost:8501
```

##  Important Notes

- Camera-based rPPG is intended for demonstration purposes only
- Model training is performed exclusively on sensor-based PPG data
- Prediction accuracy is highly dependent on input signal quality


##  Future Improvements

- Multi-class stress level classification
- Deep learning–based feature learning
- Improved motion-robust rPPG extraction
- Mobile-friendly deployment
