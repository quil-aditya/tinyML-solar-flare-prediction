# tinyML-solar-flare-prediction

# ☀️ Solar Flare Prediction Using Machine Learning & Embedded AI

This project develops a comprehensive machine learning pipeline to **predict solar flare occurrence and intensity** based on satellite sensor data. It also includes **embedded-ready neural networks (TFLite)** for deployment in space/weather-grade hardware.

---

## 🔍 Overview

Solar flares are sudden bursts of radiation from the sun that can disrupt satellites, GPS, and communications. Using NASA-style sensor features (like sunspot number, X-ray flux, and proton/electron fluences), this project:

- **Classifies whether a flare will occur**
- **Predicts the flare's intensity class (S, C, M, X)**
- **Estimates flare size on a logarithmic scale**
- **Optimizes AI models for embedded deployment (TFLite)**

---

## 📁 Dataset & Preprocessing

**Input Features:**
- `SUN SPOT NO`
- `SUNSPOT AREA`
- `X-RAY BACKGROUND FLUX`
- `ELECTRON FLUENCES`
- `PROTON FLUENCES (aggregated from 3 channels)`

**Target Variables:**
- `flare_occurrence` (binary classification)
- `flare_intensity` (multi-class: S, C, M, X)
- `flare_size` (log-scaled numerical value)

The dataset was cleaned, engineered, and saved as `preprocessed_solar_flare_data.csv`.

---

## 🧠 ML Models Used

### 🔹 Binary Classification (Flare Occurrence):
| Model | Accuracy |
|-------|----------|
| Random Forest | 86.99% |
| XGBoost | 86.86% |
| Naïve Bayes | 86.50% |
| SVM | 86.04% |
| TFLite Neural Net (Quantized) | 86.40% |

### 🔹 Multiclass Classification (Flare Intensity):
| Model | Accuracy |
|-------|----------|
| XGBoost | 83–86% |
| LightGBM | 84% |
| Decision Tree | ~81% |
| TFLite Neural Net | ~82% |
| Spiking Neural Net (SNN) | Experimental |

---

## 📊 Visualizations

- 📈 **Feature correlation** with flare occurrence (Spearman heatmap)
- 📉 **ROC & PR curves** for neural networks
- 💡 **Latency vs Accuracy trade-off** for models
- ⚡ **SEU (radiation error) resilience** of QNN vs SHARP CNN

All plots saved as high-res `.png` (e.g., `fig3_latency_vs_accuracy.png`).

---

## 🛰️ Embedded Deployment

- Converted trained neural networks to **TensorFlow Lite** format
- Applied **model quantization** for edge hardware
- Generated: `solar_flare_model.tflite`, `flare_intensity_model.tflite`

Ready for deployment on low-power devices (e.g., Raspberry Pi, Cortex-M, NVIDIA Jetson Nano).

---

## 🛠️ How to Run

### 🔗 Colab:
Run the full notebook here:  
[🔗 Open in Google Colab](https://colab.research.google.com/drive/1LBZKnrgHeHh-qdcnhE2Gmgr0mIzpHgYb)

### 🐍 Locally:
```bash
pip install -r requirements.txt
python solar_flare_pred_2_0.py
