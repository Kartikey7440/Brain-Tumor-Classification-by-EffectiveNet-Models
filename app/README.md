# 🧠 NeuroScan AI — Brain Tumor Detection
### Streamlit Web Application · 4th Semester Macro Project

---

## 📁 Project Structure

```
brain_tumor_streamlit/
│
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
├── best_brain_tumor_b0.keras ← (Place your trained model here)  ← PRIMARY
├── best_brain_tumor_b3.h5   ← (Place your B3 model here)        ← OPTIONAL
└── best_glioma_model.h5     ← (Fallback model)                  ← OPTIONAL
```

---

## ⚡ Quick Setup (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Copy your trained model
Copy your saved model file into the same folder as `app.py`:

```bash
# From your project folder (Macro project 4th sem/)
cp best_brain_tumor_b0.keras  brain_tumor_streamlit/
# OR
cp best_brain_tumor_b3.h5    brain_tumor_streamlit/
# OR
cp best_glioma_model.h5      brain_tumor_streamlit/
```

> ⚠️ If no model file is found, the app will load ImageNet pretrained weights
> (architecture only, predictions will be random — place your .keras file!)

### Step 3 — Run the app
```bash
cd brain_tumor_streamlit
streamlit run app.py
```

The app will open at → **http://localhost:8501**

---

## 🔬 Features

| Feature | Description |
|--------|------------|
| **Multi-class Classification** | Glioma · Meningioma · Pituitary · No Tumor |
| **Model Selection** | Switch between EfficientNetB0 and EfficientNetB3 in sidebar |
| **Grad-CAM Heatmap** | Visual explanation of model attention regions |
| **Bounding Box** | Automatic tumor localization from Grad-CAM |
| **Probability Chart** | Per-class confidence bar chart |
| **Clinical Info** | Tumor descriptions and recommended actions |
| **Configurable** | Adjust heatmap opacity, colormap, bbox threshold |

---

## 📊 Model Performance

| Model | Val Accuracy | Val Loss | Parameters |
|-------|-------------|----------|-----------|
| EfficientNetB0 | **96.07%** | 0.1212 | ~4.05M |
| EfficientNetB3 | 95.45% | 0.1253 | ~10.78M |

Training: 25 epochs · Adam (lr=1e-4) · Balanced class weights · EarlyStopping

---

## 🖥️ How to Use (For Professors)

1. Run the app with `streamlit run app.py`
2. Select model in the **left sidebar** (B0 or B3)
3. Upload a brain MRI image (JPG/PNG) in the **Upload** section
4. Click **"🔬 Analyze MRI Scan"**
5. View results across 3 tabs:
   - **📊 Probabilities** — confidence scores for all 4 classes
   - **🔥 Grad-CAM** — heatmap + bounding box visualization
   - **📋 Clinical Info** — tumor description and recommendations

---

## 🗂️ Dataset

- **Training**: 5,604 images (4 classes)
- **Testing**: 1,604 images (4 classes)
- **Classes**: Glioma (0) · Meningioma (1) · Pituitary (2) · No Tumor (3)
- **Image Size**: 224×224 px (center-cropped from original)

---

## ⚕️ Disclaimer

This application is developed for **academic and research purposes** as part of a 4th semester macro project. It is **not** a substitute for professional medical diagnosis. Always consult a qualified medical professional for clinical decisions.

---

*Built with TensorFlow 2.20 · Streamlit · OpenCV · Matplotlib*
