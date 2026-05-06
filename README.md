# Brain Tumor Multi-Class Classification Using EfficientNet with Grad-CAM Explainability

A deep learning-based medical imaging project for **multi-class brain tumor classification** using MRI scans and **EfficientNet architectures** with **Grad-CAM explainability**. This project compares the performance of **EfficientNetB0** and **EfficientNetB3** for classifying brain tumors into four categories while also providing interpretable visual explanations for predictions.

---

## 📌 Project Overview

Brain tumor classification from MRI images is an important task in medical diagnosis. Manual analysis of MRI scans can be time-consuming and may vary depending on expert interpretation. This project uses **transfer learning** with EfficientNet models to automate tumor classification and improve diagnostic support.

The system classifies MRI images into the following categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

To improve interpretability, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is integrated to highlight image regions influencing the model’s predictions.

---

# 🚀 Features

- Multi-class MRI brain tumor classification
- Transfer learning using EfficientNetB0 and EfficientNetB3
- Grad-CAM visualization for explainability
- Bounding box localization of important regions
- Data augmentation for better generalization
- Fine-tuning with pretrained ImageNet weights
- Class-balanced training using class weights
- Early stopping and learning rate scheduling
- Functional API implementation for Grad-CAM compatibility

---

# 🧠 Model Architectures

## EfficientNetB0
- ~4.05 Million parameters
- Lightweight and computationally efficient
- Achieved best overall performance

## EfficientNetB3
- ~10.78 Million parameters
- Higher representational capacity
- Slightly lower validation accuracy than B0

Both models use:
- GlobalAveragePooling2D
- BatchNormalization
- Dense layers with ReLU activation
- Dropout regularization
- Softmax output layer for classification

---

# 📂 Dataset

The dataset used is a publicly available **Brain Tumor MRI Dataset** from Kaggle.

Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Classes
- Glioma
- Meningioma
- Pituitary
- No Tumor

### Dataset Size
- ~5,600 training images
- ~1,600 testing images

### Image Size
- Resized to **224 × 224**

---

# 🔄 Data Preprocessing & Augmentation

The following preprocessing steps were applied:

- Image resizing
- EfficientNet preprocessing
- Normalization

### Data Augmentation
- Rotation
- Zoom
- Horizontal flip
- Width/height shifts
- Brightness adjustment
- Shear transformation

These techniques help reduce overfitting and improve generalization.

---

# 🔥 Grad-CAM Explainability

Grad-CAM is used to visualize the regions influencing model predictions.

### Pipeline
1. Image preprocessing
2. Model prediction
3. Grad-CAM heatmap generation
4. Heatmap overlay
5. Bounding box extraction
6. Visualization output

### Purpose
- Improve model interpretability
- Highlight tumor regions
- Provide visual support for predictions

> Note: Grad-CAM provides approximate localization, not exact tumor segmentation.

---

# 📊 Experimental Results

| Metric | EfficientNetB0 | EfficientNetB3 |
|---|---|---|
| Validation Accuracy | 96.07% | 95.45% |
| Validation Loss | 0.1212 | 0.1253 |
| Training Accuracy | 97.50% | 97.06% |
| Parameters | ~4.05M | ~10.78M |

### Observations
- EfficientNetB0 achieved slightly better performance.
- Pituitary and No Tumor classes achieved highest F1-scores.
- Glioma classification remained the most challenging task.

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Grad-CAM
- EfficientNet

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Brain_Tumor_Classification.git
cd Brain_Tumor_Classification
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---


# 📈 Future Improvements

- Integration of CBAM attention mechanism
- Ensemble learning using multiple EfficientNet variants
- 3D MRI volume processing
- Multi-modal MRI support
- Better generalization on external datasets
- Improved tumor localization using segmentation models

---

# 👨‍💻 Author

**Kartikey Soni**  
B.Tech Artificial Intelligence  
MITS Gwalior

---

# 🙏 Acknowledgement

Special thanks to:

- **Ms. Geetika Sharma Hazra** for guidance and supervision
- Centre for Artificial Intelligence, MITS Gwalior
