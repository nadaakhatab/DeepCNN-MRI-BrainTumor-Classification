# 🧠 NeuroScan AI - Brain Tumor Detection

A powerful **Brain Tumor Detection System** built with a custom Deep Convolutional Neural Network (CNN) from scratch and a modern web interface.

## Project Overview

**NeuroScan AI** is an intelligent system that classifies brain MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **No Tumor** (Healthy)
- **Pituitary Tumor**

The project consists of two main parts:
1. **Deep Learning Model** (PyTorch - Built from scratch)
2. **Web Application** (FastAPI + Modern HTML/JS Frontend)

---

## ✨ Features

- **Custom CNN Architecture** built from scratch (No pre-trained models)
- High accuracy: **~92%** on test set
- Data augmentation and proper validation
- FastAPI backend with REST API
- Beautiful, responsive web interface with drag & drop support
- Real-time prediction with confidence scores
- Easy-to-use inference script
- Fully local deployment (no internet required after setup)

---

## Model Architecture

**DeepCNN** (Custom Architecture):

- 5 Convolutional Blocks with Batch Normalization
- Adaptive Average Pooling
- Dropout regularization (0.4 & 0.3)
- Fully connected classifier
- Input size: `128x128x3`

### Training Details
- **Framework**: PyTorch
- **Optimizer**: Adam (lr=0.0005)
- **Loss**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau
- **Epochs**: 50
- **Batch Size**: 32
- **Image Size**: 128x128

**Final Test Performance:**
- **Accuracy**: 92%
- **Macro Avg F1-Score**: 0.92

---

## 🚀 How to Run the Project

### 1. Clone / Open Project
```bash
cd Brain_tumor
```

### 2. Activate Virtual Environment
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
uvicorn app:app --reload --port 8000
```

### 5. Open in Browser
Go to: **[http://localhost:8000](http://localhost:8000)**

---

## 📁 Project Structure

```
Brain_tumor/
├── app.py                 # FastAPI backend
├── front.html             # Modern frontend
├── model.pth              # Trained model weights
├── classes.json           # Class labels
├── inference.py           # Standalone inference script
├── requirements.txt
├── brain2.ipynb           # Training notebook
├── brain2.py              # Training script
└── README.md
```


## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, torchvision
- **Backend**: FastAPI
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: OpenCV, Pillow, NumPy, pandas
- **Visualization**: Matplotlib, Seaborn

---

## 📊 Dataset

- **Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: 4 categories
- **Total Images**: ~7000 MRI scans

