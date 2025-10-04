# Vibration-Based Fault Detection System

A comprehensive machine learning pipeline for detecting mechanical faults in rotating machinery using video-based vibration analysis. This project achieves **99% accuracy** across multiple approaches including frequency-domain analysis and vision-based deep learning models.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Output Data](#output-data)
- [Technical Details](#technical-details)
- [Requirements](#requirements)
- [License](#license)

## Overview

This project implements two complementary approaches for mechanical fault detection:

1. **Frequency-Domain Analysis Pipeline**: Extracts vibration signals from video using optical flow and Eulerian motion magnification, then analyzes frequency characteristics
2. **Vision-Based Deep Learning Pipeline**: Directly classifies fault conditions from processed video frames using state-of-the-art computer vision models

Both approaches achieve **~99% accuracy** in classifying:
-  Normal operation
-  Outer ring bearing fault
-  Unbalance (10g)
-  Unbalance (37g)

## Features

### Video Processing (`Approach 1 - Frequency Parameters Training/PreProcessing.py.py`)
- **Eulerian Motion Magnification**: Amplifies subtle vibrations invisible to the naked eye
- **Optical Flow Analysis**: Extracts displacement time series from video
- **ROI Selection**: Focus analysis on specific machine components
- **Multi-Clip Processing**: Analyzes videos in overlapping temporal windows

### Feature Extraction
**Frequency Domain Features:**
- Total Energy
- Spectral Centroid & Bandwidth
- Spectral Entropy
- Dominant Frequency & Amplitude

**Time Domain Features:**
- Statistical moments (Mean, Std Dev, Variance)
- Skewness & Kurtosis
- Zero Crossing Rate
- Peak-to-Peak amplitude

### Classification Models

#### Frequency-Based Approach (`Approach 1 - Frequency Parameters Training/model.py`)
- **Random Forest Classifier** with optimized hyperparameters
- Input: 14 frequency and time-domain features
- Achieves 99% accuracy on test set

#### Vision-Based Approach (PDM Notebook)
Multiple state-of-the-art architectures tested:
- **Random Forest** (baseline)
- **Vision Transformer (ViT)**
- **Swin Transformer**
- **ConvNeXt**

All models achieve average **~99% accuracy** 🎉

## 📁 Project Structure

```
├── 
├── Approach 1 - Frequency Parameters Training/                   # Generated videos, plots, and CSVs
│   ├── Preprocessing.py                    # Video processing & feature extraction
│   ├── model.py                   # Random Forest training on frequency features
├── Approach 2 - ROI Image Training/              # Training data organized by class
│   ├── EulerMag_Optical_Flow_Freq_Analysis.ipynb
│   ├── pdm-transformer-project.ipynb
└── README.md
```

## Installation

### Prerequisites
```bash
Python 3.8+
OpenCV
NumPy, Pandas, Matplotlib, Seaborn
scikit-learn
scipy
joblib
PyTorch (for deep learning models)
timm (PyTorch Image Models)
```

### Install Dependencies
```bash
pip install opencv-python numpy pandas matplotlib seaborn
pip install scikit-learn scipy joblib
pip install torch torchvision torchaudio
pip install timm
```

## Usage

### Step 1: Extract Vibration Features from Video
```python
# Edit VIDEO_PATH in test.py
VIDEO_PATH = "path/to/your/video.MOV"

# Run feature extraction
python Preprocessing.py
```

This will:
1. Prompt you to select ROI on first frame
2. Process video in 20-second clips with 5-second stride
3. Generate enhanced videos, optical flow visualizations, and CSV files
4. Save spectrograms and frequency analysis plots

### Step 2: Train Frequency-Based Classifier
```python
# Organize your CSV files into folders by class:
# Second Batch/Normal/*.csv
# Second Batch/Outer Ring/*.csv
# Second Batch/10g/*.csv
# Second Batch/37g/*.csv

# Run training
python model.py
```

Outputs:
- `random_forest_model.pkl` (trained model)
- `confusion_matrix.png`
- Classification report in console

### Step 3: Train Vision-Based Models
```python
# Open PDM_Notebook.ipynb in Jupyter/Colab
# Configure data paths and run cells
# Models tested: RF, ViT, Swin Transformer, ConvNeXt
```

## Results

### Model Performance Summary

| Approach | Model | Accuracy | Notes |
|----------|-------|----------|-------|
| Frequency | Random Forest | ~99% | 200 estimators, max_depth=10 |
| Vision | Random Forest | ~99% | Baseline on image features |
| Vision | Vision Transformer | ~99% | Transfer learning |
| Vision | Swin Transformer | ~99% | Hierarchical attention |
| Vision | ConvNeXt | ~99% | Modern CNN architecture |

### Sample Confusion Matrix
All models show strong diagonal patterns with minimal misclassification across all four fault categories.

## Output Data

All processed results are available on Google Drive:

**🔗 [Download Complete Approach 1 - Results]((https://drive.google.com/file/d/1EVzg_uigl8PW_WhsY_WVXukThkvSnCd8/view?usp=sharing))**
**🔗 [Download Complete Approach 2 - Results]((https://drive.google.com/drive/folders/1QmBHUpVfNfE-_thFkHcxBzosBTbzR9Ej?usp=drive_link))**

Contents:
- Enhanced video outputs (Eulerian magnification)
- Optical flow visualizations
- Extracted frequency features (CSV)
- Trained model checkpoints
- Confusion matrices and performance plots
- Raw and processed image datasets

## 🔬 Technical Details

### Eulerian Motion Magnification
```python
alpha = 30          # Magnification factor
filter_size = 5     # Gaussian kernel size
lambda_ = 5         # Spatial wavelength cutoff
```

### Optical Flow Parameters
```python
pyr_scale = 0.5
levels = 3
winsize = 15
iterations = 3
poly_n = 5
poly_sigma = 1.2
```

### Random Forest Hyperparameters
```python
n_estimators = 200
max_depth = 10
random_state = 42
test_size = 0.2
```

### Processing Pipeline
1. **Video Capture** → Resize to 640×360
2. **Grayscale Conversion** → ROI Selection
3. **Eulerian Magnification** → Enhance subtle motion
4. **Optical Flow** → Extract displacement vectors
5. **FFT Analysis** → Frequency domain features
6. **Feature Engineering** → 14 statistical & spectral features
7. **Classification** → RF/ViT/Swin/ConvNeXt

## Requirements

**Hardware Recommendations:**
- GPU with 8GB+ VRAM (for deep learning models)
- 16GB+ RAM (for video processing)
- SSD storage (for fast I/O)

**Software:**
```
opencv-python>=4.5.0
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.6.0
torch>=1.9.0
torchvision>=0.10.0
timm>=0.4.12
joblib>=1.0.0
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Real-time processing pipeline
- Additional fault types
- Model optimization and pruning
- Mobile deployment
- Web interface for inference

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
