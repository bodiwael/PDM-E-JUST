# 🎥 Video-Based Vibration Analysis & Signal Extraction Pipeline

This repository implements a **video-based vibration analysis system** using computer vision techniques like **Eulerian Magnification** and **Dense Optical Flow**. The system extracts subtle periodic vibrations from video, processes them into time series, and performs full **signal processing & frequency domain analysis**, including:

- Eulerian magnification
- Optical flow magnitude extraction
- Signal smoothing
- FFT and Spectrogram analysis
- Statistical feature extraction
- Visualization & CSV export

---

## 📦 Features

- 🎞 ROI-based vibration extraction from videos
- 🔍 Eulerian motion magnification
- 🔁 Dense Optical Flow-based motion tracking
- 📈 Time series smoothing & analysis
- 🔊 Frequency domain analysis (FFT, Spectrogram, Welch)
- 🧠 Signal features: Spectral Centroid, Entropy, ZCR, Kurtosis, etc.
- 📊 Data & plot export as CSV and PNG
- 💻 Fully modular pipeline

---

## 🧰 Dependencies

Install the following Python packages before running:

```bash
pip install opencv-python numpy matplotlib pandas scipy scikit-learn
```

---

## 📁 Directory Structure

```
.
├── Outputs/                     # All output videos, plots, and CSVs
├── vibration_analysis.py       # Main pipeline script
├── utils/                      # (optional) Utility functions if modularized
└── README.md                   # This documentation
```

---

## 🚀 How It Works

### 1. Select ROI

```python
get_roi_from_video(video_path)
```
User manually selects the **Region of Interest (ROI)** for vibration tracking in the first frame. This ROI is used to crop each frame.

---

### 2. Read & Resize Video

```python
amplify_motion(video_path)
```
Reads and resizes the video frames (up to 60 seconds) to conserve memory. Returns grayscale frames and video FPS.

---

### 3. Eulerian Motion Magnification

```python
eulerian_magnification(frames, alpha=30)
```
Enhances subtle periodic motions by subtracting a spatially low-pass filtered version and amplifying the difference. Works in grayscale or color.

---

### 4. Crop to ROI

```python
cropped = [frame[y:y+h, x:x+w] for frame in enhanced]
```
Only the selected ROI is kept for further optical flow processing.

---

### 5. Optical Flow Motion Tracking

- For general vibration (magnitude of flow vectors):

```python
extract_optical_flow_magnitude(frames)
```

- For vertical motion tracking:

```python
extract_vertical_displacement(frames)
```

Returns the **motion magnitude time series**.

---

### 6. Signal Smoothing

```python
smooth_signal_with_transformer(magnitudes)
```

Applies Gaussian smoothing and MinMax scaling to reduce noise in the time series signal.

---

### 7. FFT Frequency Analysis

```python
perform_fft_analysis(magnitudes, fps)
```

Extracts the **dominant frequencies** in the signal using FFT and calculates:

- Total Energy
- Max Amplitude & Frequency
- Spectral Centroid
- Spectral Bandwidth
- Spectral Entropy

---

### 8. Time-Domain Features

```python
compute_time_features(signal)
```

Computes time-domain statistics like:

- Mean, Std Dev, Variance
- Skewness & Kurtosis
- Entropy & Peak-to-Peak
- Zero-Crossing Rate

---

### 9. Spectrogram Visualization

```python
generate_spectrogram(signal, fps)
```

Creates a spectrogram (Short-Time Fourier Transform) for time-varying frequency representation.

---

### 10. CSV Export

```python
save_vibration_data_to_csv(mags, smoothed, freq_features)
```

Exports signal and frequency features to a `.csv` file for further analysis or ML usage.

---

### 11. Plot Generation

```python
plot_results(...)
```

Creates a plot with:

- Original & Smoothed time series
- FFT Frequency Spectrum
- Spectrogram
- Frequency Feature Bar Chart

Saved as `.png` in the `Outputs/` folder.

---

## 🧪 Full Pipeline Function

```python
process_vibration_video(video_path, track_vertical=False)
```

Combines all steps in order, and allows toggling between **vertical motion tracking** and **flow magnitude** mode.

---

## 🎯 Use Cases

- Machine fault detection
- Structural health monitoring
- Biomedical tremor analysis
- MEMS vibration validation
- Remote sensorless vibration diagnostics

---

## 🖼 Example Output

- ✅ Eulerian enhanced clip video
- ✅ Optical flow vector visualization video
- ✅ Vibration signal CSV file
- ✅ Plot with frequency and spectrogram insights
