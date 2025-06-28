import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
import os
from scipy.stats import moment
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch


# Ensure output directory exists
OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to read and process video frames in smaller chunks to avoid memory issues
def amplify_motion(video_path, target_width=640, target_height=360, chunk_size=100, max_duration=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None, None

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_duration)
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (target_width, target_height))
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frames.append(frame_gray)
        frame_count += 1
        print(f"Processed {frame_count} frames so far.")

    cap.release()
    print(f"Successfully read {len(frames)} frames from the video.")
    return frames, fps

# Save magnified grayscale video
def save_video(frames, output_path, fps=30):
    output_path = os.path.join(OUTPUT_DIR, output_path)
    h, w = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)
    for frame in frames:
        out.write(np.uint8(frame))
    out.release()
    print(f"Enhanced video saved to {output_path}")

# Save optical flow vector field visualization as video
def save_optical_flow_video(frames, output_path, fps):
    output_path = os.path.join(OUTPUT_DIR, output_path)
    h, w = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)
    prev = frames[0].astype(np.float32)

    for next_frame in frames[1:]:
        next_f = next_frame.astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(prev, next_f, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out.write(bgr)
        prev = next_f

    out.release()
    print(f"Optical flow vector field video saved to {output_path}")

# Eulerian magnification - now operates on ROI frames
def eulerian_magnification(frames, alpha=30, filter_size=5, lambda_=5, use_color=False):
    enhanced_frames = []
    for frame in frames:
        if use_color:
            smoothed = gaussian_filter1d(frame, filter_size, axis=0)
            smoothed = gaussian_filter1d(smoothed, filter_size, axis=1)
            amplified = (frame - smoothed) * alpha
            enhanced = np.clip(smoothed + amplified, 0, 255)
        else:
            smoothed = gaussian_filter1d(frame, filter_size, axis=0)
            smoothed = gaussian_filter1d(smoothed, filter_size, axis=1)
            amplified = (frame - smoothed) * alpha
            enhanced = smoothed + amplified
        enhanced_frames.append(enhanced.astype(np.uint8))
    return enhanced_frames

def extract_vertical_displacement(frames):
    vertical_signal = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vertical_movement = flow[..., 1].mean()
        vertical_signal.append(vertical_movement)
        prev = gray
    return np.array(vertical_signal)

# Optical Flow-based displacement time series
def extract_optical_flow_magnitude(frames):
    magnitudes = []
    prev = frames[0].astype(np.float32)
    for next_frame in frames[1:]:
        next_f = next_frame.astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(prev, next_f, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.linalg.norm(flow, axis=2).mean()
        magnitudes.append(magnitude)
        prev = next_f
    return np.array(magnitudes)

# Function to crop frames to ROI
def crop_frames_to_roi(frames, roi, scale):
    x, y, w, h = roi
    x_start = int(x / scale)
    y_start = int(y / scale)
    x_end = int((x + w) / scale)
    y_end = int((y + h) / scale)
    
    cropped_frames = []
    for frame in frames:
        # Handle both grayscale and color frames
        if len(frame.shape) == 2:  # Grayscale
            cropped = frame[y_start:y_end, x_start:x_end]
        else:  # Color
            cropped = frame[y_start:y_end, x_start:x_end, :]
        cropped_frames.append(cropped)
    
    return cropped_frames

# Signal smoothing
def smooth_signal_with_transformer(magnitudes, smoothing_factor=0.1):
    scaler = MinMaxScaler()
    magnitudes_scaled = scaler.fit_transform(magnitudes.reshape(-1, 1)).flatten()
    smoothed_signal = gaussian_filter1d(magnitudes_scaled, int(smoothing_factor * len(magnitudes)))
    return scaler.inverse_transform(smoothed_signal.reshape(-1, 1)).flatten()

# Fourier transform and frequency analysis
def perform_fft_analysis(magnitudes, fps):
    N = len(magnitudes)
    T = 1.0 / fps
    yf = fft(magnitudes)
    xf = fftfreq(N, T)[:N // 2]
    amplitude = 2.0 / N * np.abs(yf[:N // 2])
    return xf, amplitude

def compute_frequency_features(xf, amplitude):
    norm_amp = amplitude / np.sum(amplitude) if np.sum(amplitude) != 0 else amplitude
    spectral_entropy = entropy(norm_amp)
    spectral_centroid = np.sum(xf * amplitude) / np.sum(amplitude) if np.sum(amplitude) != 0 else 0
    spectral_bandwidth = np.sqrt(np.sum(((xf - spectral_centroid) ** 2) * amplitude) / np.sum(amplitude)) if np.sum(amplitude) != 0 else 0

    return {
        "Total Energy": np.sum(amplitude ** 2),
        "Max Amplitude": np.max(amplitude),
        "Frequency of Max Amplitude": xf[np.argmax(amplitude)],
        "Spectral Centroid": spectral_centroid,
        "Spectral Bandwidth": spectral_bandwidth,
        "Spectral Entropy": spectral_entropy
    }

def compute_time_features(signal):
    signal = signal - np.mean(signal)
    zcr = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
    features = {
        "Mean": np.mean(signal),
        "Std Dev": np.std(signal),
        "Variance": np.var(signal),
        "Skewness": skew(signal),
        "Kurtosis": kurtosis(signal),
        "Entropy": entropy(np.abs(signal) / np.sum(np.abs(signal))),
        "Peak-to-Peak": np.ptp(signal),
        "Zero Crossing Rate": zcr
    }
    return features

# Short-Term Fourier Transform (Spectrogram)
def generate_spectrogram(magnitudes, fps):
    f, t, Sxx = spectrogram(magnitudes, fs=fps, nperseg=64, noverlap=32)
    return f, t, Sxx

# CSV export
def save_vibration_data_to_csv(magnitudes, smoothed_magnitudes, freq_features, output_file="vibration_data.csv"):
    output_file = os.path.join(OUTPUT_DIR, output_file)
    df = pd.DataFrame({
        'Magnitude': magnitudes,
        'Smoothed Magnitude': smoothed_magnitudes
    })
    for key, value in freq_features.items():
        df[key] = [value] * len(df)  # Repeat for all rows
    df.to_csv(output_file, index=False)
    print(f"Saved processed vibration data with frequency features to {output_file}")

# Plotting
def plot_results(magnitudes, smoothed_magnitudes, xf, amplitude, f, t, Sxx, clip_name, freq_features):
    plot_filename = f"{clip_name}_plot_results.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)

    plt.figure(figsize=(14, 10))

    # Time series
    plt.subplot(4, 1, 1)
    plt.plot(magnitudes, label='Original')
    plt.plot(smoothed_magnitudes, label='Smoothed', linestyle='--')
    plt.title(f"Vibration Time Series - {clip_name}")
    plt.legend()

    # FFT
    plt.subplot(4, 1, 2)
    plt.plot(xf, amplitude)
    plt.title("FFT - Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")

    # Spectrogram
    plt.subplot(4, 1, 3)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.title("Log Spectrogram")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label='dB')

    # Frequency Features Summary
    plt.subplot(4, 1, 4)
    plt.bar(freq_features.keys(), freq_features.values(), color='skyblue')
    plt.title("Frequency Features Summary")
    plt.xticks(rotation=15)
    plt.ylabel("Value")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot for {clip_name} to {plot_path}")

# ROI selection
def get_roi_from_video(video_path, target_width=640, target_height=360):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read video.")
        return None, None
    
    # Resize frame to match processing resolution
    frame_resized = cv2.resize(frame, (target_width, target_height))
    
    # Further scale for display if needed
    height, width = frame_resized.shape[:2]
    display_scale = 1.0
    if height > 600:
        display_scale = 600 / height
        display_frame = cv2.resize(frame_resized, (int(width * display_scale), 600))
    else:
        display_frame = frame_resized
    
    roi = cv2.selectROI("Select ROI", display_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    cap.release()
    
    # Adjust ROI coordinates back to processing resolution
    if display_scale != 1.0:
        x, y, w, h = roi
        roi = (int(x / display_scale), int(y / display_scale), 
               int(w / display_scale), int(h / display_scale))
    
    return roi, 1.0  # Return scale as 1.0 since we're working with processing resolution

# Pipeline
def generate_clip_indices(fps, total_frames, clip_duration=20, stride=5):
    step = int(stride * fps)
    length = int(clip_duration * fps)
    indices = []
    for start in range(0, total_frames - length + 1, step):
        end = start + length
        indices.append((start, end))
    return indices

def process_vibration_video(video_path, track_vertical=False):
    print("[1] Select ROI...")
    roi, scale = get_roi_from_video(video_path)
    if roi is None:
        return

    print(f"Selected ROI: {roi}")

    print("[2] Reading video and amplifying...")
    frames, fps = amplify_motion(video_path)
    if frames is None:
        return

    clip_indices = generate_clip_indices(fps, len(frames), clip_duration=20, stride=5)
    print(f"[INFO] Total clips to process: {len(clip_indices)}")

    for idx, (start, end) in enumerate(clip_indices):
        print(f"\n=== Processing Clip {idx+1}: frames {start} to {end} ===")

        clip_frames = frames[start:end]
        if len(clip_frames) < 2:
            print(f"Skipping clip {idx+1} due to insufficient frames.")
            continue

        print("[3] Cropping frames to ROI...")
        # Crop frames to ROI BEFORE applying Eulerian magnification
        cropped_frames = crop_frames_to_roi(clip_frames, roi, scale)
        
        print(f"ROI dimensions: {cropped_frames[0].shape}")

        print("[4] Applying Eulerian magnification to ROI...")
        if track_vertical:
            # Convert grayscale ROI frames to color for vertical tracking
            color_cropped = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in cropped_frames]
            enhanced_roi = eulerian_magnification(color_cropped, use_color=True)
            # Convert back to grayscale for saving
            enhanced_roi_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in enhanced_roi]
        else:
            enhanced_roi = eulerian_magnification(cropped_frames)
            enhanced_roi_gray = enhanced_roi

        clip_video_name = f"eulerian_output_roi_clip_{idx+1}.mp4"
        save_video(enhanced_roi_gray, clip_video_name, fps)

        print("[5] Calculating optical flow on ROI...")
        if track_vertical:
            magnitudes = extract_vertical_displacement(enhanced_roi)
        else:
            magnitudes = extract_optical_flow_magnitude(enhanced_roi_gray)

        magnitudes = magnitudes - np.mean(magnitudes)  # Remove DC component
        
        optical_flow_video_name = f"optical_flow_vectors_roi_clip_{idx+1}.mp4"
        save_optical_flow_video(enhanced_roi_gray, optical_flow_video_name, fps)

        print("[6] Smoothing signal...")
        smoothed = smooth_signal_with_transformer(magnitudes)

        print("[7] Performing FFT analysis...")
        xf, amp = perform_fft_analysis(magnitudes, fps)

        print("[8] Generating spectrogram...")
        f, t, Sxx = generate_spectrogram(magnitudes, fps)
        freq_features = compute_frequency_features(xf, amp)
        time_features = compute_time_features(magnitudes)

        print("[9] Saving CSV and plotting...")
        csv_name = f"vibration_data_roi_clip_{idx+1}.csv"
        save_vibration_data_to_csv(magnitudes, smoothed, {**freq_features, **time_features}, output_file=csv_name)

        clip_name = f"ROI_Plot_{idx+1}"
        plot_results(magnitudes, smoothed, xf, amp, f, t, Sxx, clip_name, {**freq_features, **time_features})

# Usage
VIDEO_PATH = "250 RPM/Rotary machine with Bearing fault in outer ring/250 rpm/From Above.MOV"
process_vibration_video(VIDEO_PATH, track_vertical=True)
