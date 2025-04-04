#!/usr/bin/env python3
import glob
import natsort
import os
import sys
import csv
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import imageio
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

# ------------------------
# File Listing and ID Extraction
# ------------------------
general_path = "/Users/emilio/Library/CloudStorage/Box-Box/GitHub/lightning-template/"
ob1_files = glob.glob(os.path.join(general_path, "LabelDataObstetrics", "*.mat"))
ob1_files = natsort.natsorted(ob1_files)

patient_t = []
sweep_t = []
for file in ob1_files:
    sweep = file.split("/")[-1].split(".")[0]
    if "copy" in sweep:
        sweep = sweep.split(" copy")[0]
    patient = sweep[0:3]
    patient = "0" + patient if "_" in patient else patient
    patient = patient.split("_")[0]
    patient_t.append(patient)
    sweep_t.append(sweep)


# ------------------------
# Utility Functions
# ------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    lowcut = lowcut / nyq
    highcut = highcut / nyq
    b, a = butter(order, [lowcut, highcut], btype="band", analog=False)
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def compute_optical_flow(fixed_frame, moving_frame):
    # Compute optical flow using the Farneback method with custom parameters
    flow = cv2.calcOpticalFlowFarneback(
        fixed_frame, moving_frame, None, 0.5, 3, 25, 5, 7, 1.5, 0
    )
    return flow


def create_circular_mask(frame, margin=20):
    """
    Create a mask by detecting the largest contour in the frame,
    then dilate it (via erosion with a margin) to include extra area.
    """
    _, thresh = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        dilated_mask = cv2.erode(mask, None, iterations=margin)
        return dilated_mask
    else:
        return None


# ------------------------
# Video Processing Function
# ------------------------
def process_video(general_path, patient, sweep, ob_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Crear subcarpetas para cada tipo de salida
    overlay_dir = os.path.join(output_dir, "overlay")
    optical_flow_dir = os.path.join(output_dir, "optical_flow")
    fft_plot_dir = os.path.join(output_dir, "fft_plot")
    angle_plot_dir = os.path.join(output_dir, "angle_plot")
    for subfolder in [overlay_dir, optical_flow_dir, fft_plot_dir, angle_plot_dir]:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    # Build file paths for the video and crop coordinates
    mp4_file_path = os.path.join(general_path, "Study", patient, sweep + ".mp4")
    crop_coordinates_path = os.path.join(
        general_path, "crop_coordinates", patient + ".mat"
    )

    # Cargar coordenadas de recorte y ajustar índices para Python
    coordinates = sio.loadmat(crop_coordinates_path)
    x1 = int(coordinates["x1"][0][0]) - 1
    x2 = int(coordinates["x2"][0][0])
    y1 = int(coordinates["y1"][0][0]) - 1
    y2 = int(coordinates["y2"][0][0])
    width = x2 - x1
    height = y2 - y1

    # Cargar la máscara de segmentación y ajustar dimensiones
    ob = sio.loadmat(ob_file)["labels"]
    ob = np.moveaxis(ob, 2, 0)

    # Leer frames del video
    cap = cv2.VideoCapture(mp4_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    gt_video = np.array(frames)
    total_frames = gt_video.shape[0]

    # Calcular diferencias de frames para encontrar el inicio y fin
    diff = gt_video[1:, :, :, 0] - gt_video[:-1, :, :, 0]
    diff_sum = np.sum(diff, axis=(1, 2))
    diff_sum = diff_sum / np.max(diff_sum)

    threshold = 0.2
    first_frame = np.where(diff_sum > threshold)[0][0]
    last_frame = np.where(diff_sum > threshold)[0][-1]
    for j in range(last_frame, diff.shape[0]):
        if np.sum(ob[j]) != 0:
            last_frame = j
    for j in range(first_frame, 0, -1):
        if np.sum(ob[j]) != 0:
            first_frame = j

    # Recortar video y máscara a la región de interés
    if ob.shape[1] != gt_video.shape[1] or ob.shape[2] != gt_video.shape[2]:
        if width != ob.shape[2] or height != ob.shape[1]:
            gt_video = gt_video[first_frame:last_frame, y1:y2, x1:x2, 0]
            ob = ob[first_frame:last_frame, :, :]
            pad_or_crop_h = gt_video.shape[1] - ob.shape[1]
            pad_h_before = max(0, pad_or_crop_h // 2)
            pad_h_after = max(0, pad_or_crop_h - pad_h_before)
            crop_h_before = max(0, -pad_or_crop_h // 2)
            crop_h_after = max(0, -pad_or_crop_h - crop_h_before)
            pad_or_crop_w = gt_video.shape[2] - ob.shape[2]
            pad_w_before = max(0, pad_or_crop_w // 2)
            pad_w_after = max(0, pad_or_crop_w - pad_w_before)
            crop_w_before = max(0, -pad_or_crop_w // 2)
            crop_w_after = max(0, -pad_or_crop_w - crop_w_before)
            if (
                pad_h_before > 0
                or pad_h_after > 0
                or pad_w_before > 0
                or pad_w_after > 0
            ):
                ob = np.pad(
                    ob,
                    ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
                    "constant",
                )
            if (
                crop_h_before > 0
                or crop_h_after > 0
                or crop_w_before > 0
                or crop_w_after > 0
            ):
                ob = ob[
                    :,
                    crop_h_before : ob.shape[1] - crop_h_after,
                    crop_w_before : ob.shape[2] - crop_w_after,
                ]
        else:
            gt_video = gt_video[first_frame:last_frame, y1:y2, x1:x2, 0]
            ob = ob[first_frame:last_frame, :, :]
    else:
        gt_video = gt_video[first_frame:last_frame, y1:y2, x1:x2, 0]
        ob = ob[first_frame:last_frame, y1:y2, x1:x2]

    # Seleccionar frames con máscara no vacía
    non_empty_frames = np.where(np.any(ob > 0, axis=(1, 2)))[0]
    if non_empty_frames.size == 0:
        print(
            f"Skipping patient {patient}, sweep {sweep} because no non-empty frames found"
        )
        return None  # Salta al siguiente paciente

    ob = ob[non_empty_frames[0] : non_empty_frames[-1] + 1]
    gt_video = gt_video[non_empty_frames[0] : non_empty_frames[-1] + 1]
    resultant = (ob > 0).astype(int) * gt_video

    # Determinar la región para el rectángulo de overlay a partir de los contornos de la máscara
    centers, rows_min, rows_max, cols_min, cols_max = [], [], [], [], []
    max_value = 0
    id_max = 0
    for idx in range(ob.shape[0]):
        mask_i = ob[idx]
        if np.sum(mask_i) > 0:
            row_inds, col_inds = np.where(mask_i > 0)
            centers.append(np.mean(col_inds))
            rows_min.append(row_inds.min())
            rows_max.append(row_inds.max())
            cols_min.append(col_inds.min())
            cols_max.append(col_inds.max())
            if col_inds.max() > max_value:
                max_value = col_inds.max()
                id_max = idx
    if len(centers) > 0:
        col_center = int(np.mean(centers))
    else:
        col_center = gt_video.shape[1] // 2
    window_size = 10
    half_window = window_size // 2
    col_start = max(0, col_center - half_window)
    col_end = min(gt_video.shape[1], col_center + half_window)
    col_start = min(cols_min)
    col_end = max(cols_max)
    row_start = min(rows_min)
    row_end = max(rows_max)

    # Crear y guardar el GIF de overlay (frames de video con el rectángulo dibujado)
    frames_with_rect = []
    for frame in gt_video:
        if len(frame.shape) == 2:
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame.copy()
        cv2.rectangle(
            frame_color,
            (col_start, row_start),
            (col_end, row_end),
            (0, 0, 255),
            thickness=2,
        )
        frame_rgb = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)
        frames_with_rect.append(frame_rgb)
    overlay_gif_filename = os.path.join(overlay_dir, f"{sweep}_overlay.gif")
    imageio.mimsave(overlay_gif_filename, frames_with_rect, fps=10)

    # ------------------------
    # Análisis de Optical Flow y creación del GIF
    # ------------------------
    analisis_gt = gt_video[:, row_start:row_end, col_start:col_end]
    ob_analisis = ob[:, row_start:row_end, col_start:col_end]
    analisis_gt = analisis_gt * ob_analisis

    margin = 12  # Margen para la máscara de optical flow
    frames_with_flow = []
    magnitude_changes = []
    angle_changes = []
    num_frames = gt_video.shape[0]
    for frame_idx in range(1, num_frames):
        frame1 = analisis_gt[frame_idx - 1]
        frame2 = analisis_gt[frame_idx]
        mask = create_circular_mask(frame1, margin)
        if mask is None:
            mask = np.ones_like(frame1, dtype=np.uint8) * 255
        mask = mask / 255.0
        flow = compute_optical_flow(frame1, frame2)
        # Aplicar la máscara al flujo óptico
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = mag * ob_analisis[frame_idx - 1] * mask
        ang = ang * ob_analisis[frame_idx - 1] * mask
        non_zero_mask = mag > 0
        if np.any(non_zero_mask):
            mag_fil = mag[non_zero_mask]
            ang_fil = ang[non_zero_mask]
            mean_magnitude = np.mean(mag_fil)
            mean_angle = np.mean(ang_fil) * 180 / np.pi
        else:
            mean_magnitude = 0
            mean_angle = 0
        magnitude_changes.append(mean_magnitude)
        angle_changes.append(mean_angle)

        hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame1_bgr = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        frame2_bgr = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
        flow_with_overlay = cv2.addWeighted(frame2_bgr, 0.7, flow_rgb, 0.3, 0)
        frame_rgb = cv2.cvtColor(flow_with_overlay, cv2.COLOR_BGR2RGB)
        frames_with_flow.append(frame_rgb)
    optical_flow_gif_filename = os.path.join(
        optical_flow_dir, f"{sweep}_optical_flow.gif"
    )
    imageio.mimsave(optical_flow_gif_filename, frames_with_flow, fps=10)

    # ------------------------
    # Análisis de Heartbeat: filtrado de cambios de ángulo y FFT
    # ------------------------
    angle_changes = np.array(angle_changes) - np.mean(angle_changes)
    n_frames = len(angle_changes)
    low_bpm = 100
    high_bpm = 200
    lowcut = low_bpm / 60.0
    highcut = high_bpm / 60.0
    fs_val = fps

    # Intentamos filtrar con orden inicial 5 y, en caso de error, reducimos el orden
    filter_order = 5
    while True:
        try:
            filtered_angle_changes = bandpass_filter(
                angle_changes, lowcut, highcut, fs_val, order=filter_order
            )
            break
        except ValueError as e:
            filter_order -= 1
            if filter_order < 1:
                print(
                    "Warning: no se pudo aplicar el filtro; se usará la señal sin filtrar"
                )
                filtered_angle_changes = angle_changes
                break

    # Guardar plot de los cambios filtrados
    plt.figure()
    plt.plot(filtered_angle_changes)
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Optical Flow Angle (degrees)")
    plt.title(
        f"Filtered Angle - Patient: {sweep}, Frames: {n_frames}, Filter Order: {filter_order}"
    )
    plt.grid(True)
    filtered_plot_filename = os.path.join(
        angle_plot_dir, f"{sweep}_filtered_angle_changes.png"
    )
    plt.savefig(filtered_plot_filename)
    plt.close()

    # FFT del cambio de ángulo filtrado y guardar el plot
    N = len(filtered_angle_changes)
    T = 1.0 / fps
    yf = fft(filtered_angle_changes)
    xf = fftfreq(N, T)[: N // 2]

    peaks, _ = find_peaks(2.0 / N * np.abs(yf[: N // 2]), height=0.1)
    xf_peaks = xf[peaks]
    if xf_peaks.size > 0:
        max_freq_index = np.argmax(2.0 / N * np.abs(yf[: N // 2]))
        max_freq = xf[max_freq_index]
        max_bpm = max_freq * 60
    else:
        max_bpm = 0

    plt.figure()
    plt.stem(xf, 2.0 / N * np.abs(yf[: N // 2]))
    plt.grid(True)
    plt.title(f"FFT of Filtered Angle Changes - Max BPM: {max_bpm:.2f}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(1, 4)
    fft_plot_filename = os.path.join(
        fft_plot_dir, f"{sweep}_fft_filtered_angle_changes.png"
    )
    plt.savefig(fft_plot_filename)
    plt.close()

    return max_bpm, n_frames, filter_order, total_frames


# ------------------------
# Main Script
# ------------------------
if __name__ == "__main__":
    output_dir = os.path.join(general_path, "output_correccion")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bpm_results = []
    for i in range(len(patient_t)):
        print(f"Processing patient sweep {sweep_t[i]}...")
        result = process_video(
            general_path, patient_t[i], sweep_t[i], ob1_files[i], output_dir
        )
        if result is None:
            # Se salta el paciente si no hay frames no vacíos
            continue
        bpm, n_frames, filter_order_used, total_frames = result
        bpm_results.append(
            {
                "patient": patient_t[i],
                "sweep": sweep_t[i],
                "bpm": bpm,
                "n_frames": n_frames,
                "total_frames": total_frames,
                "filter_order": filter_order_used,
            }
        )

    csv_filename = os.path.join(output_dir, "bpm_results.csv")
    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = [
            "patient",
            "sweep",
            "bpm",
            "n_frames",
            "total_frames",
            "filter_order",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in bpm_results:
            writer.writerow(row)
