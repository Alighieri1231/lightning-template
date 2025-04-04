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
    # Calcula el flujo óptico usando el método de Farneback
    flow = cv2.calcOpticalFlowFarneback(
        fixed_frame, moving_frame, None, 0.5, 3, 25, 5, 7, 1.5, 0
    )
    return flow


def create_circular_mask(frame, margin=20):
    """
    Crea una máscara circular a partir del mayor contorno en el frame.
    Se dilata (mediante erosión) para incluir un margen extra.
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


def fixBorder(frame):
    """
    Corrige artefactos en los bordes escalando levemente el frame.
    """
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def detrend_transforms(transforms):
    """
    Aplica detrending a las transformaciones para eliminar el movimiento global (sweep).
    Se calcula la trayectoria acumulada, se ajusta una tendencia lineal y se resta para "congelar"
    el desplazamiento constante, dejando solo las vibraciones.

    :param transforms: numpy array de shape (n_frames-1, 3) con [dx, dy, da]
    :return: numpy array con las transformaciones corregidas.
    """
    # Calcular la trayectoria acumulada
    trajectory = np.cumsum(transforms, axis=0)
    n = trajectory.shape[0]
    t = np.linspace(0, 1, n)
    trend = np.zeros_like(trajectory)
    # Ajuste lineal para cada componente: dx, dy, da
    for i in range(3):
        coeffs = np.polyfit(t, trajectory[:, i], 1)
        trend[:, i] = np.polyval(coeffs, t)
    # Trayectoria sin la tendencia (detrended)
    detrended_trajectory = trajectory - trend
    # Diferencia que se debe aplicar a cada transformación
    difference = detrended_trajectory - trajectory
    transforms_detrended = transforms + difference
    return transforms_detrended


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

    # Build file paths para el video y las coordenadas de recorte
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

    # Determinar inicio y fin del video basándose en diferencias de frames
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

    # Seleccionar solo frames con máscara no vacía
    non_empty_frames = np.where(np.any(ob > 0, axis=(1, 2)))[0]
    if non_empty_frames.size == 0:
        print(
            f"Skipping patient {patient}, sweep {sweep} because no non-empty frames found"
        )
        return None

    # Insertamos el frame anterior, asegurándonos de que no sea negativo (al menos 0)
    non_empty_frames = np.concatenate(
        ([max(non_empty_frames[0] - 1, 0)], non_empty_frames)
    )

    ob = ob[non_empty_frames[0] : non_empty_frames[-1] + 1]
    gt_video = gt_video[non_empty_frames[0] : non_empty_frames[-1] + 1]

    print(f"Frames used: {non_empty_frames[0]} to {non_empty_frames[-1]}")
    print(f"Total frames: {gt_video.shape[0]}")

    resultant = (ob > 0).astype(int) * gt_video

    # ==============================
    # Preprocesamiento: Estabilización vía detrending
    # ==============================
    # Se calcula la transformación entre frames y se aplica detrending para eliminar el movimiento global.
    n_frames_video = gt_video.shape[0]
    transforms = np.zeros((n_frames_video - 1, 3), np.float32)
    prev_gray = gt_video[0]
    print(gt_video.shape)
    for i in range(n_frames_video - 1):
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
        )
        curr_gray = gt_video[i + 1]
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None
        )
        # Asegurarse de que se han seguido los mismos puntos
        assert prev_pts.shape == curr_pts.shape
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        m, inliers = cv2.estimateAffine2D(prev_pts, curr_pts)
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

    # Aplicar detrending para "congelar" el sweep (movimiento global)
    transforms_detrended = detrend_transforms(transforms)

    # Aplicar las transformaciones corregidas a cada frame para obtener el video estabilizado
    gt_video_stabilized = []
    for i in range(n_frames_video - 1):
        # Read next frame
        frame = gt_video[i + 1]  # Current frame from gt_video

        dx = transforms_detrended[i, 0]
        dy = transforms_detrended[i, 1]
        da = transforms_detrended[i, 2]
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        w = frame.shape[1]
        h = frame.shape[0]
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        frame_stabilized = fixBorder(frame_stabilized)
        gt_video_stabilized.append(frame_stabilized)
    gt_video_stabilized = np.array(gt_video_stabilized)
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(gt_video[0], cmap="gray")
    # plt.title("Original")
    # plt.subplot(132)
    # plt.imshow(gt_video_stabilized[0], cmap="gray")
    # plt.title("Estabilizado")
    # plt.subplot(133)
    # plt.imshow(gt_video_stabilized[0] - gt_video[0], cmap="gray")
    # plt.title("Diferencia")
    # plt.show()

    # Ahora usamos gt_video_stabilized para todo el análisis (overlay, optical flow, heartbeat, etc.)
    # ------------------------
    # Creación del GIF de overlay (rectángulo sobre la región de interés)
    # ------------------------
    # Se utiliza el video estabilizado en lugar del original
    centers, rows_min, rows_max, cols_min, cols_max = [], [], [], [], []
    max_value = 0
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
    if len(centers) > 0:
        col_center = int(np.mean(centers))
    else:
        col_center = gt_video_stabilized.shape[2] // 2
    window_size = 10
    half_window = window_size // 2
    col_start = min(cols_min)
    col_end = max(cols_max)
    row_start = min(rows_min)
    row_end = max(rows_max)
    frames_with_rect = []
    for frame in gt_video_stabilized:
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
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
    analisis_gt = gt_video_stabilized[:, row_start:row_end, col_start:col_end]
    ob_analisis = ob[1:, row_start:row_end, col_start:col_end]
    analisis_gt = analisis_gt * ob_analisis

    margin = 12  # Margen para la máscara
    frames_with_flow = []
    magnitude_changes = []
    angle_changes = []
    num_frames = gt_video_stabilized.shape[0]
    for frame_idx in range(1, num_frames):
        frame1 = analisis_gt[frame_idx - 1]
        frame2 = analisis_gt[frame_idx]
        mask = create_circular_mask(frame1, margin)
        if mask is None:
            mask = np.ones_like(frame1, dtype=np.uint8) * 255
        mask = mask / 255.0
        flow = compute_optical_flow(frame1, frame2)
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
    n_frames_angle = len(angle_changes)
    low_bpm = 100
    high_bpm = 200
    lowcut = low_bpm / 60.0
    highcut = high_bpm / 60.0
    fs_val = fps

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

    plt.figure()
    plt.plot(filtered_angle_changes)
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Optical Flow Angle (degrees)")
    plt.title(
        f"Filtered Angle - Patient: {sweep}, Frames: {n_frames_angle}, Filter Order: {filter_order}"
    )
    plt.grid(True)
    filtered_plot_filename = os.path.join(
        angle_plot_dir, f"{sweep}_filtered_angle_changes.png"
    )
    plt.savefig(filtered_plot_filename)
    plt.close()

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

    return max_bpm, n_frames_angle, filter_order, total_frames


# ------------------------
# Main Script
# ------------------------
if __name__ == "__main__":
    output_dir = os.path.join(general_path, "output_correccion_detrending")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bpm_results = []
    for i in range(len(patient_t)):
        print(f"Processing patient sweep {sweep_t[i]}...")
        result = process_video(
            general_path, patient_t[i], sweep_t[i], ob1_files[i], output_dir
        )
        if result is None:
            continue
        bpm, n_frames_angle, filter_order_used, total_frames = result
        bpm_results.append(
            {
                "patient": patient_t[i],
                "sweep": sweep_t[i],
                "bpm": bpm,
                "n_frames": n_frames_angle,
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
