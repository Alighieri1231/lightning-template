# convert to function
import numpy as np
import cv2


# Función para asignar colores a las máscaras
def apply_colors(mask_combined, colors):
    # Crear una imagen en color basada en las categorías
    color_mask = np.zeros(
        (mask_combined.shape[0], mask_combined.shape[1], 3), dtype=np.uint8
    )
    for i, color in enumerate(colors):
        color_mask[mask_combined == i] = color
    return color_mask


def generate_overlay_video(
    groundtruth, one_hot, fps=15, output_video="output_video.mp4"
):
    # Supongamos que `groundtruth` y `one_hot_mask` son tus datos:
    # - `groundtruth` tiene la forma (696, 1053, 1219)
    # - `one_hot_mask` tiene la forma (696, 1053, 1219, 6)
    # output_video = "output_video.mp4"  # Nombre del video de salida
    # fps = 10  # Frames por segundo

    # Definir los colores para las categorías (BGR para OpenCV)
    colors = [
        (0, 0, 0),  # Negro para la categoría 0
        (0, 0, 255),  # Rojo para la categoría 1 heart
        (0, 255, 0),  # Verde para la categoría 2 cabeza
        (255, 0, 0),  # Azul para la categoría 3 abdomen
        (0, 255, 255),  # Amarillo para la categoría 4 chest
        (255, 0, 255),  # Magenta para la categoría 5 placenta
    ]
    category_names = ["Background", "Heart", "Head", "Abdomen", "Chest", "Placenta"]

    frame_size = (groundtruth.shape[2], groundtruth.shape[1])  # (ancho, alto)

    # Crear el objeto de video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # Generar cada frame
    for i in range(groundtruth.shape[0]):  # Iterar sobre los frames (en la dimensión 0)
        # Obtener el frame de ground truth y la máscara
        gt_frame = groundtruth[i]  # Frame de groundtruth
        mask_frame = one_hot[i]  # Frame de la máscara (one-hot)

        # Combinar las categorías con los colores
        mask_combined = np.argmax(mask_frame, axis=-1)  # Reconstruir categorías
        colored_mask = apply_colors(
            mask_combined, colors
        )  # Asignar colores a las categorías

        # Normalizar el ground truth si es necesario (opcional)
        if gt_frame.max() > 1:
            gt_frame = gt_frame / gt_frame.max()

        # Convertir ground truth a un formato RGB
        gt_rgb = (np.stack([gt_frame] * 3, axis=-1) * 255).astype(np.uint8)

        # Combinar el ground truth con la máscara (alpha blending)
        overlay = cv2.addWeighted(gt_rgb, 0.5, colored_mask, 0.5, 0)
        # Add legend to the frame
        legend_start_y = 10
        for idx, (name, color) in enumerate(zip(category_names, colors)):
            legend_y = legend_start_y + idx * 20
            cv2.rectangle(overlay, (10, legend_y), (30, legend_y + 15), color, -1)
            cv2.putText(
                overlay,
                name,
                (35, legend_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Escribir el frame en el video
        video_writer.write(overlay)

    # Liberar el objeto de video
    video_writer.release()
    print(f"Video guardado como {output_video}")
