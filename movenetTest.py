import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
import argparse  # Pro ovládání skriptu z terminálu
from tqdm import tqdm # Pro ukazatel průběhu u videa

# Importy pro vizualizaci
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio # Pro ukládání GIFu

# ##################################################################
# DOPLNĚNÉ POMOCNÉ FUNKCE (Chyběly ve tvém textu)
# ##################################################################

# Slovník pro mapování klíčových bodů
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Mapování hran kostry pro vykreslení
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores, height, width, keypoint_threshold=0.11):
    """Vrátí body a hrany pro vykreslení."""
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh = kpts_scores > keypoint_threshold
        keypoints_all.append(kpts_absolute_xy[kpts_above_thresh])

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_above_thresh[edge_pair[0]] and
                    kpts_above_thresh[edge_pair[1]]):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None,
    keypoint_threshold=0.11, output_image_height=None):
    """Vykreslí body a hrany na obrázek."""
    height, width, _ = image.shape
    (keypoints_xy, edges_xy,
     edge_colors) = _keypoints_and_edges_for_display(
         keypoints_with_scores, height, width, keypoint_threshold)

    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image = tf.image.resize(
            image, [output_image_height, output_image_width]).numpy()
        keypoints_xy *= [output_image_width / width, output_image_height / height]
        edges_xy *= [output_image_width / width, output_image_height / height]
        height, width, _ = image.shape
        
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(image.astype(np.uint8))
    ax.set_axis_off()

    # Vykreslení bodů
    ax.scatter(keypoints_xy[:, 0], keypoints_xy[:, 1], c='r', s=10)

    # Vykreslení hran
    line_segments = LineCollection(
        edges_xy, linewidths=(2), linestyle='solid', colors=edge_colors)
    ax.add_collection(line_segments)
    
    # Převod plátna na obrázek
    fig.canvas.draw()
    image_from_plot = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image_from_plot

# --- Funkce pro ořez (video) ---
_MIN_CROP_KEYPOINT_SCORE = 0.2

def init_crop_region(image_height, image_width):
    """Definuje počáteční ořez."""
    if image_width > image_height:
        box_height = image_width
        box_width = image_width
        y_min = (image_height - box_height) / 2
        x_min = 0
    else:
        box_height = image_height
        box_width = image_height
        y_min = 0
        x_min = (image_width - box_width) / 2
    return {
        'y_min': y_min / image_height,
        'x_min': x_min / image_width,
        'y_max': (y_min + box_height) / image_height,
        'x_max': (x_min + box_width) / image_width
    }

def determine_crop_region(keypoints_with_scores, image_height, image_width):
    """Určí další ořez na základě detekovaných bodů."""
    instance_keypoints = keypoints_with_scores[0, 0, :, :]
    keypoint_scores = instance_keypoints[:, 2]
    keypoint_coords = instance_keypoints[:, :2]
    
    valid_keypoints = keypoint_scores > _MIN_CROP_KEYPOINT_SCORE
    if not np.any(valid_keypoints):
        return init_crop_region(image_height, image_width)
        
    valid_coords = keypoint_coords[valid_keypoints, :]
    y_min, x_min = np.min(valid_coords, axis=0)
    y_max, x_max = np.max(valid_coords, axis=0)
    
    center_y = (y_min + y_max) / 2
    center_x = (x_min + x_max) / 2

    # Vytvoření čtvercového ořezu
    box_size = np.max([y_max - y_min, x_max - x_min]) * 1.5
    y_min = center_y - box_size / 2
    x_min = center_x - box_size / 2
    box_height = box_size
    box_width = box_size
    
    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width
    }

def run_inference(movenet, image, crop_region, crop_size):
    """Spustí detekci na ořezaném obrázku."""
    image_height, image_width, _ = image.shape
    # Převedení normalizovaných souřadnic ořezu na pixely
    y_min_pix = int(crop_region['y_min'] * image_height)
    x_min_pix = int(crop_region['x_min'] * image_width)
    y_max_pix = int(crop_region['y_max'] * image_height)
    x_max_pix = int(crop_region['x_max'] * image_width)
    
    # Ořez obrázku
    crop_height = y_max_pix - y_min_pix
    crop_width = x_max_pix - x_min_pix
    
    # Ošetření případů, kdy je ořez mimo hranice
    if crop_height <= 0 or crop_width <= 0:
        return np.zeros((1, 1, 17, 3)) # Vrátí prázdné body

    # Vytvoření tensoru pro ořez
    crop_box = [y_min_pix / (image_height - 1), x_min_pix / (image_width - 1),
                y_max_pix / (image_height - 1), x_max_pix / (image_width - 1)]
    
    # Ořez a změna velikosti
    cropped_image = tf.image.crop_and_resize(
        tf.expand_dims(image, axis=0),
        [crop_box],
        [0],
        crop_size
    )
    
    # Spuštění detekce
    keypoints_with_scores = movenet(cropped_image)
    
    # Přepočítání souřadnic zpět na původní obrázek
    keypoints_with_scores_recaled = np.copy(keypoints_with_scores)
    keypoints_with_scores_recaled[0, 0, :, 0] = (
        keypoints_with_scores[0, 0, :, 0] * (crop_height / image_height) + 
        crop_region['y_min']
    )
    keypoints_with_scores_recaled[0, 0, :, 1] = (
        keypoints_with_scores[0, 0, :, 1] * (crop_width / image_width) + 
        crop_region['x_min']
    )
    
    return keypoints_with_scores_recaled

def save_gif(images, filepath, duration=100):
    """Uloží seznam obrázků jako GIF."""
    imageio.mimsave(filepath, images, duration=duration, loop=0)
    print(f"GIF uložen do: {filepath}")

# ##################################################################
# HLAVNÍ ČÁST SKRIPTU
# ##################################################################

def load_movenet_model(model_name="lightning"):
    """N nahraje model MoveNet z TensorFlow Hub."""
    print(f"Nahrávám model movenet_{model_name}...")
    if model_name == "lightning":
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif model_name == "thunder":
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError(f"Neznámý model: {model_name}. Použij 'lightning' nebo 'thunder'.")

    def movenet_fn(input_image):
        """Wrapper pro spuštění modelu."""
        model = module.signatures['serving_default']
        # Model očekává tf.int32
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = model(input_image)
        # Výstup je [1, 1, 17, 3] tensor
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores

    print("Model nahrán.")
    return movenet_fn, input_size

def process_image(movenet_fn, input_size, input_path, output_path):
    """Zpracuje jeden obrázek."""
    print(f"Zpracovávám obrázek: {input_path}")
    
    # Načtení a příprava obrázku
    image = tf.io.read_file(input_path)
    image = tf.image.decode_image(image, channels=3) # decode_image zvládne jpg i png
    
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Spuštění detekce
    keypoints_with_scores = movenet_fn(input_image)

    # Příprava pro vizualizaci
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image, 1280, 1280), dtype=tf.int32)
    
    # Vykreslení
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores
    )

    # Uložení výsledku
    plt.imsave(output_path, output_overlay)
    print(f"Výstupní obrázek uložen do: {output_path}")

def process_video(movenet_fn, input_size, input_path, output_path):
    """Zpracuje video (GIF)."""
    print(f"Zpracovávám video/GIF: {input_path}")
    
    # Načtení GIFu
    image = tf.io.read_file(input_path)
    image_tensor = tf.image.decode_gif(image)
    num_frames, image_height, image_width, _ = image_tensor.shape
    
    crop_region = init_crop_region(image_height, image_width)
    output_images = []

    print(f"Zpracovávám {num_frames} snímků...")
    for frame_idx in tqdm(range(num_frames)):
        frame = image_tensor[frame_idx, :, :, :]
        
        # Detekce s ořezem
        keypoints_with_scores = run_inference(
            movenet_fn, frame, crop_region,
            crop_size=[input_size, input_size])
        
        # Vykreslení
        output_images.append(draw_prediction_on_image(
            frame.numpy().astype(np.int32),
            keypoints_with_scores,
            output_image_height=300)) # Menší výška pro GIF
        
        # Aktualizace ořezu pro další snímek
        crop_region = determine_crop_region(
            keypoints_with_scores, image_height, image_width)

    # Uložení GIFu
    save_gif(output_images, output_path, duration=100)


def main():
    # --- Nastavení argumentů ---
    parser = argparse.ArgumentParser(description="MoveNet detekce postoje")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['image', 'video'],
        help="Režim zpracování: 'image' (obrázek) nebo 'video' (gif)."
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="Cesta ke vstupnímu souboru (obrázek nebo .gif)."
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help="Cesta pro uložení výstupního souboru (např. vystup.jpg nebo vystup.gif)."
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='lightning', 
        choices=['lightning', 'thunder'],
        help="Verze modelu: 'lightning' (rychlý) nebo 'thunder' (přesný)."
    )
    parser.add_argument(
        '--gpu', 
        type=str, 
        help="Index GPU, který se má použít (např. '1', '2'). Pokud není zadán, TF si vybere sám (typicky GPU 0)."
    )
    
    args = parser.parse_args()

    # --- Nastavení GPU (MUSÍ BÝT PŘED IMPORTEM TF) ---
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"--- Skript poběží na GPU s indexem: {args.gpu} ---")
    else:
        print(f"--- Varování: --gpu není nastaveno. TensorFlow použije výchozí GPU (obvykle 0). ---")

    print("Kontrola dostupných GPU (podle TF):", tf.config.list_physical_devices('GPU'))

    # --- Načtení modelu ---
    movenet_fn, input_size = load_movenet_model(args.model)

    # --- Spuštění podle režimu ---
    if args.mode == 'image':
        process_image(movenet_fn, input_size, args.input, args.output)
    elif args.mode == 'video':
        process_video(movenet_fn, input_size, args.input, args.output)

if __name__ == "__main__":
    main()
