import cv2
from ultralytics import YOLO
import json
import os
import math
import numpy as np


def filter_valid(data):
    return [d for d in data if d[0] is not None]


def is_valid(confidence):
    """Kontrola, zda je keypoint dostatečně spolehlivý"""
    return confidence >= 0.5  # YOLO používá confidence místo visibility


def calculate_angle(aX, aY, bX, bY, cX, cY):
    """Výpočet úhlu mezi třemi body"""
    vec_c = (bX - aX, bY - aY)
    vec_b = (cX - aX, cY - aY)
    vec_product = vec_c[0] * vec_b[0] + vec_c[1] * vec_b[1]
    scalar_product = abs((vec_b[0] ** 2 + vec_b[1] ** 2) ** 0.5) * abs((vec_c[0] ** 2 + vec_c[1] ** 2) ** 0.5)

    if scalar_product == 0:
        return None

    # Ošetření numerických chyb
    cos_angle = vec_product / scalar_product
    cos_angle = max(-1, min(1, cos_angle))

    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def calculate_right_elbow(keypoints):
    """
    YOLO indexy: Right Shoulder (6), Right Elbow (8), Right Wrist (10)
    MediaPipe ekvivalent: indexy 12, 14, 16
    """
    # Right Shoulder (index 6), Right Elbow (index 8), Right Wrist (index 10)
    if not (is_valid(keypoints[6][2]) and is_valid(keypoints[8][2]) and is_valid(keypoints[10][2])):
        return None

    rx, ry = keypoints[6][0], keypoints[6][1]  # Right Shoulder
    ex, ey = keypoints[8][0], keypoints[8][1]  # Right Elbow
    wx, wy = keypoints[10][0], keypoints[10][1]  # Right Wrist

    return calculate_angle(ex, ey, rx, ry, wx, wy)


def calculate_left_elbow(keypoints):
    """
    YOLO indexy: Left Shoulder (5), Left Elbow (7), Left Wrist (9)
    """
    if not (is_valid(keypoints[5][2]) and is_valid(keypoints[7][2]) and is_valid(keypoints[9][2])):
        return None

    lx, ly = keypoints[5][0], keypoints[5][1]  # Left Shoulder
    ex, ey = keypoints[7][0], keypoints[7][1]  # Left Elbow
    wx, wy = keypoints[9][0], keypoints[9][1]  # Left Wrist

    return calculate_angle(ex, ey, lx, ly, wx, wy)


def calculate_right_shoulder(keypoints):
    """
    YOLO indexy: Right Shoulder (6), Right Elbow (8), Right Hip (12)
    """
    if not (is_valid(keypoints[6][2]) and is_valid(keypoints[8][2]) and is_valid(keypoints[12][2])):
        return None

    sx, sy = keypoints[6][0], keypoints[6][1]  # Right Shoulder
    ex, ey = keypoints[8][0], keypoints[8][1]  # Right Elbow
    hx, hy = keypoints[12][0], keypoints[12][1]  # Right Hip

    return calculate_angle(sx, sy, ex, ey, hx, hy)


def calculate_right_knee(keypoints):
    """
    YOLO indexy: Right Hip (12), Right Knee (14), Right Ankle (16)
    """
    if not (is_valid(keypoints[12][2]) and is_valid(keypoints[14][2]) and is_valid(keypoints[16][2])):
        return None

    hx, hy = keypoints[12][0], keypoints[12][1]  # Right Hip
    kx, ky = keypoints[14][0], keypoints[14][1]  # Right Knee
    ax, ay = keypoints[16][0], keypoints[16][1]  # Right Ankle

    return calculate_angle(kx, ky, hx, hy, ax, ay)


def calculate_right_ankle(keypoints):
    """
    YOLO indexy: Right Knee (14), Right Ankle (16)
    Pro kotník potřebujeme třetí bod - použijeme aproximaci směru chodidla
    """
    if not (is_valid(keypoints[14][2]) and is_valid(keypoints[16][2])):
        return None

    kx, ky = keypoints[14][0], keypoints[14][1]  # Right Knee
    ax, ay = keypoints[16][0], keypoints[16][1]  # Right Ankle

    # Aproximace bodu chodidla (vertikálně pod kotníkem)
    fx, fy = ax, ay + 50  # 50 pixelů dolů od kotníku

    return calculate_angle(ax, ay, kx, ky, fx, fy)


# Inicializace YOLO modelu
model = YOLO('yolo11x-pose.pt')

video_path = "video/RLelb_RLshou_RLknee.mp4"
output_folder = "frames_with_pose"
json_output = []
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0

right_knee = []
right_shoulder = []
right_elbow = []
right_ankle = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLO detekce
    results = model(frame, verbose=False)

    # Zpracování výsledků
    keypoints_data = []
    if len(results) > 0 and results[0].keypoints is not None:
        # Získání keypoints pro první detekovanou osobu
        keypoints = results[0].keypoints.xy[0].cpu().numpy()  # [17, 2] - souřadnice x, y
        confidences = results[0].keypoints.conf[0].cpu().numpy()  # [17] - confidence skóre

        # Vytvoření struktury keypoints [x, y, confidence]
        for i, (kpt, conf) in enumerate(zip(keypoints, confidences)):
            keypoints_data.append([float(kpt[0]), float(kpt[1]), float(conf)])

        # Vykreslení pose na frame
        annotated_frame = results[0].plot()

        # Výpočet úhlů
        right_shoulder.append((calculate_right_shoulder(keypoints_data), frame_id))
        right_elbow.append((calculate_right_elbow(keypoints_data), frame_id))
        right_knee.append((calculate_right_knee(keypoints_data), frame_id))
        right_ankle.append((calculate_right_ankle(keypoints_data), frame_id))

    else:
        # Žádná pose nebyla detekována
        keypoints_data = [[0, 0, 0] for _ in range(17)]  # YOLO má 17 keypoints
        annotated_frame = frame
        right_shoulder.append((None, frame_id))
        right_elbow.append((None, frame_id))
        right_knee.append((None, frame_id))
        right_ankle.append((None, frame_id))

    # Uložení snímku
    image_filename = f"{frame_id:05d}.jpg"
    image_path = os.path.join(output_folder, image_filename)
    cv2.imwrite(image_path, annotated_frame)

    # Příprava dat pro JSON (kompatibilita s původním formátem)
    flat_keypoints = []
    for kpt in keypoints_data:
        flat_keypoints.extend(kpt)  # [x, y, conf, x, y, conf, ...]

    json_output.append({
        "image_id": frame_id,
        "category_id": 0,
        "keypoints": flat_keypoints
    })

    frame_id += 1

cap.release()

# Filtrování platných hodnot
right_shoulder_valid = filter_valid(right_shoulder)
right_elbow_valid = filter_valid(right_elbow)
right_knee_valid = filter_valid(right_knee)
right_ankle_valid = filter_valid(right_ankle)

# Výpis výsledků
print("PRAVE RAMENO!")
if right_shoulder_valid:
    print("Max:", max(right_shoulder_valid, key=lambda x: x[0]))
    print("Min:", min(right_shoulder_valid, key=lambda x: x[0]))

print("PRAVY LOKET!")
if right_elbow_valid:
    print("Max:", max(right_elbow_valid, key=lambda x: x[0]))
    print("Min:", min(right_elbow_valid, key=lambda x: x[0]))

print("PRAVE KOLENO!")
if right_knee_valid:
    print("Max:", max(right_knee_valid, key=lambda x: x[0]))
    print("Min:", min(right_knee_valid, key=lambda x: x[0]))

print("PRAVY KOTNIK!")
if right_ankle_valid:
    print("Max:", max(right_ankle_valid, key=lambda x: x[0]))
    print("Min:", min(right_ankle_valid, key=lambda x: x[0]))

# Ulož JSON
with open("annotations.json", "w") as f:
    json.dump(json_output, f, indent=2)

# Markdown výstup
with open("vysledky_yolo.md", "w", encoding="utf-8") as md:
    def write_section(name, data, label):
        if not data:
            md.write(f"## {name}\n\n**Žádná validní data nebyla nalezena.**\n\n")
            return
        max_angle, max_frame = max(data, key=lambda x: x[0])
        min_angle, min_frame = min(data, key=lambda x: x[0])
        md.write(f"## {name}\n\n")
        md.write(f"**Maximální úhel:** {max_angle:.2f}° (snímek {max_frame})  \n")
        md.write(f"![Max {label}](frames_with_pose/{max_frame:05d}.jpg)\n\n")
        md.write(f"**Minimální úhel:** {min_angle:.2f}° (snímek {min_frame})  \n")
        md.write(f"![Min {label}](frames_with_pose/{min_frame:05d}.jpg)\n\n")


    md.write("# Výsledky analýzy pohybu (YOLO11)\n\n")
    write_section("Pravé rameno", right_shoulder_valid, "rameno")
    write_section("Pravý loket", right_elbow_valid, "loket")
    write_section("Pravé koleno", right_knee_valid, "koleno")
    write_section("Pravý kotník", right_ankle_valid, "kotnik")

print("Success - YOLO11 pose estimation completed!")