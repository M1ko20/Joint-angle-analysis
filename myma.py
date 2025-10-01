import cv2
import mediapipe as mp
import json
import os
import math


def filter_valid(data):
    return [d for d in data if d[0] is not None]


def is_valid(visibility):
    return visibility >= 0.9


def calculate_angle(aX, aY, bX, bY, cX, cY):
    vec_c = (bX - aX, bY - aY)
    vec_b = (cX - aX, cY - aY)
    vec_product = vec_c[0] * vec_b[0] + vec_c[1] * vec_b[1]
    scalar_product = abs((vec_b[0] ** 2 + vec_b[1] ** 2) ** 0.5) * abs((vec_c[0] ** 2 + vec_c[1] ** 2) ** 0.5)
    angle_rad = math.acos(vec_product / scalar_product)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def calculate_right_elbow(keypoints):
    if not (is_valid(keypoints[12 * 3 + 2]) and is_valid(keypoints[14 * 3 + 2]) and is_valid(keypoints[16 * 3 + 2])):
        return None
    rx, ry = keypoints[12 * 3], keypoints[12 * 3 + 1]
    ex, ey = keypoints[14 * 3], keypoints[14 * 3 + 1]
    wx, wy = keypoints[16 * 3], keypoints[16 * 3 + 1]
    return calculate_angle(ex, ey, rx, ry, wx, wy)


def calculate_left_elbow(keypoints):
    if not (is_valid(keypoints[11 * 3 + 2]) and is_valid(keypoints[13 * 3 + 2]) and is_valid(keypoints[15 * 3 + 2])):
        return None
    lx, ly = keypoints[11 * 3], keypoints[11 * 3 + 1]
    ex, ey = keypoints[13 * 3], keypoints[13 * 3 + 1]
    wx, wy = keypoints[15 * 3], keypoints[15 * 3 + 1]
    return calculate_angle(ex, ey, lx, ly, wx, wy)


def calculate_right_shoulder(keypoints):
    if not (is_valid(keypoints[12 * 3 + 2]) and is_valid(keypoints[14 * 3 + 2]) and is_valid(keypoints[24 * 3 + 2])):
        return None
    sx, sy = keypoints[12 * 3], keypoints[12 * 3 + 1]
    ex, ey = keypoints[14 * 3], keypoints[14 * 3 + 1]
    hx, hy = keypoints[24 * 3], keypoints[24 * 3 + 1]
    return calculate_angle(sx, sy, ex, ey, hx, hy)


def calculate_right_knee(keypoints):
    if not (is_valid(keypoints[26 * 3 + 2]) and is_valid(keypoints[28 * 3 + 2]) and is_valid(keypoints[24 * 3 + 2])):
        return None
    kx, ky = keypoints[26 * 3], keypoints[26 * 3 + 1]
    ax, ay = keypoints[28 * 3], keypoints[28 * 3 + 1]
    hx, hy = keypoints[24 * 3], keypoints[24 * 3 + 1]
    return calculate_angle(kx, ky, ax, ay, hx, hy)


def calculate_right_ankle(keypoints):
    if not (is_valid(keypoints[26 * 3 + 2]) and is_valid(keypoints[28 * 3 + 2]) and is_valid(keypoints[32 * 3 + 2])):
        return None
    kx, ky = keypoints[26 * 3], keypoints[26 * 3 + 1]
    ax, ay = keypoints[28 * 3], keypoints[28 * 3 + 1]
    fx, fy = keypoints[32 * 3], keypoints[32 * 3 + 1]
    return calculate_angle(ax, ay, kx, ky, fx, fy)


# Inicializace
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=True)

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

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    results = pose.process(image_rgb)
    keypoints = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            v = landmark.visibility
            keypoints.extend([x, y, v])

        right_shoulder.append((calculate_right_shoulder(keypoints), frame_id))
        right_elbow.append((calculate_right_elbow(keypoints), frame_id))
        right_knee.append((calculate_right_knee(keypoints), frame_id))
        right_ankle.append((calculate_right_ankle(keypoints), frame_id))
    else:
        keypoints = [0] * (33 * 3)

    image_filename = f"{frame_id:05d}.jpg"
    image_path = os.path.join(output_folder, image_filename)
    cv2.imwrite(image_path, frame)

    json_output.append({
        "image_id": frame_id,
        "category_id": 0,
        "keypoints": keypoints
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
with open("vysledky.md", "w", encoding="utf-8") as md:
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

    md.write("# Výsledky analýzy pohybu\n\n")
    write_section("Pravé rameno", right_shoulder_valid, "rameno")
    write_section("Pravý loket", right_elbow_valid, "loket")
    write_section("Pravé koleno", right_knee_valid, "koleno")
    write_section("Pravý kotník", right_ankle_valid, "kotnik")

print("Success")
