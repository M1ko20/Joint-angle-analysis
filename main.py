import cv2
import mediapipe as mp
import json
import os
import math
import numpy as np
from typing import List, Tuple, Optional

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO není nainstalováno. Nainstaluj: pip install ultralytics")


def calculate_angle(ax, ay, bx, by, cx, cy):
    """Spocita uhel mezi tremi body. B je vertex."""
    vec_ba = (ax - bx, ay - by)
    vec_bc = (cx - bx, cy - by)

    dot_product = vec_ba[0] * vec_bc[0] + vec_ba[1] * vec_bc[1]
    mag_ba = math.sqrt(vec_ba[0] ** 2 + vec_ba[1] ** 2)
    mag_bc = math.sqrt(vec_bc[0] ** 2 + vec_bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_angle = dot_product / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # aby to necrashnulo

    return math.degrees(math.acos(cos_angle))


def is_keypoint_ok(keypoints, idx, threshold=0.9):
    """Zkontroluje jestli je keypoint viditelny."""
    return keypoints[idx * 3 + 2] >= threshold


# Mapovani keypoints mezi ruznyma modelama
KEYPOINT_MAPPING = {
    'mediapipe': {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28
    },
    'yolo': {
        # YOLO11 COCO keypoints (indexy 0-16)
        'left_shoulder': 5,  # Left Shoulder
        'right_shoulder': 6,  # Right Shoulder
        'left_elbow': 7,  # Left Elbow
        'right_elbow': 8,  # Right Elbow
        'left_wrist': 9,  # Left Wrist
        'right_wrist': 10,  # Right Wrist
        'left_hip': 11,  # Left Hip
        'right_hip': 12,  # Right Hip
        'left_knee': 13,  # Left Knee
        'right_knee': 14,  # Right Knee
        'left_ankle': 15,  # Left Ankle
        'right_ankle': 16  # Right Ankle
    },
    'alphapose': {
        # TODO: pridat AlphaPose mapping kdyz budes implementovat
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
    },
    'vitpose': {
        # TODO: pridat VitPose mapping kdyz budes implementovat
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
}


def get_joint_angles(keypoints, model_name):
    """Spocita uhly kloubu. Pouzije spravny mapping podle modelu."""
    if model_name not in KEYPOINT_MAPPING:
        print(f"Neznamy model pro mapping: {model_name}")
        return {}

    mapping = KEYPOINT_MAPPING[model_name]
    angles = {}

    # Pravy loket: right_shoulder -> right_elbow -> right_wrist
    try:
        rs_idx = mapping['right_shoulder']
        re_idx = mapping['right_elbow']
        rw_idx = mapping['right_wrist']

        if all(is_keypoint_ok(keypoints, i) for i in [rs_idx, re_idx, rw_idx]):
            sx, sy = keypoints[rs_idx * 3], keypoints[rs_idx * 3 + 1]
            ex, ey = keypoints[re_idx * 3], keypoints[re_idx * 3 + 1]
            wx, wy = keypoints[rw_idx * 3], keypoints[rw_idx * 3 + 1]
            angles['right_elbow'] = calculate_angle(sx, sy, ex, ey, wx, wy)
    except (KeyError, IndexError):
        pass

    # Levy loket: left_shoulder -> left_elbow -> left_wrist
    try:
        ls_idx = mapping['left_shoulder']
        le_idx = mapping['left_elbow']
        lw_idx = mapping['left_wrist']

        if all(is_keypoint_ok(keypoints, i) for i in [ls_idx, le_idx, lw_idx]):
            sx, sy = keypoints[ls_idx * 3], keypoints[ls_idx * 3 + 1]
            ex, ey = keypoints[le_idx * 3], keypoints[le_idx * 3 + 1]
            wx, wy = keypoints[lw_idx * 3], keypoints[lw_idx * 3 + 1]
            angles['left_elbow'] = calculate_angle(sx, sy, ex, ey, wx, wy)
    except (KeyError, IndexError):
        pass

    # Prave rameno: right_elbow -> right_shoulder -> right_hip
    try:
        re_idx = mapping['right_elbow']
        rs_idx = mapping['right_shoulder']
        rh_idx = mapping['right_hip']

        if all(is_keypoint_ok(keypoints, i) for i in [re_idx, rs_idx, rh_idx]):
            ex, ey = keypoints[re_idx * 3], keypoints[re_idx * 3 + 1]
            sx, sy = keypoints[rs_idx * 3], keypoints[rs_idx * 3 + 1]
            hx, hy = keypoints[rh_idx * 3], keypoints[rh_idx * 3 + 1]
            angles['right_shoulder'] = calculate_angle(ex, ey, sx, sy, hx, hy)
    except (KeyError, IndexError):
        pass

    # Prave koleno: right_hip -> right_knee -> right_ankle
    try:
        rh_idx = mapping['right_hip']
        rk_idx = mapping['right_knee']
        ra_idx = mapping['right_ankle']

        if all(is_keypoint_ok(keypoints, i) for i in [rh_idx, rk_idx, ra_idx]):
            hx, hy = keypoints[rh_idx * 3], keypoints[rh_idx * 3 + 1]
            kx, ky = keypoints[rk_idx * 3], keypoints[rk_idx * 3 + 1]
            ax, ay = keypoints[ra_idx * 3], keypoints[ra_idx * 3 + 1]
            angles['right_knee'] = calculate_angle(hx, hy, kx, ky, ax, ay)
    except (KeyError, IndexError):
        pass
    # TODO
    # RAMENO
    """udelat rameno, bod cislo jedno kdyz jsou vsechny pod sebou a zaroven je loket pod ramenem, bod cislo 2 kdyz je loket co nejvyse"""

    # RAMENO (dovnitr)
    """kam az jsem schopny dat rameno dovnitr"""

    # KOLENO (zleva doprava)
    """udelat koleno tak ze vezme nejdrive rovne koleno - kotnik pod kolenem, pak kdyz je kotnik oc nejvice v levo a pak kdyz je co nejvice vpravo"""

    # KOTNIK
    """Body jsou prostrednicek, pata a kotnik. spocitat uhel mezi prostrednikem, kotnikem a kolenem, nejspis z boku """

    return angles


class PoseDetector:
    """Base class pro detektory. Kdyz chces pridat novy model, zdedis tuhle."""

    def __init__(self, model_name):
        self.model_name = model_name

    def detect_keypoints(self, image):
        """Override tuhle metodu. Ma vratit keypoints nebo None."""
        raise NotImplementedError("Implementuj si to sam!")

    def draw_skeleton(self, image, keypoints):
        """Override tuhle taky."""
        return image


class MediaPipeDetector(PoseDetector):
    """MediaPipe implementace. Funguje."""

    def __init__(self):
        super().__init__('mediapipe')
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(static_image_mode=True)

    def detect_keypoints(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if not results.pose_landmarks:
            return None

        h, w, _ = image.shape
        keypoints = []

        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            v = landmark.visibility
            keypoints.extend([x, y, v])

        return keypoints

    def draw_skeleton(self, image, keypoints=None):
        # Jednoduche reseni - spustim detekci znovu pro kresleni
        # Neni to efektivni ale funguje to
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return image


class YOLODetector(PoseDetector):
    """YOLO11 implementace pro pose estimation."""

    def __init__(self, model_path="yolo11x-pose.pt"):
        super().__init__('yolo')

        if not YOLO_AVAILABLE:
            raise ImportError("YOLO není dostupné. Nainstaluj: pip install ultralytics")

        try:
            print(f"Načítám YOLO model: {model_path}")
            self.model = YOLO(model_path)
            print("YOLO model úspěšně načten!")
        except Exception as e:
            print(f"Chyba při načítání YOLO modelu: {e}")
            raise

        # YOLO11 COCO pose connections pro kreslení skeletu
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],  # torso + arms
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],  # arms + face
            [2, 4], [3, 5], [4, 6], [5, 7]  # face to shoulders
        ]

        # Převod na 0-based indexy (YOLO používá 1-based v dokumentaci)
        self.skeleton = [[s[0] - 1, s[1] - 1] for s in self.skeleton]

    def detect_keypoints(self, image):
        """Detekuje keypoints pomocí YOLO11."""
        try:
            results = self.model(image, verbose=False)

            if not results or len(results) == 0:
                return None

            # Vezmi první detekci (nejlepší skóre)
            result = results[0]

            if result.keypoints is None or len(result.keypoints.data) == 0:
                return None

            # YOLO vrací keypoints ve formátu [n_detections, n_keypoints, 3]
            # kde 3 = [x, y, confidence]
            keypoints_data = result.keypoints.data[0]  # první (nejlepší) detekce

            h, w = image.shape[:2]
            keypoints = []

            # Převod do formátu [x, y, confidence, x, y, confidence, ...]
            for kp in keypoints_data:
                x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                keypoints.extend([x, y, conf])

            return keypoints

        except Exception as e:
            print(f"Chyba při YOLO detekci: {e}")
            return None

    def draw_skeleton(self, image, keypoints=None):
        """Nakreslí skeleton na obrázek."""
        if keypoints is None:
            # Pokud nejsou keypoints poskytnuty, spusť detekci
            keypoints = self.detect_keypoints(image)

        if keypoints is None or len(keypoints) < 17 * 3:
            return image

        image_copy = image.copy()

        # Nakresli keypoints
        for i in range(17):  # YOLO má 17 keypoints
            x = int(keypoints[i * 3])
            y = int(keypoints[i * 3 + 1])
            conf = keypoints[i * 3 + 2]

            if conf > 0.5:  # pouze viditelné keypoints
                cv2.circle(image_copy, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(image_copy, str(i), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Nakresli skeleton connections
        for connection in self.skeleton:
            kp1_idx, kp2_idx = connection

            # Zkontroluj validitu indexů
            if kp1_idx >= 17 or kp2_idx >= 17:
                continue

            x1 = int(keypoints[kp1_idx * 3])
            y1 = int(keypoints[kp1_idx * 3 + 1])
            conf1 = keypoints[kp1_idx * 3 + 2]

            x2 = int(keypoints[kp2_idx * 3])
            y2 = int(keypoints[kp2_idx * 3 + 1])
            conf2 = keypoints[kp2_idx * 3 + 2]

            # Nakresli čáru pouze pokud jsou oba body viditelné
            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(image_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)

        return image_copy


class AlphaPoseDetector(PoseDetector):
    """AlphaPose - k implementaci."""

    def __init__(self):
        super().__init__('alphapose')
        print("AlphaPose detector - TODO")
        # self.model = load_alphapose_somehow()

    def detect_keypoints(self, image):
        # TODO: implementovat AlphaPose detekci
        print("AlphaPose detection not implemented yet")
        return None


class VitPoseDetector(PoseDetector):
    """VitPose - k implementaci."""

    def __init__(self):
        super().__init__('vitpose')
        print("VitPose detector - TODO")
        # self.model = load_vitpose_somehow()

    def detect_keypoints(self, image):
        # TODO: implementovat VitPose detekci
        print("VitPose detection not implemented yet")
        return None


def get_detector(model_name):
    """Factory funkce. Vrati detektor podle nazvu."""
    detectors = {
        'mediapipe': MediaPipeDetector,
        'yolo': YOLODetector,
        'alphapose': AlphaPoseDetector,
        'vitpose': VitPoseDetector
    }

    if model_name not in detectors:
        print(f"Neznamy model: {model_name}")
        print(f"Dostupne: {list(detectors.keys())}")
        return MediaPipeDetector()  # fallback

    try:
        return detectors[model_name]()
    except Exception as e:
        print(f"Chyba při vytváření detektoru {model_name}: {e}")
        print("Použiji MediaPipe jako fallback")
        return MediaPipeDetector()


def analyze_video(video_path, detector, output_folder="frames_with_pose"):
    """Hlavni funkce. Analyzuje video a vrati vysledky."""

    os.makedirs(output_folder, exist_ok=True)

    # Uloziste pro uhly
    all_angles = {
        'right_elbow': [],
        'left_elbow': [],
        'right_shoulder': [],
        'right_knee': []
    }

    json_data = []
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    print(f"Zpracovávám video s modelem {detector.model_name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detekce keypoints
        keypoints = detector.detect_keypoints(frame)

        if keypoints:
            # Nakresli skeleton
            frame = detector.draw_skeleton(frame, keypoints)

            # Spocitaj uhly s pouzitim spravneho mappingu
            angles = get_joint_angles(keypoints, detector.model_name)

            # Uloz platne uhly
            for joint, angle in angles.items():
                all_angles[joint].append((angle, frame_num))
        else:
            # Prazdne keypoints pro JSON - velikost zavisi na modelu
            if detector.model_name == 'mediapipe':
                keypoints = [0] * (33 * 3)  # MediaPipe ma 33 bodu
            elif detector.model_name == 'yolo':
                keypoints = [0] * (17 * 3)  # YOLO ma 17 bodu
            else:
                keypoints = [0] * (17 * 3)  # default pro ostatni

        # Uloz frame
        frame_file = f"{frame_num:05d}.jpg"
        cv2.imwrite(os.path.join(output_folder, frame_file), frame)

        # JSON data
        json_data.append({
            "image_id": frame_num,
            "category_id": 0,
            "keypoints": keypoints,
            "model": detector.model_name  # pridano info o modelu
        })

        frame_num += 1

        # Progress info
        if frame_num % 30 == 0:
            print(f"Zpracováno {frame_num} snímků...")

    cap.release()

    # Uloz JSON
    json_file = f"annotations_{detector.model_name}.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Model {detector.model_name}: Zpracováno {frame_num} snímků")

    return all_angles


def print_results(all_angles):
    """Vytiskne vysledky do konzole."""

    joint_names = {
        'right_elbow': 'PRAVY LOKET',
        'left_elbow': 'LEVY LOKET',
        'right_shoulder': 'PRAVE RAMENO',
        'right_knee': 'PRAVE KOLENO'
    }

    for joint, angles in all_angles.items():
        if not angles:
            continue

        print(f"\n{joint_names.get(joint, joint)}!")
        max_angle, max_frame = max(angles, key=lambda x: x[0])
        min_angle, min_frame = min(angles, key=lambda x: x[0])
        print(f"Max: {max_angle:.2f}° (frame {max_frame})")
        print(f"Min: {min_angle:.2f}° (frame {min_frame})")


def save_markdown_report(all_results, output_file="vysledky.md"):
    """Ulozi markdown report pro vsechny modely."""

    joint_names = {
        'right_elbow': 'Pravý loket',
        'left_elbow': 'Levý loket',
        'right_shoulder': 'Pravé rameno',
        'right_knee': 'Pravé koleno'
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Výsledky analýzy pohybu - Porovnání modelů\n\n")

        for model_name, data in all_results.items():
            all_angles = data['angles']
            frames_folder = data['frames_folder']

            f.write(f"# {model_name.upper()} Model\n\n")

            for joint, angles in all_angles.items():
                if not angles:
                    continue

                name = joint_names.get(joint, joint)
                max_angle, max_frame = max(angles, key=lambda x: x[0])
                min_angle, min_frame = min(angles, key=lambda x: x[0])

                f.write(f"## {name} ({model_name})\n\n")
                f.write(f"**Maximální úhel:** {max_angle:.2f}° (snímek {max_frame})  \n")
                f.write(f"![Max {joint}]({frames_folder}/{max_frame:05d}.jpg)\n\n")
                f.write(f"**Minimální úhel:** {min_angle:.2f}° (snímek {min_frame})  \n")
                f.write(f"![Min {joint}]({frames_folder}/{min_frame:05d}.jpg)\n\n")

            f.write("---\n\n")


# Main kod
if __name__ == "__main__":
    video_path = "video/t03.mp4"

    # Modely k testování (v pořadí MediaPipe -> YOLO)
    models_to_test = ["mediapipe", "yolo"]

    # Uložiště pro všechny výsledky
    all_results = {}

    print("=== ZAČÍNÁM ANALÝZU PRO VŠECHNY MODELY ===\n")

    for model_name in models_to_test:
        print(f"{'=' * 50}")
        print(f"TESTUJU MODEL: {model_name.upper()}")
        print(f"{'=' * 50}")

        try:
            # Vytvor detektor
            detector = get_detector(model_name)

            # Nastav výstupní složku pro tento model
            output_folder = f"frames_with_pose_{model_name}"

            # Analyzuj video
            results = analyze_video(video_path, detector, output_folder)

            # Ulož výsledky
            all_results[model_name] = {
                'angles': results,
                'frames_folder': output_folder
            }

            # Vypis výsledky pro tento model
            print(f"\n--- VÝSLEDKY PRO {model_name.upper()} ---")
            print_results(results)

        except Exception as e:
            print(f"CHYBA při analýze s modelem {model_name}: {e}")
            continue

        print(f"\nModel {model_name} dokončen!\n")

    # Ulož souhrnný report
    if all_results:
        save_markdown_report(all_results)
        print("=" * 50)
        print("HOTOVO! Všechny modely otestovány.")
        print("Podívej se na vysledky.md pro porovnání")
        print("=" * 50)

        # Vypiš souhrnné informace
        print("\nSOUHRNNÉ INFORMACE:")
        for model_name, data in all_results.items():
            angles = data['angles']
            total_detections = sum(len(joint_angles) for joint_angles in angles.values())
            print(f"- {model_name.upper()}: {total_detections} úspěšných detekcí úhlů")
    else:
        print("ŽÁDNÉ VÝSLEDKY! Zkontroluj video cestu a modely.")