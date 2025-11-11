#!/usr/bin/env python3
"""
Pokročilá 3D analýza polohy těla s využitím MediaPipe World Landmarks
"""

import cv2
import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
import sys

# Import vlastního VIDEO pose detectoru
# Používáme VideoPoseDetector pro správné nastavení (static_image_mode=False)
from video_pose_detector import VideoPoseDetector

# Globální proměnná pro typ detektoru (VŽDY MEDIAPIPE pro 3D)
CURRENT_DETECTOR_TYPE = "mediapipe"

# Globální proměnná pro custom confidence threshold
CUSTOM_CONFIDENCE_THRESHOLD = 0.5


def is_valid(visibility, threshold=None):
    """
    Kontroluje, zda je bod dostatečně viditelný (confidence)
    """
    # Použij globální threshold, pokud je nastaven
    if threshold is None and CUSTOM_CONFIDENCE_THRESHOLD is not None:
        threshold = CUSTOM_CONFIDENCE_THRESHOLD
    
    # Fallback, pokud není nastaven ani globálně
    if threshold is None:
        threshold = 0.5 # Default pro MediaPipe
    
    return visibility >= threshold


def calculate_angle_3d(a, b, c):
    """
    Vypočítá 3D úhel mezi třemi body (A-B-C, kde B je vrchol úhlu)
    Body 'a', 'b', 'c' jsou 3D body (Landmarky nebo numpy array [x, y, z])
    """
    try:
        # Převedení Landmark objektů na numpy pole, pokud je to nutné
        if not isinstance(a, np.ndarray):
            a = np.array([a.x, a.y, a.z])
        if not isinstance(b, np.ndarray):
            b = np.array([b.x, b.y, b.z])
        if not isinstance(c, np.ndarray):
            c = np.array([c.x, c.y, c.z])
        
        # Vektory BA a BC
        vec_ba = a - b
        vec_bc = c - b

        # Skalární součin
        dot_product = np.dot(vec_ba, vec_bc)

        # Magnitudy vektorů
        norm_ba = np.linalg.norm(vec_ba)
        norm_bc = np.linalg.norm(vec_bc)
        
        # Zabraň dělení nulou
        if norm_ba == 0 or norm_bc == 0:
            return None

        # Výpočet úhlu (cos_theta)
        cos_theta = dot_product / (norm_ba * norm_bc)

        # Ošetření numerických chyb (hodnoty mírně mimo <-1, 1>)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Převod na stupně
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
        
    except Exception as e:
        print(f"Chyba při výpočtu 3D úhlu: {e}")
        return None


def draw_angle_arc(frame, center, point1, point2, angle, radius=30, color=(0, 255, 255)):
    """Vykreslí oblouk znázorňující úhel (2D KRESLENÍ)"""
    # Výpočet směrových vektorů
    vec1 = (point1[0] - center[0], point1[1] - center[1])
    vec2 = (point2[0] - center[0], point2[1] - center[1])
    
    # Normalizace vektorů
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return
    
    vec1_norm = (vec1[0]/mag1, vec1[1]/mag1)
    vec2_norm = (vec2[0]/mag2, vec2[1]/mag2)
    
    # Výpočet úhlů
    angle1 = math.degrees(math.atan2(vec1_norm[1], vec1_norm[0]))
    angle2 = math.degrees(math.atan2(vec2_norm[1], vec2_norm[0]))
    
    # Zajištění správného směru oblouku
    if angle1 < 0:
        angle1 += 360
    if angle2 < 0:
        angle2 += 360
    
    start_angle = min(angle1, angle2)
    end_angle = max(angle1, angle2)
    
    # Pokud je rozdíl větší než 180°, otočíme směr
    if end_angle - start_angle > 180:
        start_angle, end_angle = end_angle, start_angle + 360
    
    # Vykreslení oblouku
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, 2)


# --- Funkce pro výpočet 3D úhlů ---
# Vstupem je 'landmarks', což je seznam 33 MediaPipe Landmark objektů

def calculate_right_elbow_3d(landmarks):
    """Pravý loket: rameno-loket-zápěstí (3D)"""
    shoulder = landmarks[12]
    elbow = landmarks[14]
    wrist = landmarks[16]
    
    if not (is_valid(shoulder.visibility) and is_valid(elbow.visibility) and is_valid(wrist.visibility)):
        return None
    
    return calculate_angle_3d(shoulder, elbow, wrist)


def calculate_left_elbow_3d(landmarks):
    """Levý loket: rameno-loket-zápěstí (3D)"""
    shoulder = landmarks[11]
    elbow = landmarks[13]
    wrist = landmarks[15]
    
    if not (is_valid(shoulder.visibility) and is_valid(elbow.visibility) and is_valid(wrist.visibility)):
        return None
    
    return calculate_angle_3d(shoulder, elbow, wrist)


def calculate_right_shoulder_3d(landmarks):
    """Pravé rameno: kyčel-rameno-loket (3D)"""
    hip = landmarks[24]
    shoulder = landmarks[12]
    elbow = landmarks[14]
    
    if not (is_valid(hip.visibility) and is_valid(shoulder.visibility) and is_valid(elbow.visibility)):
        return None
    
    return calculate_angle_3d(hip, shoulder, elbow)


def calculate_left_shoulder_3d(landmarks):
    """Levé rameno: kyčel-rameno-loket (3D)"""
    hip = landmarks[23]
    shoulder = landmarks[11]
    elbow = landmarks[13]
    
    if not (is_valid(hip.visibility) and is_valid(shoulder.visibility) and is_valid(elbow.visibility)):
        return None
    
    return calculate_angle_3d(hip, shoulder, elbow)


def calculate_right_hip_3d(landmarks):
    """Pravá kyčel: rameno-kyčel-koleno (3D)"""
    shoulder = landmarks[12]
    hip = landmarks[24]
    knee = landmarks[26]
    
    if not (is_valid(shoulder.visibility) and is_valid(hip.visibility) and is_valid(knee.visibility)):
        return None
    
    return calculate_angle_3d(shoulder, hip, knee)


def calculate_left_hip_3d(landmarks):
    """Levá kyčel: rameno-kyčel-koleno (3D)"""
    shoulder = landmarks[11]
    hip = landmarks[23]
    knee = landmarks[25]
    
    if not (is_valid(shoulder.visibility) and is_valid(hip.visibility) and is_valid(knee.visibility)):
        return None
    
    return calculate_angle_3d(shoulder, hip, knee)


def calculate_right_knee_3d(landmarks):
    """Pravé koleno: kyčel-koleno-kotník (3D)"""
    hip = landmarks[24]
    knee = landmarks[26]
    ankle = landmarks[28]
    
    if not (is_valid(hip.visibility) and is_valid(knee.visibility) and is_valid(ankle.visibility)):
        return None
    
    return calculate_angle_3d(hip, knee, ankle)


def calculate_left_knee_3d(landmarks):
    """Levé koleno: kyčel-koleno-kotník (3D)"""
    hip = landmarks[23]
    knee = landmarks[25]
    ankle = landmarks[27]
    
    if not (is_valid(hip.visibility) and is_valid(knee.visibility) and is_valid(ankle.visibility)):
        return None
    
    return calculate_angle_3d(hip, knee, ankle)


def draw_angle_on_frame(frame, keypoints_2d, angle, joint_indices, text_position, joint_name):
    """
    Vykreslí úhel do snímku (používá 2D keypoints pro pozici kreslení)
    'keypoints_2d' je ploché numpy pole [x, y, v, ...] z PoseDetector
    'angle' je hodnota úhlu (vypočítaná z 3D)
    """
    if angle is None:
        return
    
    # Zkontroluj viditelnost 2D bodů pro kreslení
    if all(is_valid(keypoints_2d[i * 3 + 2]) for i in joint_indices):
        # Získej 2D souřadnice bodů
        points = []
        for i in joint_indices:
            x = int(keypoints_2d[i * 3])
            y = int(keypoints_2d[i * 3 + 1])
            points.append((x, y))
        
        # Vykreslí linky mezi body
        cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
        cv2.line(frame, points[1], points[2], (0, 255, 0), 2)
        
        # Vykreslí kruhy na kloubech
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
        
        # Vykreslí úhlový oblouk
        center = points[1]  # Středový bod (vrchol úhlu)
        draw_angle_arc(frame, center, points[0], points[2], angle, radius=40)
        
        # Text s úhlem
        text = f"{joint_name}: {angle:.1f} Stupnu"
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def create_graphs(angles_data, output_folder, fps):
    """Vytvoří grafy pro každý kloub"""
    graphs_folder = os.path.join(output_folder, "graphs_3d")
    os.makedirs(graphs_folder, exist_ok=True)
    
    for joint_name, angles in angles_data.items():
        if not angles:
            continue
            
        # Filtruj platné hodnoty
        valid_data = [(angle, frame_id) for angle, frame_id in angles if angle is not None]
        if not valid_data:
            continue
            
        angles_list, frames_list = zip(*valid_data)
        time_list = [frame / fps for frame in frames_list]  # Převod na čas v sekundách
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_list, angles_list, 'b-', linewidth=2)
        plt.title(f'Vývoj 3D úhlu - {joint_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Čas (sekundy)', fontsize=12)
        plt.ylabel('Úhel (stupně)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Uložení grafu
        graph_path = os.path.join(graphs_folder, f"{joint_name.lower().replace(' ', '_')}.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()


def save_results(angles_data, output_folder, fps, detector_type):
    """Uloží výsledky do souborů"""
    
    # .txt soubor s min/max hodnotami
    txt_path = os.path.join(output_folder, "min_max_angles_3d.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"3D Analýza úhlů kloubů - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Detektor: {detector_type.upper()} (World Landmarks)\n")
        f.write("="*60 + "\n\n")
        
        for joint_name, angles in angles_data.items():
            valid_data = [angle for angle, _ in angles if angle is not None]
            if valid_data:
                min_angle = min(valid_data)
                max_angle = max(valid_data)
                avg_angle = sum(valid_data) / len(valid_data)
                
                f.write(f"{joint_name}:\n")
                f.write(f"  Minimální 3D úhel: {min_angle:.2f}°\n")
                f.write(f"  Maximální 3D úhel: {max_angle:.2f}°\n")
                f.write(f"  Průměrný 3D úhel: {avg_angle:.2f}°\n")
                f.write(f"  Počet platných měření: {len(valid_data)}\n\n")
            else:
                f.write(f"{joint_name}: Žádná platná data\n\n")
    
    # .json soubor s vývojem v čase
    json_data = []
    for joint_name, angles in angles_data.items():
        for angle, frame_id in angles:
            if angle is not None:
                json_data.append({
                    "joint": joint_name,
                    "frame": frame_id,
                    "time_seconds": frame_id / fps,
                    "angle_degrees_3d": angle,
                    "detector": detector_type
                })
    
    json_path = os.path.join(output_folder, "angles_timeline_3d.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def analyze_video_3d(video_path, output_folder="pose_analysis_output_3d"):
    """Hlavní funkce pro 3D analýzu videa (pouze MediaPipe)"""
    
    detector_type = "mediapipe"
    global CURRENT_DETECTOR_TYPE
    CURRENT_DETECTOR_TYPE = detector_type
    
    # Vytvoření výstupních složek
    os.makedirs(output_folder, exist_ok=True)
    frames_folder = os.path.join(output_folder, "annotated_frames_3d")
    os.makedirs(frames_folder, exist_ok=True)
    
    # Inicializace VideoPose detectoru
    try:
        # Použijeme VideoPoseDetector, který nastaví static_image_mode=False
        pose_detector = VideoPoseDetector(
            detector_type=detector_type,
            confidence_threshold=CUSTOM_CONFIDENCE_THRESHOLD or 0.5
        )
        print(f"MediaPipe (Video Režim) úspěšně inicializován pro 3D analýzu")
    except Exception as e:
        print(f"Chyba při inicializaci MediaPipe: {e}")
        return None
    
    # Otevření videa
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Chyba: Nelze otevřít video {video_path}")
        return
    
    # Získání informací o videu
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 Zpracovávám video: {video_path}")
    print(f"📊 FPS: {fps}, Rozměry: {width}x{height}, Celkové snímky: {total_frames}")
    print(f"🔍 Detektor: MediaPipe (3D World Landmarks)")
    
    # VideoWriter pro výstupní video
    output_video_path = os.path.join(output_folder, f"analyzed_video_3d.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Slovník pro ukládání úhlů
    angles_data = {
        "Pravý loket": [],
        "Levý loket": [],
        "Pravé rameno": [],
        "Levé rameno": [],
        "Pravá kyčel": [],
        "Levá kyčel": [],
        "Pravé koleno": [],
        "Levé koleno": []
    }
    
    frame_id = 0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Detekce pose
            # keypoints_2d = 2D souřadnice v pixelech (pro kreslení)
            # detection_result = Surový výsledek z MediaPipe (obsahuje 3D landmarks)
            keypoints_2d, detection_result = pose_detector.detect_pose(frame)
            
            # Kontrola, zda máme platný výsledek A ZDA OBSAHUJE 3D LANDMARKS
            if (keypoints_2d is not None and 
                detection_result and 
                hasattr(detection_result, 'pose_world_landmarks') and
                detection_result.pose_world_landmarks):
                
                # Získání 3D landmarků
                landmarks_3d = detection_result.pose_world_landmarks.landmark
                
                # Vykreslení 2D pose landmarks do snímku
                pose_detector.draw_landmarks(frame, detection_result)
                
                # Výpočet 3D úhlů
                right_elbow = calculate_right_elbow_3d(landmarks_3d)
                left_elbow = calculate_left_elbow_3d(landmarks_3d)
                right_shoulder = calculate_right_shoulder_3d(landmarks_3d)
                left_shoulder = calculate_left_shoulder_3d(landmarks_3d)
                right_hip = calculate_right_hip_3d(landmarks_3d)
                left_hip = calculate_left_hip_3d(landmarks_3d)
                right_knee = calculate_right_knee_3d(landmarks_3d)
                left_knee = calculate_left_knee_3d(landmarks_3d)
                
                # Uložení úhlů
                angles_data["Pravý loket"].append((right_elbow, frame_id))
                angles_data["Levý loket"].append((left_elbow, frame_id))
                angles_data["Pravé rameno"].append((right_shoulder, frame_id))
                angles_data["Levé rameno"].append((left_shoulder, frame_id))
                angles_data["Pravá kyčel"].append((right_hip, frame_id))
                angles_data["Levá kyčel"].append((left_hip, frame_id))
                angles_data["Pravé koleno"].append((right_knee, frame_id))
                angles_data["Levé koleno"].append((left_knee, frame_id))
                
                # Vykreslení úhlů do snímku
                # Používáme keypoints_2d pro určení pozice kreslení
                draw_angle_on_frame(frame, keypoints_2d, right_elbow, [12, 14, 16], (10, 30), "R Loket (3D)")
                draw_angle_on_frame(frame, keypoints_2d, left_elbow, [11, 13, 15], (10, 60), "L Loket (3D)")
                draw_angle_on_frame(frame, keypoints_2d, right_shoulder, [24, 12, 14], (10, 90), "R Rameno (3D)")
                draw_angle_on_frame(frame, keypoints_2d, left_shoulder, [23, 11, 13], (10, 120), "L Rameno (3D)")
                draw_angle_on_frame(frame, keypoints_2d, right_hip, [12, 24, 26], (10, 150), "R Kycel (3D)")
                draw_angle_on_frame(frame, keypoints_2d, left_hip, [11, 23, 25], (10, 180), "L Kycel (3D)")
                draw_angle_on_frame(frame, keypoints_2d, right_knee, [24, 26, 28], (10, 210), "R Koleno (3D)")
                draw_angle_on_frame(frame, keypoints_2d, left_knee, [23, 25, 27], (10, 240), "L Koleno (3D)")
                
                # Označení detektoru
                cv2.putText(frame, "MediaPipe 3D", (width - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Uložení snímku
            frame_filename = f"{frame_id:05d}.jpg"
            frame_path = os.path.join(frames_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # Zápis do výstupního videa
            out.write(frame)
            
            # Progress
            if frame_id % 30 == 0:
                progress = (frame_id / total_frames) * 100
                print(f"⏳ Zpracováno: {progress:.1f}% ({frame_id}/{total_frames})")
            
            frame_id += 1
    
    except KeyboardInterrupt:
        print("\n🚫 Analýza přerušena uživatelem")
    
    finally:
        # Uvolnění zdrojů
        cap.release()
        out.release()
        pose_detector.close()
    
    # Uložení výsledků
    print("💾 Ukládám 3D výsledky...")
    save_results(angles_data, output_folder, fps, detector_type)
    
    # Vytvoření grafů
    print("📈 Vytvářím 3D grafy...")
    create_graphs(angles_data, output_folder, fps)
    
    print(f"✅ 3D Analýza dokončena! Výsledky uloženy v: {output_folder}")
    print(f"🎥 Výstupní video: {output_video_path}")
    
    return angles_data


def main():
    """Hlavní funkce s podporou argumentů příkazové řádky"""
    parser = argparse.ArgumentParser(description="3D Analýza polohy těla pomocí MediaPipe World Landmarks")
    parser.add_argument("--video", "-v", type=str, default="video/RLelb_RLshou_RLknee.mp4",
                       help="Cesta k video souboru")
    parser.add_argument("--output", "-o", type=str, default="pose_analysis_output_3d",
                       help="Výstupní složka")
    
    args = parser.parse_args()
    
    # Kontrola existence video souboru
    if not os.path.exists(args.video):
        print(f"❌ Video soubor '{args.video}' neexistuje!")
        return 1
    
    # Spuštění analýzy
    print(f"\n🚀 Spouštím 3D analýzu (pouze MediaPipe)...")
    angles_data = analyze_video_3d(args.video, args.output)
    
    if angles_data:
        # Výpis základních statistik
        print("\n" + "="*60)
        print("📊 SOUHRN 3D VÝSLEDKŮ")
        print("="*60)
        print(f"🔍 Detektor: MediaPipe (3D World Landmarks)")
        
        for joint_name, angles in angles_data.items():
            valid_data = [angle for angle, _ in angles if angle is not None]
            if valid_data:
                min_angle = min(valid_data)
                max_angle = max(valid_data)
                avg_angle = sum(valid_data) / len(valid_data)
                
                print(f"\n{joint_name}:")
                print(f"  Min (3D): {min_angle:.2f}°")
                print(f"  Max (3D): {max_angle:.2f}°")
                print(f"  Průměr (3D): {avg_angle:.2f}°")
                print(f"  Počet měření: {len(valid_data)}")
            else:
                print(f"\n{joint_name}: Žádná platná data")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())