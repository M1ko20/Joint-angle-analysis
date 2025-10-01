#!/usr/bin/env python3
"""
Pokročilá analýza polohy těla s podporou MediaPipe a OpenPose
Autor: GitHub Copilot
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

# Import vlastního pose detectoru
from pose_detector import PoseDetector, get_available_detectors, select_detector

# Globální proměnná pro typ detektoru
CURRENT_DETECTOR_TYPE = "mediapipe"


def filter_valid(data):
    """Filtruje platné hodnoty (bez None)"""
    return [d for d in data if d[0] is not None]


def is_valid(visibility, threshold=None, detector_type=None):
    """Kontroluje, zda je bod dostatečně viditelný s adaptivním threshold podle detektoru"""
    if detector_type is None:
        detector_type = CURRENT_DETECTOR_TYPE
        
    if threshold is None:
        # Adaptivní threshold podle detektoru
        if detector_type.startswith("movenet"):
            threshold = 0.3  # Nižší threshold pro MoveNet
        elif detector_type == "openpose":
            threshold = 0.4  # Střední threshold pro OpenPose
        else:  # mediapipe
            threshold = 0.8  # Středně přísný pro MediaPipe
    
    return visibility >= threshold


def calculate_angle(aX, aY, bX, bY, cX, cY):
    """Vypočítá úhel mezi třemi body (A-B-C, kde B je vrchol úhlu)"""
    # Vektory BA a BC
    vec_ba = (aX - bX, aY - bY)
    vec_bc = (cX - bX, cY - bY)
    
    # Skalární součin
    dot_product = vec_ba[0] * vec_bc[0] + vec_ba[1] * vec_bc[1]
    
    # Délky vektorů
    mag_ba = math.sqrt(vec_ba[0]**2 + vec_ba[1]**2)
    mag_bc = math.sqrt(vec_bc[0]**2 + vec_bc[1]**2)
    
    if mag_ba == 0 or mag_bc == 0:
        return None
        
    # Kosinus úhlu
    cos_angle = dot_product / (mag_ba * mag_bc)
    
    # Zajištění, že kosinus je v rozmezí [-1, 1]
    cos_angle = max(-1, min(1, cos_angle))
    
    # Úhel v radiánech a převod na stupně
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def draw_angle_arc(frame, center, point1, point2, angle, radius=30, color=(0, 255, 255)):
    """Vykreslí oblouk znázorňující úhel"""
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


# Funkce pro výpočet úhlů kloubů
def calculate_right_elbow(keypoints):
    """Pravý loket: rameno-loket-zápěstí"""
    if not (is_valid(keypoints[12 * 3 + 2]) and is_valid(keypoints[14 * 3 + 2]) and is_valid(keypoints[16 * 3 + 2])):
        return None
    
    shoulder_x, shoulder_y = keypoints[12 * 3], keypoints[12 * 3 + 1]
    elbow_x, elbow_y = keypoints[14 * 3], keypoints[14 * 3 + 1]
    wrist_x, wrist_y = keypoints[16 * 3], keypoints[16 * 3 + 1]
    
    return calculate_angle(shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y)


def calculate_left_elbow(keypoints):
    """Levý loket: rameno-loket-zápěstí"""
    if not (is_valid(keypoints[11 * 3 + 2]) and is_valid(keypoints[13 * 3 + 2]) and is_valid(keypoints[15 * 3 + 2])):
        return None
    
    shoulder_x, shoulder_y = keypoints[11 * 3], keypoints[11 * 3 + 1]
    elbow_x, elbow_y = keypoints[13 * 3], keypoints[13 * 3 + 1]
    wrist_x, wrist_y = keypoints[15 * 3], keypoints[15 * 3 + 1]
    
    return calculate_angle(shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y)


def calculate_right_shoulder(keypoints):
    """Pravé rameno: kyčel-rameno-loket"""
    if not (is_valid(keypoints[24 * 3 + 2]) and is_valid(keypoints[12 * 3 + 2]) and is_valid(keypoints[14 * 3 + 2])):
        return None
    
    hip_x, hip_y = keypoints[24 * 3], keypoints[24 * 3 + 1]
    shoulder_x, shoulder_y = keypoints[12 * 3], keypoints[12 * 3 + 1]
    elbow_x, elbow_y = keypoints[14 * 3], keypoints[14 * 3 + 1]
    
    return calculate_angle(hip_x, hip_y, shoulder_x, shoulder_y, elbow_x, elbow_y)


def calculate_left_shoulder(keypoints):
    """Levé rameno: kyčel-rameno-loket"""
    if not (is_valid(keypoints[23 * 3 + 2]) and is_valid(keypoints[11 * 3 + 2]) and is_valid(keypoints[13 * 3 + 2])):
        return None
    
    hip_x, hip_y = keypoints[23 * 3], keypoints[23 * 3 + 1]
    shoulder_x, shoulder_y = keypoints[11 * 3], keypoints[11 * 3 + 1]
    elbow_x, elbow_y = keypoints[13 * 3], keypoints[13 * 3 + 1]
    
    return calculate_angle(hip_x, hip_y, shoulder_x, shoulder_y, elbow_x, elbow_y)


def calculate_right_hip(keypoints):
    """Pravá kyčel: rameno-kyčel-koleno"""
    if not (is_valid(keypoints[12 * 3 + 2]) and is_valid(keypoints[24 * 3 + 2]) and is_valid(keypoints[26 * 3 + 2])):
        return None
    
    shoulder_x, shoulder_y = keypoints[12 * 3], keypoints[12 * 3 + 1]
    hip_x, hip_y = keypoints[24 * 3], keypoints[24 * 3 + 1]
    knee_x, knee_y = keypoints[26 * 3], keypoints[26 * 3 + 1]
    
    return calculate_angle(shoulder_x, shoulder_y, hip_x, hip_y, knee_x, knee_y)


def calculate_left_hip(keypoints):
    """Levá kyčel: rameno-kyčel-koleno"""
    if not (is_valid(keypoints[11 * 3 + 2]) and is_valid(keypoints[23 * 3 + 2]) and is_valid(keypoints[25 * 3 + 2])):
        return None
    
    shoulder_x, shoulder_y = keypoints[11 * 3], keypoints[11 * 3 + 1]
    hip_x, hip_y = keypoints[23 * 3], keypoints[23 * 3 + 1]
    knee_x, knee_y = keypoints[25 * 3], keypoints[25 * 3 + 1]
    
    return calculate_angle(shoulder_x, shoulder_y, hip_x, hip_y, knee_x, knee_y)


def calculate_right_knee(keypoints):
    """Pravé koleno: kyčel-koleno-kotník"""
    if not (is_valid(keypoints[24 * 3 + 2]) and is_valid(keypoints[26 * 3 + 2]) and is_valid(keypoints[28 * 3 + 2])):
        return None
    
    hip_x, hip_y = keypoints[24 * 3], keypoints[24 * 3 + 1]
    knee_x, knee_y = keypoints[26 * 3], keypoints[26 * 3 + 1]
    ankle_x, ankle_y = keypoints[28 * 3], keypoints[28 * 3 + 1]
    
    angle = calculate_angle(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y)
    
    # Korekce pro úhly menší než 90° - kontrola pozice kotníku vůči kolenu
    if angle is not None and angle < 90:
        # Pokud je kotník pod kolenem (y-ová souřadnice větší), je to pokrčené koleno
        if ankle_y > knee_y:
            angle = 180 - angle
    
    return angle


def calculate_left_knee(keypoints):
    """Levé koleno: kyčel-koleno-kotník"""
    if not (is_valid(keypoints[23 * 3 + 2]) and is_valid(keypoints[25 * 3 + 2]) and is_valid(keypoints[27 * 3 + 2])):
        return None
    
    hip_x, hip_y = keypoints[23 * 3], keypoints[23 * 3 + 1]
    knee_x, knee_y = keypoints[25 * 3], keypoints[25 * 3 + 1]
    ankle_x, ankle_y = keypoints[27 * 3], keypoints[27 * 3 + 1]
    
    angle = calculate_angle(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y)
    
    # Korekce pro úhly menší než 90° - kontrola pozice kotníku vůči kolenu
    if angle is not None and angle < 90:
        # Pokud je kotník pod kolenem (y-ová souřadnice větší), je to pokrčené koleno
        if ankle_y > knee_y:
            angle = 180 - angle
    
    return angle


def draw_angle_on_frame(frame, keypoints, angle, joint_indices, text_position, joint_name):
    """Vykreslí úhel do snímku včetně oblouku"""
    if angle is None:
        return
    
    h, w, _ = frame.shape
    
    # Zkontroluj viditelnost všech bodů
    if all(is_valid(keypoints[i * 3 + 2]) for i in joint_indices):
        # Získej souřadnice bodů
        points = []
        for i in joint_indices:
            x = int(keypoints[i * 3])
            y = int(keypoints[i * 3 + 1])
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
        text = f"{joint_name}: {angle:.1f}°"
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def create_graphs(angles_data, output_folder, fps):
    """Vytvoří grafy pro každý kloub"""
    graphs_folder = os.path.join(output_folder, "graphs")
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
        plt.title(f'Vývoj úhlu - {joint_name}', fontsize=14, fontweight='bold')
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
    txt_path = os.path.join(output_folder, "min_max_angles.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Analýza úhlů kloubů - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Detektor: {detector_type.upper()}\n")
        f.write("="*60 + "\n\n")
        
        for joint_name, angles in angles_data.items():
            valid_data = [angle for angle, _ in angles if angle is not None]
            if valid_data:
                min_angle = min(valid_data)
                max_angle = max(valid_data)
                avg_angle = sum(valid_data) / len(valid_data)
                
                f.write(f"{joint_name}:\n")
                f.write(f"  Minimální úhel: {min_angle:.2f}°\n")
                f.write(f"  Maximální úhel: {max_angle:.2f}°\n")
                f.write(f"  Průměrný úhel: {avg_angle:.2f}°\n")
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
                    "angle_degrees": angle,
                    "detector": detector_type
                })
    
    json_path = os.path.join(output_folder, "angles_timeline.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def analyze_video(video_path, output_folder="pose_analysis_output", detector_type="mediapipe"):
    """Hlavní funkce pro analýzu videa"""
    
    # Nastavení globální proměnné pro typ detektoru
    global CURRENT_DETECTOR_TYPE
    CURRENT_DETECTOR_TYPE = detector_type
    
    # Vytvoření výstupních složek
    os.makedirs(output_folder, exist_ok=True)
    frames_folder = os.path.join(output_folder, "annotated_frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    # Inicializace pose detectoru
    try:
        pose_detector = PoseDetector(detector_type)
        print(f"✅ {detector_type.upper()} úspěšně inicializován")
    except Exception as e:
        print(f"❌ Chyba při inicializaci {detector_type}: {e}")
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
    print(f"🔍 Detektor: {detector_type.upper()}")
    
    # VideoWriter pro výstupní video
    output_video_path = os.path.join(output_folder, f"analyzed_video_{detector_type}.mp4")
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
            keypoints, detection_result = pose_detector.detect_pose(frame)
            
            if keypoints is not None:
                # Vykreslení pose landmarks
                pose_detector.draw_landmarks(frame, detection_result)
                
                # Výpočet úhlů
                right_elbow = calculate_right_elbow(keypoints)
                left_elbow = calculate_left_elbow(keypoints)
                right_shoulder = calculate_right_shoulder(keypoints)
                left_shoulder = calculate_left_shoulder(keypoints)
                right_hip = calculate_right_hip(keypoints)
                left_hip = calculate_left_hip(keypoints)
                right_knee = calculate_right_knee(keypoints)
                left_knee = calculate_left_knee(keypoints)
                
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
                draw_angle_on_frame(frame, keypoints, right_elbow, [12, 14, 16], (10, 30), "R Loket")
                draw_angle_on_frame(frame, keypoints, left_elbow, [11, 13, 15], (10, 60), "L Loket")
                draw_angle_on_frame(frame, keypoints, right_shoulder, [24, 12, 14], (10, 90), "R Rameno")
                draw_angle_on_frame(frame, keypoints, left_shoulder, [23, 11, 13], (10, 120), "L Rameno")
                draw_angle_on_frame(frame, keypoints, right_hip, [12, 24, 26], (10, 150), "R Kycel")
                draw_angle_on_frame(frame, keypoints, left_hip, [11, 23, 25], (10, 180), "L Kycel")
                draw_angle_on_frame(frame, keypoints, right_knee, [24, 26, 28], (10, 210), "R Koleno")
                draw_angle_on_frame(frame, keypoints, left_knee, [23, 25, 27], (10, 240), "L Koleno")
                
                # Označení detektoru
                cv2.putText(frame, detector_type.upper(), (width - 150, 30), 
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
    print("💾 Ukládám výsledky...")
    save_results(angles_data, output_folder, fps, detector_type)
    
    # Vytvoření grafů
    print("📈 Vytvářím grafy...")
    create_graphs(angles_data, output_folder, fps)
    
    print(f"✅ Analýza dokončena! Výsledky uloženy v: {output_folder}")
    print(f"🎥 Výstupní video: {output_video_path}")
    
    return angles_data


def main():
    """Hlavní funkce s podporou argumentů příkazové řádky"""
    parser = argparse.ArgumentParser(description="Analýza polohy těla pomocí MediaPipe nebo OpenPose")
    parser.add_argument("--video", "-v", type=str, default="video/RLelb_RLshou_RLknee.mp4",
                       help="Cesta k video souboru")
    parser.add_argument("--detector", "-d", type=str, 
                       choices=["mediapipe", "movenet_lightning", "movenet_thunder", "openpose"],
                       help="Typ pose detektoru (mediapipe/movenet_lightning/movenet_thunder/openpose)")
    parser.add_argument("--output", "-o", type=str, default="pose_analysis_output",
                       help="Výstupní složka")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interaktivní výběr detektoru")
    
    args = parser.parse_args()
    
    # Kontrola dostupných detektorů
    available_detectors = get_available_detectors()
    if not available_detectors:
        print("❌ Žádný pose detektor není k dispozici!")
        print("📦 Nainstalujte MediaPipe: pip install mediapipe")
        print("📦 Nebo nainstalujte OpenPose podle oficiální dokumentace")
        return 1
    
    # Výběr detektoru
    if args.interactive or args.detector is None:
        detector_type = select_detector()
        if detector_type is None:
            return 1
    else:
        detector_type = args.detector
        if detector_type not in available_detectors:
            print(f"❌ Detektor '{detector_type}' není k dispozici!")
            print(f"✅ Dostupné detektory: {', '.join(available_detectors)}")
            return 1
    
    # Kontrola existence video souboru
    if not os.path.exists(args.video):
        print(f"❌ Video soubor '{args.video}' neexistuje!")
        return 1
    
    # Spuštění analýzy
    print(f"\n🚀 Spouštím analýzu...")
    angles_data = analyze_video(args.video, args.output, detector_type)
    
    if angles_data:
        # Výpis základních statistik
        print("\n" + "="*60)
        print("📊 SOUHRN VÝSLEDKŮ")
        print("="*60)
        print(f"🔍 Detektor: {detector_type.upper()}")
        
        for joint_name, angles in angles_data.items():
            valid_data = [angle for angle, _ in angles if angle is not None]
            if valid_data:
                min_angle = min(valid_data)
                max_angle = max(valid_data)
                avg_angle = sum(valid_data) / len(valid_data)
                
                print(f"\n{joint_name}:")
                print(f"  Min: {min_angle:.2f}°")
                print(f"  Max: {max_angle:.2f}°")
                print(f"  Průměr: {avg_angle:.2f}°")
                print(f"  Počet měření: {len(valid_data)}")
            else:
                print(f"\n{joint_name}: Žádná platná data")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())