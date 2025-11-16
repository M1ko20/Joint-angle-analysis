#!/usr/bin/env python3
"""
Společné funkce pro výpočet úhlů kloubů
Obsahuje všechny korekce a normalizace z původního kódu
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# === GLOBÁLNÍ PROMĚNNÉ ===
CURRENT_DETECTOR_TYPE = "mediapipe"
CUSTOM_CONFIDENCE_THRESHOLD = None


def set_detector_type(detector_type):
    """Nastaví globální typ detektoru"""
    global CURRENT_DETECTOR_TYPE
    CURRENT_DETECTOR_TYPE = detector_type


def set_confidence_threshold(threshold):
    """Nastaví globální confidence threshold"""
    global CUSTOM_CONFIDENCE_THRESHOLD
    CUSTOM_CONFIDENCE_THRESHOLD = threshold


def is_valid(visibility, threshold=None, detector_type=None):
    """
    Kontroluje, zda je bod dostatečně viditelný
    
    Args:
        visibility: confidence/visibility hodnota keypoints
        threshold: Custom threshold (pokud je None, použije se globální)
        detector_type: Typ detektoru
    
    Returns:
        True pokud je visibility >= threshold
    """
    if detector_type is None:
        detector_type = CURRENT_DETECTOR_TYPE
    
    if threshold is None and CUSTOM_CONFIDENCE_THRESHOLD is not None:
        threshold = CUSTOM_CONFIDENCE_THRESHOLD
    
    if threshold is None:
        if detector_type.startswith("movenet"):
            threshold = 0.3
        elif detector_type == "openpose":
            threshold = 0.4
        else:
            threshold = 0.5
    
    return visibility >= threshold


def normalize_angle(angle):
    """
    Normalizuje úhel pro fyzické interpretaci.
    Opravuje problém kde koleno skoro rovně vrací 0.1° místo 180°
    """
    if angle is None:
        return None
    
    if angle < 2.0:
        return 180.0
    
    if 178.0 <= angle <= 182.0:
        return 180.0
    
    return angle


def calculate_angle(aX, aY, bX, bY, cX, cY):
    """
    Vypočítá úhel mezi třemi body (A-B-C, kde B je vrchol úhlu)
    """
    vec_ba = (aX - bX, aY - bY)
    vec_bc = (cX - bX, cY - bY)
    
    angle1 = np.arctan2(vec_ba[1], vec_ba[0])
    angle2 = np.arctan2(vec_bc[1], vec_bc[0])
    
    angle_rad = angle2 - angle1
    angle_deg = np.degrees(angle_rad)
    angle_deg = abs(angle_deg)
    
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    
    angle_deg = normalize_angle(angle_deg)
    return angle_deg


def correct_angle_for_limb_direction(angle, joint_x, center_x, is_right_limb):
    """
    Opravuje úhel pokud je > 180° na základě pozice kloubu vůči tělu
    
    Pro pravý loket: pokud je loket VLEVO od ramene -> úhel je z druhé strany
    Pro levý loket: pokud je loket VPRAVO od ramene -> úhel je z druhé strany
    """
    if angle is None:
        return None
    
    if is_right_limb:
        is_wrong_side = joint_x < center_x
    else:
        is_wrong_side = joint_x > center_x
    
    if is_wrong_side and angle > 90:
        corrected_angle = 360 - angle
        return corrected_angle
    
    return angle


# === VÝPOČET ÚHLŮ VŠECH 8 KLOUBŮ ===

def calculate_right_elbow(keypoints):
    """Pravý loket: rameno-loket-zápěstí"""
    if not (is_valid(keypoints[12*3+2]) and is_valid(keypoints[14*3+2]) and is_valid(keypoints[16*3+2])):
        return None
    
    return calculate_angle(
        keypoints[12*3], keypoints[12*3+1],
        keypoints[14*3], keypoints[14*3+1],
        keypoints[16*3], keypoints[16*3+1]
    )


def calculate_left_elbow(keypoints):
    """Levý loket: rameno-loket-zápěstí"""
    if not (is_valid(keypoints[11*3+2]) and is_valid(keypoints[13*3+2]) and is_valid(keypoints[15*3+2])):
        return None
    
    return calculate_angle(
        keypoints[11*3], keypoints[11*3+1],
        keypoints[13*3], keypoints[13*3+1],
        keypoints[15*3], keypoints[15*3+1]
    )


def calculate_right_shoulder(keypoints):
    """Pravé rameno: kyčel-rameno-loket"""
    if not (is_valid(keypoints[24*3+2]) and is_valid(keypoints[12*3+2]) and is_valid(keypoints[14*3+2])):
        return None
    
    angle = calculate_angle(
        keypoints[24*3], keypoints[24*3+1],
        keypoints[12*3], keypoints[12*3+1],
        keypoints[14*3], keypoints[14*3+1]
    )
    return correct_angle_for_limb_direction(angle, keypoints[14*3], keypoints[12*3], is_right_limb=False)


def calculate_left_shoulder(keypoints):
    """Levé rameno: kyčel-rameno-loket"""
    if not (is_valid(keypoints[23*3+2]) and is_valid(keypoints[11*3+2]) and is_valid(keypoints[13*3+2])):
        return None
    
    angle = calculate_angle(
        keypoints[23*3], keypoints[23*3+1],
        keypoints[11*3], keypoints[11*3+1],
        keypoints[13*3], keypoints[13*3+1]
    )
    return correct_angle_for_limb_direction(angle, keypoints[13*3], keypoints[11*3], is_right_limb=True)


def calculate_right_hip(keypoints):
    """Pravá kyčel: rameno-kyčel-koleno"""
    if not (is_valid(keypoints[12*3+2]) and is_valid(keypoints[24*3+2]) and is_valid(keypoints[26*3+2])):
        return None
    
    return calculate_angle(
        keypoints[12*3], keypoints[12*3+1],
        keypoints[24*3], keypoints[24*3+1],
        keypoints[26*3], keypoints[26*3+1]
    )


def calculate_left_hip(keypoints):
    """Levá kyčel: rameno-kyčel-koleno"""
    if not (is_valid(keypoints[11*3+2]) and is_valid(keypoints[23*3+2]) and is_valid(keypoints[25*3+2])):
        return None
    
    return calculate_angle(
        keypoints[11*3], keypoints[11*3+1],
        keypoints[23*3], keypoints[23*3+1],
        keypoints[25*3], keypoints[25*3+1]
    )


def calculate_right_knee(keypoints):
    """Pravé koleno: kyčel-koleno-kotník (s korekcí 0°→180°)"""
    if not (is_valid(keypoints[24*3+2]) and is_valid(keypoints[26*3+2]) and is_valid(keypoints[28*3+2])):
        return None
    
    angle = calculate_angle(
        keypoints[24*3], keypoints[24*3+1],
        keypoints[26*3], keypoints[26*3+1],
        keypoints[28*3], keypoints[28*3+1]
    )
    
    # Pokud je úhel blízko 180°, necháme ho
    if angle is not None and angle >= 160:
        return angle
    
    # Korekce pro úhly menší než 90°
    if angle is not None and angle < 90:
        # Pokud je kotník pod kolenem (y větší), je to pokrčené koleno
        if keypoints[28*3+1] > keypoints[26*3+1]:
            angle = 180 - angle
    
    return angle


def calculate_left_knee(keypoints):
    """Levé koleno: kyčel-koleno-kotník (s korekcí 0°→180°)"""
    if not (is_valid(keypoints[23*3+2]) and is_valid(keypoints[25*3+2]) and is_valid(keypoints[27*3+2])):
        return None
    
    angle = calculate_angle(
        keypoints[23*3], keypoints[23*3+1],
        keypoints[25*3], keypoints[25*3+1],
        keypoints[27*3], keypoints[27*3+1]
    )
    
    # Pokud je úhel blízko 180°, necháme ho
    if angle is not None and angle >= 160:
        return angle
    
    # Korekce pro úhly menší než 90°
    if angle is not None and angle < 90:
        # Pokud je kotník pod kolenem (y větší), je to pokrčené koleno
        if keypoints[27*3+1] > keypoints[25*3+1]:
            angle = 180 - angle
    
    return angle


def calculate_all_angles(keypoints):
    """
    Vypočítá všechny úhly najednou
    
    Returns:
        dict: {joint_name: angle}
    """
    return {
        "right_elbow": calculate_right_elbow(keypoints),
        "left_elbow": calculate_left_elbow(keypoints),
        "right_shoulder": calculate_right_shoulder(keypoints),
        "left_shoulder": calculate_left_shoulder(keypoints),
        "right_hip": calculate_right_hip(keypoints),
        "left_hip": calculate_left_hip(keypoints),
        "right_knee": calculate_right_knee(keypoints),
        "left_knee": calculate_left_knee(keypoints)
    }


# === UKLÁDÁNÍ VÝSLEDKŮ ===

def save_keypoints(keypoints_data, output_dir, detector_name, fps):
    """Uloží raw keypoints do JSON"""
    filepath = os.path.join(output_dir, "raw_keypoints.json")
    
    # Convert all NumPy types to Python types
    def convert_to_python_types(obj):
        """Recursively convert NumPy types to Python types"""
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    keypoints_data_clean = convert_to_python_types(keypoints_data)
    
    with open(filepath, 'w') as f:
        json.dump({
            "detector": detector_name,
            "fps": float(fps),
            "total_frames": len(keypoints_data),
            "frames": keypoints_data_clean
        }, f, indent=2)
    
    print(f"   💾 Keypoints: {filepath}")


def save_angles(angles_data, output_dir, detector_name, fps):
    """Uloží angles timeline do JSON"""
    filepath = os.path.join(output_dir, "angles_timeline.json")
    
    timeline = []
    for joint, data in angles_data.items():
        for angle, frame_id in data:
            if angle is not None:
                timeline.append({
                    "joint": joint,
                    "frame": int(frame_id),  # Convert to Python int
                    "timestamp": float(frame_id / fps),  # Convert to Python float
                    "angle": float(angle),  # Convert to Python float
                    "detector": detector_name
                })
    
    with open(filepath, 'w') as f:
        json.dump(timeline, f, indent=2)
    
    print(f"   💾 Angles: {filepath}")


def save_rom_summary(angles_data, output_dir, detector_name):
    """Uloží ROM summary do TXT"""
    filepath = os.path.join(output_dir, "rom_summary.txt")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"RANGE OF MOTION (ROM) SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Detector: {detector_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"{'Joint':<20} {'Min°':<10} {'Max°':<10} {'ROM°':<10} {'Avg°':<10} {'N':<10}\n")
        f.write(f"{'-'*80}\n")
        
        for joint, data in sorted(angles_data.items()):
            angles = [float(a) for a, _ in data if a is not None]  # Convert to Python float
            if angles:
                min_a, max_a = min(angles), max(angles)
                rom = max_a - min_a
                avg_a = sum(angles) / len(angles)
                f.write(f"{joint:<20} {min_a:>9.2f} {max_a:>9.2f} {rom:>9.2f} {avg_a:>9.2f} {len(angles):>9}\n")
            else:
                f.write(f"{joint:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {0:>9}\n")
    
    print(f"   💾 ROM: {filepath}")


def create_angle_graphs(angles_data, output_dir, fps):
    """Vytvoří grafy pro každý kloub"""
    graphs_dir = os.path.join(output_dir, "angle_graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    for joint, data in angles_data.items():
        if not data:
            continue
        
        # Convert to Python floats - filter out None values
        valid_data = [(float(a), int(f)) for a, f in data if a is not None]
        
        if not valid_data:
            continue
        
        angles, frames = zip(*valid_data)
        times = [float(f / fps) for f in frames]
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, angles, 'b-', linewidth=2)
        plt.title(f'{joint.replace("_", " ").title()} Angle Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Angle (degrees)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(graphs_dir, f"{joint}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"   📈 Grafy: {graphs_dir}/ ({len([d for d in angles_data.values() if d])} grafů)")


def save_all_results(keypoints_data, angles_data, output_dir, detector_name, fps):
    """Uloží všechny výsledky najednou"""
    save_keypoints(keypoints_data, output_dir, detector_name, fps)
    save_angles(angles_data, output_dir, detector_name, fps)
    save_rom_summary(angles_data, output_dir, detector_name)
    create_angle_graphs(angles_data, output_dir, fps)
