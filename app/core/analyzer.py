"""
Worker thread pro běh analýzy bez blokování UI
"""
import sys
import os
from pathlib import Path

# Přidej parent directory do path - jdi o 2 úrovně výš
parent_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, parent_dir)

from PyQt6.QtCore import QObject, pyqtSignal
import cv2
import numpy as np

# Importuj detektory a analýzu - z Analyses adresáře
from pose_detector import PoseDetector
from pose_analysis_unified import (
    calculate_right_elbow, calculate_left_elbow,
    calculate_right_shoulder, calculate_left_shoulder,
    calculate_right_hip, calculate_left_hip,
    calculate_right_knee, calculate_left_knee
)
import pose_analysis_unified  # Pro přístup k globální proměnné CUSTOM_THRESHOLD

class AnalysisWorker(QObject):
    progress = pyqtSignal(int, dict)  # (progress_percent, frame_data)
    finished = pyqtSignal(dict)  # (results)
    
    def __init__(self, video_path, model_name, selected_joints, confidence_threshold, rotation=0):
        super().__init__()
        self.video_path = video_path
        self.model_name = model_name
        self.selected_joints = selected_joints
        self.confidence_threshold = confidence_threshold
        self.rotation = rotation  # Rotace videa (0, 90, 180, 270)
        
        # Mapování výpočtů úhlů
        self.angle_calculators = {
            'right_elbow': calculate_right_elbow,
            'left_elbow': calculate_left_elbow,
            'right_shoulder': calculate_right_shoulder,
            'left_shoulder': calculate_left_shoulder,
            'right_hip': calculate_right_hip,
            'left_hip': calculate_left_hip,
            'right_knee': calculate_right_knee,
            'left_knee': calculate_left_knee,
        }
    
    def _rotate_frame(self, frame, rotation):
        """
        Rotuje frame o zadaný úhel
        rotation: 0, 90, 180, 270 stupňů
        """
        if rotation == 0:
            return frame
        elif rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return frame
    
    def run(self):
        """Spustí analýzu"""
        try:
            # Nastav global confidence threshold pro is_valid() v pose_analysis_unified
            pose_analysis_unified.CUSTOM_CONFIDENCE_THRESHOLD = self.confidence_threshold
            
            # Inicializuj detektor
            detector = PoseDetector(self.model_name)
            
            # Otevři video
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Slovník pro ukládání úhlů
            angles_data = {joint: [] for joint in self.selected_joints}
            keypoints_data = []
            
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Aplikuj rotaci pokud je nastavena
                if self.rotation != 0:
                    frame = self._rotate_frame(frame, self.rotation)
                
                # Detekuj poses
                keypoints, result = detector.detect_pose(frame)
                
                if keypoints is not None:
                    # Konvertuj keypoints na numpy array pokud je list
                    if isinstance(keypoints, list):
                        keypoints = np.array(keypoints)
                    
                    frame_angles = {}
                    
                    # Vypočítej úhly pro vybrané klouby
                    for joint in self.selected_joints:
                        if joint in self.angle_calculators:
                            try:
                                angle = self.angle_calculators[joint](keypoints)
                                if angle is not None:
                                    angles_data[joint].append(angle)
                                    frame_angles[joint] = angle
                            except Exception as e:
                                print(f"Chyba při výpočtu {joint}: {e}")
                    
                    # Ukládej VŠECHNY frames s keypoints a úhly do exportu
                    keypoints_data.append({
                        'frame': frame_index,
                        'keypoints': keypoints.tolist() if isinstance(keypoints, np.ndarray) else keypoints,
                        'angles': frame_angles
                    })
                
                frame_index += 1
                progress = int((frame_index / total_frames) * 100)
                self.progress.emit(progress, {'frame': frame_index, 'total': total_frames})
            
            cap.release()
            
            # Připrav výsledky - exportuj VŠE včetně všech 600 frames
            results = {
                'video_path': self.video_path,
                'model': self.model_name,
                'total_frames': frame_index,
                'selected_joints': self.selected_joints,
                'angles': angles_data,
                'keypoints': keypoints_data,
                'statistics': self._calculate_statistics(angles_data, keypoints_data)
            }
            
            self.finished.emit(results)
            
        except Exception as e:
            print(f"Chyba při analýze: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit({'error': str(e)})
    
    def _calculate_statistics(self, angles_data, keypoints_data):
        """Vypočítá statistiky pro úhly"""
        stats = {}

        # Build per-joint list of (frame_index, angle) from keypoints_data to preserve absolute frame numbers
        joint_frame_angles = {joint: [] for joint in angles_data.keys()}

        for entry in keypoints_data:
            frame_idx = entry.get('frame')
            frame_angles = entry.get('angles', {})
            for joint, angle in frame_angles.items():
                # only include if angle is not None
                if angle is not None and joint in joint_frame_angles:
                    joint_frame_angles[joint].append((frame_idx, angle))

        for joint, frame_angle_list in joint_frame_angles.items():
            if frame_angle_list:
                # Extract angles only for statistical calculations
                angles = [fa[1] for fa in frame_angle_list]
                min_angle = min(angles)
                max_angle = max(angles)

                # Find absolute frame indices corresponding to min and max
                # If multiple occurrences, take the first occurrence
                min_frame = next((fa[0] for fa in frame_angle_list if fa[1] == min_angle), None)
                max_frame = next((fa[0] for fa in frame_angle_list if fa[1] == max_angle), None)

                stats[joint] = {
                    'average': sum(angles) / len(angles) if len(angles) > 0 else 0,
                    'min': min_angle,
                    'min_frame': min_frame,
                    'max': max_angle,
                    'max_frame': max_frame,
                    'std_dev': float(np.std(angles)) if len(angles) > 1 else 0,
                    'count': len(angles)
                }

        return stats
