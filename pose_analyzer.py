import cv2
import mediapipe as mp
import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Dict


class PoseAnalyzer:
    def __init__(self, video_path: str, output_dir: str = "output"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        
        # Vytvoření výstupních složek
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        
        # Úložiště pro úhly
        self.angles_data = {
            'right_shoulder': [],
            'left_shoulder': [],
            'right_elbow': [],
            'left_elbow': [],
            'right_hip': [],
            'left_hip': [],
            'right_knee': [],
            'left_knee': []
        }
        
        # MediaPipe klíčové body indexy
        self.pose_landmarks = {
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
        }
    
    def is_valid_visibility(self, visibility: float, threshold: float = 0.7) -> bool:
        """Kontroluje zda je viditelnost klíčového bodu dostatečná"""
        return visibility >= threshold
    
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """
        Vypočítá úhel mezi třemi body.
        point2 je vrchol úhlu.
        """
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        
        # Vektory z point2 k point1 a point3
        vector1 = (x1 - x2, y1 - y2)
        vector2 = (x3 - x2, y3 - y2)
        
        # Skalární součin
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        
        # Velikosti vektorů
        magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return None
        
        # Kosinus úhlu
        cos_angle = dot_product / (magnitude1 * magnitude2)
        
        # Ošetření numerických chyb
        cos_angle = max(-1, min(1, cos_angle))
        
        # Úhel v radiánech a převod na stupně
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def get_landmark_coords(self, landmarks, landmark_idx: int, 
                           width: int, height: int) -> Tuple[Optional[Tuple[float, float]], float]:
        """Získá souřadnice a viditelnost klíčového bodu"""
        if landmark_idx >= len(landmarks.landmark):
            return None, 0.0
            
        landmark = landmarks.landmark[landmark_idx]
        if not self.is_valid_visibility(landmark.visibility):
            return None, landmark.visibility
            
        x = landmark.x * width
        y = landmark.y * height
        return (x, y), landmark.visibility
    
    def calculate_joint_angles(self, landmarks, width: int, height: int, frame_id: int) -> Dict[str, float]:
        """Vypočítá všechny úhly kloubů pro daný snímek"""
        angles = {}
        
        # Pravé rameno (kyčel-rameno-loket)
        right_hip, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['right_hip'], width, height)
        right_shoulder, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['right_shoulder'], width, height)
        right_elbow, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['right_elbow'], width, height)
        
        if all(coord is not None for coord in [right_hip, right_shoulder, right_elbow]):
            angle = self.calculate_angle(right_hip, right_shoulder, right_elbow)
            if angle is not None:
                angles['right_shoulder'] = angle
                self.angles_data['right_shoulder'].append((angle, frame_id))
        
        # Levé rameno (kyčel-rameno-loket)
        left_hip, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['left_hip'], width, height)
        left_shoulder, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['left_shoulder'], width, height)
        left_elbow, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['left_elbow'], width, height)
        
        if all(coord is not None for coord in [left_hip, left_shoulder, left_elbow]):
            angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
            if angle is not None:
                angles['left_shoulder'] = angle
                self.angles_data['left_shoulder'].append((angle, frame_id))
        
        # Pravý loket (rameno-loket-zápěstí)
        right_wrist, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['right_wrist'], width, height)
        
        if all(coord is not None for coord in [right_shoulder, right_elbow, right_wrist]):
            angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            if angle is not None:
                angles['right_elbow'] = angle
                self.angles_data['right_elbow'].append((angle, frame_id))
        
        # Levý loket (rameno-loket-zápěstí)
        left_wrist, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['left_wrist'], width, height)
        
        if all(coord is not None for coord in [left_shoulder, left_elbow, left_wrist]):
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            if angle is not None:
                angles['left_elbow'] = angle
                self.angles_data['left_elbow'].append((angle, frame_id))
        
        # Pravá kyčel (rameno-kyčel-koleno)
        right_knee, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['right_knee'], width, height)
        
        if all(coord is not None for coord in [right_shoulder, right_hip, right_knee]):
            angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
            if angle is not None:
                angles['right_hip'] = angle
                self.angles_data['right_hip'].append((angle, frame_id))
        
        # Levá kyčel (rameno-kyčel-koleno)
        left_knee, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['left_knee'], width, height)
        
        if all(coord is not None for coord in [left_shoulder, left_hip, left_knee]):
            angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            if angle is not None:
                angles['left_hip'] = angle
                self.angles_data['left_hip'].append((angle, frame_id))
        
        # Pravé koleno (kyčel-koleno-kotník)
        right_ankle, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['right_ankle'], width, height)
        
        if all(coord is not None for coord in [right_hip, right_knee, right_ankle]):
            angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            if angle is not None:
                angles['right_knee'] = angle
                self.angles_data['right_knee'].append((angle, frame_id))
        
        # Levé koleno (kyčel-koleno-kotník)
        left_ankle, _ = self.get_landmark_coords(landmarks, self.pose_landmarks['left_ankle'], width, height)
        
        if all(coord is not None for coord in [left_hip, left_knee, left_ankle]):
            angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            if angle is not None:
                angles['left_knee'] = angle
                self.angles_data['left_knee'].append((angle, frame_id))
        
        return angles
    
    def draw_angle_text(self, frame: np.ndarray, angles: Dict[str, float]) -> np.ndarray:
        """Nakreslí úhly do snímku"""
        y_offset = 30
        for joint_name, angle in angles.items():
            text = f"{joint_name}: {angle:.1f}°"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            y_offset += 25
        return frame
    
    def process_video(self) -> str:
        """Zpracuje video a vytvoří výstupní video s úhly"""
        cap = cv2.VideoCapture(self.video_path)
        
        # Získání vlastností videa
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Nastavení kodeku pro výstupní video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = os.path.join(self.output_dir, 'output_with_angles.mp4')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_id = 0
        
        print("Zpracovávám video...")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Převod do RGB pro MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Kreslení kostry
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Výpočet úhlů
                angles = self.calculate_joint_angles(results.pose_landmarks, width, height, frame_id)
                
                # Kreslení úhlů do snímku
                frame = self.draw_angle_text(frame, angles)
            
            # Uložení snímku
            frame_filename = os.path.join(self.output_dir, "frames", f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            # Zápis do výstupního videa
            out.write(frame)
            
            frame_id += 1
            
            if frame_id % 30 == 0:
                print(f"Zpracováno {frame_id} snímků...")
        
        cap.release()
        out.release()
        
        print(f"Video zpracováno. Celkem snímků: {frame_id}")
        return output_video_path
    
    def save_results(self):
        """Uloží výsledky do souborů"""
        # Příprava dat pro JSON
        json_data = {}
        for joint_name, angles_list in self.angles_data.items():
            if angles_list:
                json_data[joint_name] = [
                    {"frame": frame_id, "angle": angle, "time": frame_id / 30.0}  # Předpokládáme 30 FPS
                    for angle, frame_id in angles_list
                ]
        
        # Uložení JSON
        json_path = os.path.join(self.output_dir, 'angles_timeline.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Uložení TXT se statistikami
        txt_path = os.path.join(self.output_dir, 'angles_statistics.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("STATISTIKY ÚHLŮ KLOUBŮ\n")
            f.write("=" * 50 + "\n\n")
            
            for joint_name, angles_list in self.angles_data.items():
                if angles_list:
                    angles_only = [angle for angle, _ in angles_list]
                    max_angle = max(angles_only)
                    min_angle = min(angles_only)
                    avg_angle = sum(angles_only) / len(angles_only)
                    
                    max_frame = next(frame for angle, frame in angles_list if angle == max_angle)
                    min_frame = next(frame for angle, frame in angles_list if angle == min_angle)
                    
                    f.write(f"{joint_name.upper().replace('_', ' ')}:\n")
                    f.write(f"  Maximální úhel: {max_angle:.2f}° (snímek {max_frame})\n")
                    f.write(f"  Minimální úhel: {min_angle:.2f}° (snímek {min_frame})\n")
                    f.write(f"  Průměrný úhel: {avg_angle:.2f}°\n")
                    f.write(f"  Počet měření: {len(angles_only)}\n\n")
                else:
                    f.write(f"{joint_name.upper().replace('_', ' ')}:\n")
                    f.write("  Žádná validní data\n\n")
    
    def create_graphs(self):
        """Vytvoří grafy vývoje úhlů"""
        # Nastavení stylu grafu
        plt.style.use('default')
        
        # Rozdělení na horní a dolní končetiny
        upper_joints = ['right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow']
        lower_joints = ['right_hip', 'left_hip', 'right_knee', 'left_knee']
        
        # Graf horních končetin
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        for joint_name in upper_joints:
            if self.angles_data[joint_name]:
                frames = [frame for _, frame in self.angles_data[joint_name]]
                angles = [angle for angle, _ in self.angles_data[joint_name]]
                times = [frame / 30.0 for frame in frames]  # Převod na čas (předpokládáme 30 FPS)
                
                plt.plot(times, angles, label=joint_name.replace('_', ' ').title(), linewidth=2)
        
        plt.title('Vývoj úhlů - Horní končetiny', fontsize=14, fontweight='bold')
        plt.xlabel('Čas (s)')
        plt.ylabel('Úhel (°)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graf dolních končetin
        plt.subplot(2, 1, 2)
        for joint_name in lower_joints:
            if self.angles_data[joint_name]:
                frames = [frame for _, frame in self.angles_data[joint_name]]
                angles = [angle for angle, _ in self.angles_data[joint_name]]
                times = [frame / 30.0 for frame in frames]
                
                plt.plot(times, angles, label=joint_name.replace('_', ' ').title(), linewidth=2)
        
        plt.title('Vývoj úhlů - Dolní končetiny', fontsize=14, fontweight='bold')
        plt.xlabel('Čas (s)')
        plt.ylabel('Úhel (°)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'angles_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individuální grafy pro každý kloub
        for joint_name, angles_list in self.angles_data.items():
            if angles_list:
                plt.figure(figsize=(12, 6))
                
                frames = [frame for _, frame in angles_list]
                angles = [angle for angle, _ in angles_list]
                times = [frame / 30.0 for frame in frames]
                
                plt.plot(times, angles, 'b-', linewidth=2, label='Úhel')
                plt.fill_between(times, angles, alpha=0.3)
                
                # Označení min/max hodnot
                max_angle = max(angles)
                min_angle = min(angles)
                max_time = times[angles.index(max_angle)]
                min_time = times[angles.index(min_angle)]
                
                plt.plot(max_time, max_angle, 'ro', markersize=8, label=f'Max: {max_angle:.1f}°')
                plt.plot(min_time, min_angle, 'go', markersize=8, label=f'Min: {min_angle:.1f}°')
                
                plt.title(f'Vývoj úhlu - {joint_name.replace("_", " ").title()}', 
                         fontsize=14, fontweight='bold')
                plt.xlabel('Čas (s)')
                plt.ylabel('Úhel (°)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{joint_name}_timeline.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def run_analysis(self):
        """Spustí kompletní analýzu"""
        print("Začínám analýzu pohybu...")
        
        # Zpracování videa
        output_video = self.process_video()
        print(f"Výstupní video uloženo: {output_video}")
        
        # Uložení výsledků
        self.save_results()
        print("Statistiky uloženy do souborů")
        
        # Vytvoření grafů
        self.create_graphs()
        print("Grafy vytvořeny")
        
        # Výpis souhrnu
        print("\n" + "="*50)
        print("SOUHRN ANALÝZY")
        print("="*50)
        
        for joint_name, angles_list in self.angles_data.items():
            if angles_list:
                angles_only = [angle for angle, _ in angles_list]
                print(f"{joint_name.replace('_', ' ').title()}: "
                      f"Min={min(angles_only):.1f}°, "
                      f"Max={max(angles_only):.1f}°, "
                      f"Avg={sum(angles_only)/len(angles_only):.1f}°")
        
        print(f"\nVšechny výsledky uloženy ve složce: {self.output_dir}")


def main():
    # Nastavení cest
    video_path = "video/RLelb_RLshou_RLknee.mp4"  # Změňte podle potřeby
    output_dir = "pose_analysis_output"
    
    # Kontrola existence videa
    if not os.path.exists(video_path):
        print(f"Chyba: Video soubor '{video_path}' nebyl nalezen!")
        print("Prosím, upravte cestu k videu v proměnné 'video_path'.")
        return
    
    # Vytvoření analyzátoru a spuštění analýzy
    analyzer = PoseAnalyzer(video_path, output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()