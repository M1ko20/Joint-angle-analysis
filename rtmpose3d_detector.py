#!/usr/bin/env python3
"""
RTMPose3D detector wrapper pro 3D pose estimation
Tento soubor musí běžet v conda prostředí 'openmmlab'
Kombinuje RT-DETR person detection + RTMPose 3D pose estimation
"""

import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path


class RTMPose3DDetector:
    """Wrapper pro RTMPose3D model (2-stage: detection + pose)"""
    
    def __init__(self, confidence_threshold=0.5):
        """
        Args:
            confidence_threshold: Minimální confidence pro keypoints (0-1)
        """
        # Lazy import MMPose a MMDet
        try:
            from mmpose.apis import inference_topdown, init_model
            from mmpose.utils import register_all_modules
            from mmdet.apis import inference_detector, init_detector
            register_all_modules()
            self._inference_topdown = inference_topdown
            self._init_model = init_model
            self._inference_detector = inference_detector
            self._init_detector = init_detector
        except ImportError:
            raise ImportError("MMPose není dostupné. Aktivuj conda: conda activate openmmlab")
        
        self.confidence_threshold = confidence_threshold
        self.detector = None  # Person detector
        self.pose_model = None  # Pose estimator
        
        # Základní adresář (Analysis/)
        self.base_dir = Path(__file__).parent.absolute()
        
        # Cesty k MMPose
        self.mmpose_dir = self.base_dir / "mmpose"
        
        if not self.mmpose_dir.exists():
            raise FileNotFoundError(f"MMPose adresář nenalezen: {self.mmpose_dir}")
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializuje detection + pose modely"""
        print(f"🔧 Inicializuji RTMPose3D...")
        
        # 1. Person detector (RT-DETR)
        det_config = self.mmpose_dir / "projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py"
        det_checkpoint_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
        
        if not det_config.exists():
            raise FileNotFoundError(f"Detection config nenalezen: {det_config}")
        
        print(f"   Loading person detector...")
        print(f"   Config: {det_config.name}")
        print(f"   Checkpoint: {det_checkpoint_url}")
        
        self.detector = self._init_detector(
            str(det_config),
            det_checkpoint_url,
            device='cpu'
        )
        print(f"   ✅ Person detector načten")
        
        # 2. Pose estimator (RTMPose3D)
        pose_config = self.mmpose_dir / "projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py"
        pose_checkpoint = self.mmpose_dir / "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
        
        if not pose_config.exists():
            raise FileNotFoundError(f"Pose config nenalezen: {pose_config}")
        if not pose_checkpoint.exists():
            raise FileNotFoundError(f"Pose checkpoint nenalezen: {pose_checkpoint}")
        
        print(f"   Loading pose estimator...")
        print(f"   Config: {pose_config.name}")
        print(f"   Checkpoint: {pose_checkpoint.name}")
        print(f"   Confidence: {self.confidence_threshold}")
        
        self.pose_model = self._init_model(
            str(pose_config),
            str(pose_checkpoint),
            device='cpu'
        )
        print(f"   ✅ RTMPose3D úspěšně načten (CPU)")
    
    def detect_pose(self, frame):
        """
        Detekuje 3D pose v rámci
        
        Args:
            frame: OpenCV obrázek (BGR)
        
        Returns:
            tuple: (keypoints, raw_result)
                - keypoints: MediaPipe formát [x,y,z,confidence, x,y,z,confidence, ...]
                - raw_result: Originální MMPose výsledek
        """
        if self.detector is None or self.pose_model is None:
            return None, None
        
        # Stage 1: Person detection
        det_result = self._inference_detector(self.detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()
        
        # Filter person bboxes (category 0 = person in COCO)
        bboxes = pred_instance.bboxes
        scores = pred_instance.scores
        labels = pred_instance.labels
        
        # Filter by class (person) and confidence
        person_mask = np.logical_and(
            labels == 0,  # person class
            scores > self.confidence_threshold
        )
        bboxes = bboxes[person_mask]
        
        if len(bboxes) == 0:
            return None, None
        
        # Stage 2: Pose estimation on detected persons
        pose_results = self._inference_topdown(self.pose_model, frame, bboxes)
        
        if not pose_results or len(pose_results) == 0:
            return None, None
        
        # Vezmi první (nejlepší) detekci
        result = pose_results[0]
        
        try:
            # Extrakce 3D dat
            data = result.pred_instances.to_dict()
            keypoints_3d = data.get('keypoints', None)  # Shape: (17, 3) - x, y, z
            scores = data.get('keypoint_scores', None)  # Shape: (17,)
            
            if keypoints_3d is None or scores is None:
                return None, result
            
            # Převod na MediaPipe formát (33 bodů s x,y,z,confidence)
            mediapipe_keypoints = self._convert_to_mediapipe_format(
                keypoints_3d[0],  # První osoba
                scores[0]
            )
            
            return mediapipe_keypoints, result
            
        except Exception as e:
            print(f"⚠️ Chyba při zpracování RTMPose3D výsledku: {e}")
            return None, result
    
    def _convert_to_mediapipe_format(self, rtmpose_keypoints, scores):
        """
        Převádí RTMPose3D COCO keypoints (17 bodů) na MediaPipe formát (33 bodů)
        
        Args:
            rtmpose_keypoints: numpy array shape (17, 3) - [x, y, z]
            scores: numpy array shape (17,) - confidence scores
        
        Returns:
            list: MediaPipe formát [x,y,z,conf, x,y,z,conf, ...] (33*4 = 132 hodnot)
        """
        # MediaPipe má 33 bodů, RTMPose COCO má 17
        # Pro 3D modely: 4 hodnoty na bod (x, y, z, confidence)
        mediapipe_keypoints = [0.0] * (33 * 4)
        
        # COCO 17 -> MediaPipe 33 mapping (stejné jako u 2D)
        mapping = {
            0: 0,   # nose
            1: 2,   # left_eye
            2: 5,   # right_eye
            3: 7,   # left_ear
            4: 8,   # right_ear
            5: 11,  # left_shoulder
            6: 12,  # right_shoulder
            7: 13,  # left_elbow
            8: 14,  # right_elbow
            9: 15,  # left_wrist
            10: 16, # right_wrist
            11: 23, # left_hip
            12: 24, # right_hip
            13: 25, # left_knee
            14: 26, # right_knee
            15: 27, # left_ankle
            16: 28, # right_ankle
        }
        
        for coco_idx, mediapipe_idx in mapping.items():
            if coco_idx < len(rtmpose_keypoints):
                x, y, z = rtmpose_keypoints[coco_idx]
                confidence = float(scores[coco_idx])
                
                # Pouze body s dostatečným confidence
                if confidence > self.confidence_threshold:
                    base_idx = mediapipe_idx * 4
                    mediapipe_keypoints[base_idx] = float(x)
                    mediapipe_keypoints[base_idx + 1] = float(y)
                    mediapipe_keypoints[base_idx + 2] = float(z)
                    mediapipe_keypoints[base_idx + 3] = confidence
        
        return mediapipe_keypoints
    
    def draw_landmarks(self, frame, detection_result):
        """Vykreslí pose landmarks do snímku (2D projekce)"""
        if detection_result is None:
            return
        
        try:
            data = detection_result.pred_instances.to_dict()
            keypoints = data.get('keypoints', None)[0]  # První osoba, (17, 3)
            scores = data.get('keypoint_scores', None)[0]
            
            if keypoints is None or scores is None:
                return
            
            # COCO skeleton connections
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
            
            # Vykreslení bodů (použij x,y souřadnice, ignoruj z)
            for i, (x, y, z) in enumerate(keypoints):
                if scores[i] > self.confidence_threshold:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                    
                    # Volitelně: zobraz z-hodnotu jako text
                    # cv2.putText(frame, f"{z:.1f}", (int(x)+5, int(y)-5),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
            # Vykreslení spojnic
            for start_idx, end_idx in connections:
                if (scores[start_idx] > self.confidence_threshold and
                    scores[end_idx] > self.confidence_threshold):
                    
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                    
        except Exception as e:
            print(f"⚠️ Chyba při vykreslování: {e}")
    
    def close(self):
        """Uvolní modely"""
        self.detector = None
        self.pose_model = None


def test_rtmpose3d_detector(test_image_path):
    """Testovací funkce"""
    print(f"\n{'='*60}")
    print(f"Testování RTMPose3D")
    print(f"{'='*60}\n")
    
    # Inicializace
    detector = RTMPose3DDetector(confidence_threshold=0.5)
    
    # Načtení testovacího obrázku
    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"❌ Nelze načíst obrázek: {test_image_path}")
        return None
    
    print(f"📷 Načten obrázek: {frame.shape}")
    
    # Detekce
    keypoints, result = detector.detect_pose(frame)
    
    if keypoints is not None:
        print(f"✅ Detekce úspěšná!")
        
        # Počet detekovaných bodů
        valid_points = sum(1 for i in range(0, len(keypoints), 4) if keypoints[i+3] > 0.5)
        print(f"🔍 Detekováno {valid_points} keypoints (z 33)")
        
        # Vykreslení
        detector.draw_landmarks(frame, result)
        
        # Uložení
        output_path = "output_rtmpose3d_test.jpg"
        cv2.imwrite(output_path, frame)
        print(f"💾 Výsledek uložen: {output_path}")
        
        detector.close()
        return keypoints
    else:
        print(f"❌ Detekce selhala")
        detector.close()
        return None


if __name__ == "__main__":
    # Testovací obrázek
    test_image = "pose.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Testovací obrázek nenalezen: {test_image}")
        print("💡 Vytvoř testovací obrázek nebo uprav cestu")
        sys.exit(1)
    
    # Test RTMPose3D
    rtmpose3d_result = test_rtmpose3d_detector(test_image)
    
    print(f"\n{'='*60}")
    print("✅ Testování dokončeno!")
    print(f"{'='*60}")