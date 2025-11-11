#!/usr/bin/env python3
"""
MMPose detector wrapper pro HRNet a RTMPose
Tento soubor musí běžet v conda prostředí 'openmmlab'
"""

import cv2
import numpy as np
import json
import sys
import os


class MMPoseDetector:
    """Wrapper pro MMPose modely (HRNet, RTMPose)"""
    
    def __init__(self, detector_type="hrnet", confidence_threshold=0.5):
        """
        Args:
            detector_type: "hrnet" nebo "rtmpose"
            confidence_threshold: Minimální confidence pro keypoints (0-1)
        """
        # Lazy import MMPose
        try:
            from mmpose.apis import inference_topdown, init_model
            from mmpose.utils import register_all_modules
            register_all_modules()
            self._inference_topdown = inference_topdown
            self._init_model = init_model
        except ImportError:
            raise ImportError("MMPose není dostupné. Aktivuj conda: conda activate openmmlab")
        
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Cesty k modelům (relativní k Analysis/)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializuje MMPose model"""
        if self.detector_type == "hrnet":
            self._init_hrnet()
        elif self.detector_type == "rtmpose":
            self._init_rtmpose()
        else:
            raise ValueError(f"Neznámý MMPose detektor: {self.detector_type}")
    
    def _init_hrnet(self):
        """Inicializuje HRNet model"""
        config_file = os.path.join(
            self.base_dir,
            'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
        )
        checkpoint_file = os.path.join(
            self.base_dir,
            'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
        )
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"HRNet config nenalezen: {config_file}")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"HRNet checkpoint nenalezen: {checkpoint_file}")
        
        print(f"🔧 Inicializuji HRNet...")
        print(f"   Config: {os.path.basename(config_file)}")
        print(f"   Checkpoint: {os.path.basename(checkpoint_file)}")
        print(f"   Confidence: {self.confidence_threshold}")
        
        self.model = self._init_model(config_file, checkpoint_file, device='cpu')
        print(f"✅ HRNet úspěšně načten (CPU)")
    
    def _init_rtmpose(self):
        """Inicializuje RTMPose model"""
        config_file = os.path.join(
            self.base_dir,
            'RTMPose/rtmpose-l_8xb256-420e_coco-384x288.py' # tohle existuje i ve filu v mmpose a je zbytecne mit tady mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py

        )
        checkpoint_file = os.path.join(
            self.base_dir,
            'RTMPose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth'
        )
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"RTMPose config nenalezen: {config_file}")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"RTMPose checkpoint nenalezen: {checkpoint_file}")
        
        print(f"🔧 Inicializuji RTMPose...")
        print(f"   Config: {os.path.basename(config_file)}")
        print(f"   Checkpoint: {os.path.basename(checkpoint_file)}")
        print(f"   Confidence: {self.confidence_threshold}")
        
        self.model = self._init_model(config_file, checkpoint_file, device='cpu')
        print(f"✅ RTMPose úspěšně načten (CPU)")
    
    def detect_pose(self, frame):
        """
        Detekuje pose v rámci
        
        Args:
            frame: OpenCV obrázek (BGR)
        
        Returns:
            tuple: (keypoints, raw_result)
                - keypoints: MediaPipe formát [x,y,confidence, x,y,confidence, ...]
                - raw_result: Originální MMPose výsledek
        """
        if self.model is None:
            return None, None
        
        # MMPose inference
        results = self._inference_topdown(self.model, frame)
        
        if not results or len(results) == 0:
            return None, None
        
        # Vezmi první (nejlepší) detekci
        result = results[0]
        
        try:
            # Extrakce dat
            data = result.pred_instances.to_dict()
            mmpose_keypoints = data.get('keypoints', None)  # Shape: (17, 2)
            scores = data.get('keypoint_scores', None)      # Shape: (17,)
            
            if mmpose_keypoints is None or scores is None:
                return None, result
            
            # Převod na MediaPipe formát
            mediapipe_keypoints = self._convert_to_mediapipe_format(
                mmpose_keypoints[0],  # První osoba
                scores[0]
            )
            
            return mediapipe_keypoints, result
            
        except Exception as e:
            print(f"⚠️ Chyba při zpracování MMPose výsledku: {e}")
            return None, result
    
    def _convert_to_mediapipe_format(self, mmpose_keypoints, scores):
        """
        Převádí MMPose COCO keypoints (17 bodů) na MediaPipe formát (33 bodů)
        
        Args:
            mmpose_keypoints: numpy array shape (17, 2) - [x, y]
            scores: numpy array shape (17,) - confidence scores
        
        Returns:
            list: MediaPipe formát [x,y,conf, x,y,conf, ...] (33*3 = 99 hodnot)
        """
        # MediaPipe má 33 bodů, MMPose COCO má 17
        mediapipe_keypoints = [0.0] * (33 * 3)
        
        # COCO 17 -> MediaPipe 33 mapping
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
            if coco_idx < len(mmpose_keypoints):
                x, y = mmpose_keypoints[coco_idx]
                confidence = float(scores[coco_idx])
                
                # Pouze body s dostatečným confidence
                if confidence > self.confidence_threshold:
                    base_idx = mediapipe_idx * 3
                    mediapipe_keypoints[base_idx] = float(x)
                    mediapipe_keypoints[base_idx + 1] = float(y)
                    mediapipe_keypoints[base_idx + 2] = confidence
        
        return mediapipe_keypoints
    
    def draw_landmarks(self, frame, detection_result):
        """Vykreslí pose landmarks do snímku"""
        if detection_result is None:
            return
        
        try:
            data = detection_result.pred_instances.to_dict()
            keypoints = data.get('keypoints', None)[0]  # První osoba
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
            
            # Vykreslení bodů
            for i, (x, y) in enumerate(keypoints):
                if scores[i] > self.confidence_threshold:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            
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
        """Uvolní model"""
        self.model = None


def test_mmpose_detector(detector_type, test_image_path):
    """Testovací funkce"""
    print(f"\n{'='*60}")
    print(f"Testování {detector_type.upper()}")
    print(f"{'='*60}\n")
    
    # Inicializace
    detector = MMPoseDetector(detector_type, confidence_threshold=0.5)
    
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
        valid_points = sum(1 for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0.5)
        print(f"📍 Detekováno {valid_points} keypoints (z 33)")
        
        # Vykreslení
        detector.draw_landmarks(frame, result)
        
        # Uložení
        output_path = f"output_{detector_type}_test.jpg"
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
    
    # Test HRNet
    hrnet_result = test_mmpose_detector("hrnet", test_image)
    
    # Test RTMPose
    rtmpose_result = test_mmpose_detector("rtmpose", test_image)
    
    print(f"\n{'='*60}")
    print("✅ Testování dokončeno!")
    print(f"{'='*60}")