#!/usr/bin/env python3
"""
MMPose detector wrapper pro HRNet, RTMPose a RTMPose3D
Tento soubor musí běžet v conda prostředí 'openmmlab'
"""

import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path

try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import register_all_modules
    MMPOSE_AVAILABLE = True
    register_all_modules()
except ImportError:
    MMPOSE_AVAILABLE = False
    print("❌ MMPose není k dispozici. Spusť v conda prostředí: conda activate openmmlab")

# Pro RTMPose3D
try:
    from mmdet.apis import inference_detector, init_detector
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False


class MMPoseDetector:
    """Wrapper pro MMPose modely (HRNet, RTMPose, RTMPose3D)"""
    
    def __init__(self, detector_type="hrnet", confidence_threshold=0.5):
        """
        Args:
            detector_type: "hrnet", "rtmpose", nebo "rtmpose3d"
            confidence_threshold: Minimální confidence pro keypoints (0-1)
        """
        if not MMPOSE_AVAILABLE:
            raise ImportError("MMPose není dostupné. Aktivuj conda: conda activate openmmlab")
        
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.person_detector = None  # Pro RTMPose3D
        self.is_3d = (detector_type == "rtmpose3d")
        
        # Cesty k modelům (relativní k Analysis/)
        self.base_dir = Path(__file__).parent.absolute()
        self.mmpose_dir = self.base_dir / "mmpose"
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializuje MMPose model"""
        if self.detector_type == "hrnet":
            self._init_hrnet()
        elif self.detector_type == "rtmpose":
            self._init_rtmpose()
        elif self.detector_type == "rtmpose3d":
            if not MMDET_AVAILABLE:
                raise ImportError("MMDet není dostupné pro RTMPose3D")
            self._init_rtmpose3d()
        else:
            raise ValueError(f"Neznámý MMPose detektor: {self.detector_type}")
    
    def _init_hrnet(self):
        """Inicializuje HRNet model"""
        config_file = self.base_dir / 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
        checkpoint_file = self.base_dir / 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
        
        if not config_file.exists():
            raise FileNotFoundError(f"HRNet config nenalezen: {config_file}")
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"HRNet checkpoint nenalezen: {checkpoint_file}")
        
        print(f"🔧 Inicializuji HRNet...")
        print(f"   Config: {config_file.name}")
        print(f"   Checkpoint: {checkpoint_file.name}")
        print(f"   Confidence: {self.confidence_threshold}")
        
        self.model = init_model(str(config_file), str(checkpoint_file), device='cpu')
        print(f"✅ HRNet úspěšně načten (CPU)")
    
    def _init_rtmpose(self):
        """Inicializuje RTMPose model"""
        config_file = self.base_dir / 'RTMPose/rtmpose-l_8xb256-420e_coco-384x288.py'
        checkpoint_file = self.base_dir / 'RTMPose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth'
        
        if not config_file.exists():
            raise FileNotFoundError(f"RTMPose config nenalezen: {config_file}")
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"RTMPose checkpoint nenalezen: {checkpoint_file}")
        
        print(f"🔧 Inicializuji RTMPose...")
        print(f"   Config: {config_file.name}")
        print(f"   Checkpoint: {checkpoint_file.name}")
        print(f"   Confidence: {self.confidence_threshold}")
        
        self.model = init_model(str(config_file), str(checkpoint_file), device='cpu')
        print(f"✅ RTMPose úspěšně načten (CPU)")
    
    def _init_rtmpose3d(self):
        """Inicializuje RTMPose3D model (2-stage)"""
        print(f"🔧 Inicializuji RTMPose3D...")
        
        # 1. Person detector (RT-DETR)
        det_config = self.mmpose_dir / "projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py"
        det_checkpoint_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
        
        if not det_config.exists():
            raise FileNotFoundError(f"Detection config nenalezen: {det_config}")
        
        print(f"   Loading person detector...")
        print(f"   Config: {det_config.name}")
        print(f"   Checkpoint: {det_checkpoint_url}")
        
        self.person_detector = init_detector(
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
        
        self.model = init_model(
            str(pose_config),
            str(pose_checkpoint),
            device='cpu'
        )
        print(f"✅ RTMPose3D úspěšně načten (CPU)")
    
    def detect_pose(self, frame):
        """
        Detekuje pose v rámci
        
        Args:
            frame: OpenCV obrázek (BGR)
        
        Returns:
            tuple: (keypoints, raw_result)
                - keypoints: MediaPipe formát [x,y,confidence, ...] nebo [x,y,z,confidence, ...] pro 3D
                - raw_result: Originální MMPose výsledek
        """
        if self.model is None:
            return None, None
        
        if self.detector_type == "rtmpose3d":
            return self._detect_rtmpose3d(frame)
        else:
            return self._detect_2d(frame)
    
    def _detect_2d(self, frame):
        """2D detekce (HRNet, RTMPose)"""
        # MMPose inference
        results = inference_topdown(self.model, frame)
        
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
    
    def _detect_rtmpose3d(self, frame):
        """3D detekce (RTMPose3D)"""
        # Stage 1: Person detection
        det_result = inference_detector(self.person_detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()
        
        # Filter person bboxes
        bboxes = pred_instance.bboxes
        scores = pred_instance.scores
        labels = pred_instance.labels
        
        person_mask = np.logical_and(
            labels == 0,  # person class
            scores > self.confidence_threshold
        )
        bboxes = bboxes[person_mask]
        
        if len(bboxes) == 0:
            return None, None
        
        # Stage 2: Pose estimation
        pose_results = inference_topdown(self.model, frame, bboxes)
        
        if not pose_results or len(pose_results) == 0:
            return None, None
        
        result = pose_results[0]
        
        try:
            data = result.pred_instances.to_dict()
            keypoints_3d = data.get('keypoints', None)  # Shape: (17, 3)
            scores = data.get('keypoint_scores', None)  # Shape: (17,)
            
            if keypoints_3d is None or scores is None:
                return None, result
            
            # Převod na MediaPipe 3D formát
            mediapipe_keypoints = self._convert_to_mediapipe_3d_format(
                keypoints_3d[0],
                scores[0]
            )
            
            return mediapipe_keypoints, result
            
        except Exception as e:
            print(f"⚠️ Chyba při zpracování RTMPose3D výsledku: {e}")
            return None, result
    
    def _convert_to_mediapipe_format(self, mmpose_keypoints, scores):
        """
        Převádí MMPose COCO keypoints (17 bodů) na MediaPipe formát (33 bodů) - 2D
        
        Args:
            mmpose_keypoints: numpy array shape (17, 2) - [x, y]
            scores: numpy array shape (17,) - confidence scores
        
        Returns:
            list: MediaPipe formát [x,y,conf, x,y,conf, ...] (33*3 = 99 hodnot)
        """
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
                
                if confidence > self.confidence_threshold:
                    base_idx = mediapipe_idx * 3
                    mediapipe_keypoints[base_idx] = float(x)
                    mediapipe_keypoints[base_idx + 1] = float(y)
                    mediapipe_keypoints[base_idx + 2] = confidence
        
        return mediapipe_keypoints
    
    def _convert_to_mediapipe_3d_format(self, mmpose_keypoints, scores):
        """
        Převádí MMPose COCO keypoints (17 bodů) na MediaPipe formát (33 bodů) - 3D
        
        Args:
            mmpose_keypoints: numpy array shape (17, 3) - [x, y, z]
            scores: numpy array shape (17,) - confidence scores
        
        Returns:
            list: MediaPipe formát [x,y,z,conf, x,y,z,conf, ...] (33*4 = 132 hodnot)
        """
        mediapipe_keypoints = [0.0] * (33 * 4)
        
        mapping = {
            0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
            5: 11, 6: 12, 7: 13, 8: 14,
            9: 15, 10: 16, 11: 23, 12: 24,
            13: 25, 14: 26, 15: 27, 16: 28
        }
        
        for coco_idx, mediapipe_idx in mapping.items():
            if coco_idx < len(mmpose_keypoints):
                x, y, z = mmpose_keypoints[coco_idx]
                confidence = float(scores[coco_idx])
                
                if confidence > self.confidence_threshold:
                    base_idx = mediapipe_idx * 4
                    mediapipe_keypoints[base_idx] = float(x)
                    mediapipe_keypoints[base_idx + 1] = float(y)
                    mediapipe_keypoints[base_idx + 2] = float(z)
                    mediapipe_keypoints[base_idx + 3] = confidence
        
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
            for i, kp in enumerate(keypoints):
                if scores[i] > self.confidence_threshold:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            
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
        self.person_detector = None


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
        stride = 4 if detector.is_3d else 3
        valid_points = sum(1 for i in range(0, len(keypoints), stride) if keypoints[i+stride-1] > 0.5)
        print(f"🔍 Detekováno {valid_points} keypoints (z 33)")
        
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
    if not MMPOSE_AVAILABLE:
        print("\n❌ MMPose není dostupné!")
        print("💡 Spusť v conda prostředí: conda activate openmmlab")
        sys.exit(1)
    
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
    
    # Test RTMPose3D
    if MMDET_AVAILABLE:
        rtmpose3d_result = test_mmpose_detector("rtmpose3d", test_image)
    
    print(f"\n{'='*60}")
    print("✅ Testování dokončeno!")
    print(f"{'='*60}")