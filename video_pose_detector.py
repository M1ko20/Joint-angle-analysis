"""
Video režim pro pose detection s trackingem a optimalizacemi
Rozšíření PoseDetector pro video streaming s lepším využitím temporálních informací
"""

import cv2
import numpy as np
from pose_detector import PoseDetector


class VideoPoseDetector(PoseDetector):
    """
    Rozšířený detektor pro video režim s podporou:
    - Tracking mezi framy
    - Temporální filtrování (smooth)
    - Video-optimalizované konfigurace
    """
    
    def __init__(self, detector_type="mediapipe", smooth_factor=0.5, confidence_threshold=0.5):
        """
        Args:
            detector_type: Typ detektoru (mediapipe, movenet, yolo, atd.)
            smooth_factor: Faktor pro temporální vyhlazování (0-1, 0=vypnuto)
            confidence_threshold: Minimální confidence pro detekci (0-1)
        """
        self.smooth_factor = smooth_factor
        self.prev_keypoints = None
        self.frame_count = 0
        self.video_confidence_threshold = confidence_threshold
        
        # Volání parent konstruktoru
        super().__init__(detector_type)
        
        # Reinicializace s video-specifickými parametry
        self._initialize_video_mode()
    
    def _initialize_video_mode(self):
        """Přenastaví detektory pro video režim"""
        if self.detector_type == "mediapipe":
            self._init_mediapipe_video()
        elif self.detector_type in ["movenet", "movenet_lightning", "movenet_thunder"]:
            # MoveNet už má tracking zabudovaný, žádná změna není potřeba
            pass
        # OpenPose, ViTPose a YOLO nemají speciální video režim
    
    def _init_mediapipe_video(self):
        """Reinicializuje MediaPipe pro video režim"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # VIDEO MODE - static_image_mode=False pro tracking
            # Použij confidence z UI
            conf = self.video_confidence_threshold
            self.detector = self.mp_pose.Pose(
                static_image_mode=False,  # ← KLÍČOVÉ pro video!
                model_complexity=2,
                enable_segmentation=False,
                smooth_landmarks=True,     # ← Vyhlazování pro video
                min_detection_confidence=conf,
                min_tracking_confidence=conf
            )
            print(f"✓ MediaPipe - Video režim aktivován (tracking + smoothing, confidence={conf})")
        except Exception as e:
            print(f"⚠️  MediaPipe video režim selhal: {e}")
    
    def detect_pose(self, frame):
        """
        Detekuje pose s video optimalizacemi
        Returns: (keypoints, detection_result)
        """
        self.frame_count += 1
        
        # Detekce
        if self.detector_type == "mediapipe":
            keypoints, result = self._detect_mediapipe_video(frame)
        else:
            # Pro ostatní použij standardní detekci
            keypoints, result = super().detect_pose(frame)
        
        # Temporální vyhlazování (smoothing)
        if keypoints is not None and self.smooth_factor > 0:
            keypoints = self._smooth_keypoints(keypoints)
        
        self.prev_keypoints = keypoints
        return keypoints, result
    
    def _detect_mediapipe_video(self, frame):
        """MediaPipe video detection - používá tracking"""
        # MediaPipe s static_image_mode=False už má tracking zabudovaný
        return super()._detect_mediapipe(frame)
    
    def _smooth_keypoints(self, keypoints):
        """
        Temporální vyhlazování keypoints pomocí exponenciálního průměru
        Args:
            keypoints: Aktuální keypoints
        Returns:
            Vyhlazené keypoints
        """
        if self.prev_keypoints is None:
            return keypoints
        
        if keypoints is None:
            return self.prev_keypoints
        
        # Převod na numpy array
        if isinstance(keypoints, list):
            keypoints = np.array(keypoints)
        if isinstance(self.prev_keypoints, list):
            prev = np.array(self.prev_keypoints)
        else:
            prev = self.prev_keypoints
        
        # Ujisti se, že mají stejnou velikost
        if keypoints.shape != prev.shape:
            return keypoints
        
        # Exponenciální moving average
        # smoothed = α * current + (1-α) * previous
        alpha = 1.0 - self.smooth_factor
        smoothed = alpha * keypoints + (1.0 - alpha) * prev
        
        # Zachovej confidence scores z aktuálních keypoints
        if len(keypoints.shape) == 1:
            # Flat array [x,y,c, x,y,c, ...]
            for i in range(2, len(keypoints), 3):
                smoothed[i] = keypoints[i]  # Confidence
        elif len(keypoints.shape) == 2:
            # Array [[x,y,c], [x,y,c], ...]
            smoothed[:, 2] = keypoints[:, 2]  # Confidence column
        
        return smoothed
    
    def reset_tracking(self):
        """Resetuje tracking informace (při změně videa nebo scény)"""
        self.prev_keypoints = None
        self.frame_count = 0
        
        # Reset crop region pro MoveNet
        if self.detector_type in ["movenet", "movenet_lightning", "movenet_thunder"]:
            self.crop_region = None
        
        print(f"✓ {self.detector_type} - Tracking resetován")
    
    def get_tracking_info(self):
        """Vrátí informace o trackingu"""
        return {
            'frame_count': self.frame_count,
            'has_prev_keypoints': self.prev_keypoints is not None,
            'smooth_factor': self.smooth_factor
        }


def get_video_capable_detectors():
    """
    Vrací seznam detektorů s podporou video režimu
    Returns:
        dict: {detector_name: capabilities}
    """
    from pose_detector import (MEDIAPIPE_AVAILABLE, MOVENET_AVAILABLE, 
                               YOLO_AVAILABLE, OPENPOSE_AVAILABLE, 
                               VITPOSE_AVAILABLE)
    
    detectors = {}
    
    if MEDIAPIPE_AVAILABLE:
        detectors['mediapipe'] = {
            'video_support': True,
            'tracking': True,
            'smoothing': True,
            'name': 'MediaPipe'
        }
    
    if MOVENET_AVAILABLE:
        detectors['movenet_lightning'] = {
            'video_support': True,
            'tracking': True,
            'smoothing': True,
            'name': 'MoveNet Lightning'
        }
        detectors['movenet_thunder'] = {
            'video_support': True,
            'tracking': True,
            'smoothing': True,
            'name': 'MoveNet Thunder'
        }
    
    if YOLO_AVAILABLE:
        detectors['yolo11n'] = {
            'video_support': False,  # YOLO nemá speciální video režim
            'tracking': False,
            'smoothing': False,
            'name': 'YOLO11n'
        }
        detectors['yolo11x'] = {
            'video_support': False,  # YOLO nemá speciální video režim
            'tracking': False,
            'smoothing': False,
            'name': 'YOLO11x'
        }
    
    if OPENPOSE_AVAILABLE:
        detectors['openpose'] = {
            'video_support': False,  # OpenPose nemá speciální video režim
            'tracking': False,
            'smoothing': False,
            'name': 'OpenPose'
        }
    
    if VITPOSE_AVAILABLE:
        detectors['vitpose'] = {
            'video_support': False,  # ViTPose nemá speciální video režim
            'tracking': False,
            'smoothing': False,
            'name': 'ViTPose'
        }
    
    return detectors


if __name__ == "__main__":
    # Test video režimu
    print("🎥 Testování video režimu detektorů...")
    
    capable = get_video_capable_detectors()
    
    print("\n✅ Detektory s video podporou:")
    for name, caps in capable.items():
        if caps['video_support']:
            features = []
            if caps['tracking']:
                features.append("tracking")
            if caps['smoothing']:
                features.append("smoothing")
            print(f"  • {caps['name']}: {', '.join(features)}")
    
    print("\n❌ Detektory pouze pro obrázky:")
    for name, caps in capable.items():
        if not caps['video_support']:
            print(f"  • {caps['name']}")
