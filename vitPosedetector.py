"""
ViTPose detektor pomocí Hugging Face Transformers (Mac compatible)
Použití: Tento modul přidá podporu ViTPose bez mmpose závislostí
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Zkontroluj dostupnost Transformers a dalších závislostí
try:
    import torch
    from PIL import Image
    VITPOSE_HF_AVAILABLE = True
    
    # Importy z transformers - různé verze mají různá API
    try:
        from transformers import AutoImageProcessor as AutoProcessor
    except ImportError:
        from transformers import AutoProcessor
    
    from transformers import VitPoseForPoseEstimation, RTDetrForObjectDetection
    
except ImportError as e:
    VITPOSE_HF_AVAILABLE = False
    print(f"⚠️  ViTPose (HF) není k dispozici: {e}")
    print("📦 Nainstalujte: pip install transformers torch pillow accelerate")


class ViTPoseHFDetector:
    """
    ViTPose detektor pomocí Hugging Face Transformers
    Plně funkční na Macu bez mmpose
    """
    
    def __init__(self, model_name="usyd-community/vitpose-base-simple", confidence_threshold=0.5):
        """
        Args:
            model_name: HuggingFace model ID
            confidence_threshold: Minimální confidence pro detekci
        """
        if not VITPOSE_HF_AVAILABLE:
            raise ImportError("ViTPose vyžaduje: pip install transformers torch pillow")
        
        self.confidence_threshold = confidence_threshold
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"🔧 Inicializuji ViTPose...")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_name}")
        
        # 1. Person detector (RT-DETR)
        print("   Načítám person detector...")
        self.person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365"
        ).to(self.device)
        
        # 2. Pose estimator (ViTPose)
        print("   Načítám pose estimator...")
        self.pose_processor = AutoProcessor.from_pretrained(model_name)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            model_name
        ).to(self.device)
        
        print("✅ ViTPose úspěšně načten")
        
        # COCO keypoints mapping (17 bodů)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
    def detect_pose(self, frame):
        """
        Detekuje pose v rámci
        Returns: (keypoints, raw_results)
        """
        height, width = frame.shape[:2]
        
        # Konverze frame na PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        try:
            # Krok 1: Detekuj osoby
            person_boxes = self._detect_persons(image)
            
            if person_boxes is None or len(person_boxes) == 0:
                return None, None
            
            # Krok 2: Detekuj keypoints pro každou osobu
            pose_results = self._detect_keypoints(image, person_boxes)
            
            if not pose_results:
                return None, None
            
            # Převeď na MediaPipe formát (nejlepší detekce)
            best_result = pose_results[0]  # Bere první (nejlepší) detekci
            keypoints = self._convert_to_mediapipe_format(
                best_result['keypoints'],
                best_result['scores'],
                width,
                height
            )
            
            return keypoints, pose_results
            
        except Exception as e:
            print(f"⚠️  ViTPose detekce selhala: {e}")
            return None, None
    
    def _detect_persons(self, image):
        """Detekuje osoby v obrázku pomocí RT-DETR"""
        inputs = self.person_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.person_model(**inputs)
        
        # Post-process detekce
        results = self.person_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)]),
            threshold=0.3
        )
        
        if not results:
            return None
        
        result = results[0]
        
        # Filtruj pouze osoby (label 0 v COCO)
        person_mask = result["labels"] == 0
        person_boxes = result["boxes"][person_mask]
        
        if len(person_boxes) == 0:
            return None
        
        # Konverze z VOC (x1,y1,x2,y2) na COCO (x1,y1,w,h)
        person_boxes = person_boxes.cpu().numpy()
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]  # width
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]  # height
        
        return person_boxes
    
    def _detect_keypoints(self, image, boxes):
        """Detekuje keypoints pro dané bounding boxy"""
        inputs = self.pose_processor(
            image,
            boxes=[boxes],
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.pose_model(**inputs)
        
        # Post-process
        pose_results = self.pose_processor.post_process_pose_estimation(
            outputs,
            boxes=[boxes]
        )
        
        return pose_results[0] if pose_results else []
    
    def _convert_to_mediapipe_format(self, keypoints, scores, width, height):
        """Převede ViTPose (COCO 17) keypoints na MediaPipe formát (33 bodů)"""
        mediapipe_keypoints = [0.0] * (33 * 3)
        
        # ViTPose COCO -> MediaPipe mapping
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
        
        for vitpose_idx, mediapipe_idx in mapping.items():
            if vitpose_idx < len(keypoints):
                x, y = keypoints[vitpose_idx]
                confidence = scores[vitpose_idx] if vitpose_idx < len(scores) else 0.0
                
                if confidence > self.confidence_threshold:
                    base_idx = mediapipe_idx * 3
                    mediapipe_keypoints[base_idx] = float(x)
                    mediapipe_keypoints[base_idx + 1] = float(y)
                    mediapipe_keypoints[base_idx + 2] = float(confidence)
        
        return mediapipe_keypoints
    
    def draw_landmarks(self, frame, detection_result):
        """Vykreslí keypoints do snímku"""
        if detection_result is None or not detection_result:
            return
        
        try:
            result = detection_result[0]  # První (nejlepší) detekce
            keypoints = result['keypoints']
            scores = result['scores']
            
            # Vykreslení bodů
            for i, (x, y) in enumerate(keypoints):
                if scores[i] > self.confidence_threshold:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                    
                    # Popisek (volitelně)
                    if i < len(self.keypoint_names):
                        cv2.putText(
                            frame,
                            self.keypoint_names[i][:3],
                            (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1
                        )
            
            # Vykreslení skeletu
            for start_idx, end_idx in self.skeleton:
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    scores[start_idx] > self.confidence_threshold and
                    scores[end_idx] > self.confidence_threshold):
                    
                    start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                    end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                    
        except Exception as e:
            print(f"⚠️  Chyba při vykreslování: {e}")
    
    def close(self):
        """Uvolní zdroje"""
        self.person_model = None
        self.pose_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def test_vitpose():
    """Test funkce"""
    if not VITPOSE_HF_AVAILABLE:
        print("❌ ViTPose není k dispozici")
        return
    
    print("🧪 Test ViTPose detektoru...")
    
    try:
        detector = ViTPoseHFDetector()
        print("✅ Detektor inicializován")
        
        # Test na dummy obrázku
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints, results = detector.detect_pose(dummy_frame)
        
        if keypoints is None:
            print("⚠️  Žádná detekce (očekáváno pro prázdný frame)")
        else:
            print(f"✅ Detekce funguje: {len(keypoints)} hodnot")
        
        detector.close()
        print("✅ Test dokončen")
        
    except Exception as e:
        print(f"❌ Test selhal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vitpose()