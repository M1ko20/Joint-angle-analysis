"""
ViTPose detektor optimalizovaný pro měření rozsahu pohybu (goniometrie)
Použití: Vysokopřesná detekce pózy pro fyzioterapeutické aplikace
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Zkontroluj dostupnost Transformers a dalších závislostí
try:
    import torch
    from PIL import Image
    VITPOSE_HF_AVAILABLE = True
    
    try:
        from transformers import AutoImageProcessor as AutoProcessor
    except ImportError:
        from transformers import AutoProcessor
    
    from transformers import VitPoseForPoseEstimation, RTDetrForObjectDetection
    
except ImportError as e:
    VITPOSE_HF_AVAILABLE = False
    print(f"⚠️  ViTPose (HF) není k dispozici: {e}")
    print("📦 Nainstalujte: pip install transformers torch pillow accelerate")


class ViTPoseGoniometryDetector:
    """
    ViTPose detektor optimalizovaný pro goniometrická měření
    
    Doporučené modely podle použití:
    - "usyd-community/vitpose-plus-large" - optimální poměr přesnost/rychlost (DOPORUČENO)
    - "usyd-community/vitpose-plus-huge" - maximální přesnost pro kritická měření
    - "usyd-community/vitpose-plus-base" - rychlé zpracování, nižší přesnost
    """
    
    # Dostupné modely s parametry
    AVAILABLE_MODELS = {
        'base': {
            'id': 'usyd-community/vitpose-plus-base',
            'accuracy': 'střední',
            'speed': 'velmi rychlý',
            'use_case': 'real-time zpracování, rychlý náhled'
        },
        'large': {
            'id': 'usyd-community/vitpose-plus-large',
            'accuracy': 'vysoká',
            'speed': 'rychlý',
            'use_case': 'goniometrie, běžná klinická měření (DOPORUČENO)'
        },
        'huge': {
            'id': 'usyd-community/vitpose-plus-huge',
            'accuracy': 'velmi vysoká',
            'speed': 'pomalejší',
            'use_case': 'precizní měření, výzkum, offline analýza'
        }
    }
    
    def __init__(self, 
                 model_size: str = "large",
                 confidence_threshold: float = 0.6,
                 dataset_index: int = 0):
        """
        Args:
            model_size: Velikost modelu ('base', 'large', 'huge')
            confidence_threshold: Minimální confidence pro keypoint (0.6 doporučeno pro goniometrii)
            dataset_index: MoE dataset index (0=COCO val, 1=AiC, 2=MPII, atd.)
        """
        if not VITPOSE_HF_AVAILABLE:
            raise ImportError("ViTPose vyžaduje: pip install transformers torch pillow accelerate")
        
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_size}' není podporován. Použijte: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_config = self.AVAILABLE_MODELS[model_size]
        model_name = self.model_config['id']
        self.confidence_threshold = confidence_threshold
        self.dataset_index = dataset_index
        
        # Detekce optimálního zařízení
        self.device = self._get_optimal_device()
        
        print(f"\n{'='*60}")
        print(f"🏥 ViTPose Goniometry Detector")
        print(f"{'='*60}")
        print(f"📊 Model: {model_size.upper()}")
        print(f"   ID: {model_name}")
        print(f"   Přesnost: {self.model_config['accuracy']}")
        print(f"   Rychlost: {self.model_config['speed']}")
        print(f"   Účel: {self.model_config['use_case']}")
        print(f"🔧 Device: {self.device}")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        print(f"📁 Dataset index: {dataset_index}")
        print(f"{'='*60}\n")
        
        # 1. Person detector (RT-DETR)
        print("📥 Načítám person detector (RT-DETR)...")
        self.person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365"
        ).to(self.device)
        self.person_model.eval()
        
        # 2. Pose estimator (ViTPose)
        print(f"📥 Načítám pose estimator ({model_size})...")
        self.pose_processor = AutoProcessor.from_pretrained(model_name)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            model_name
        ).to(self.device)
        self.pose_model.eval()
        
        print("✅ Modely úspěšně načteny\n")
        
        # COCO keypoints mapping (17 bodů) - standard pro goniometrii
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections pro vizualizaci
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Hlava
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Paže
            (5, 11), (6, 12), (11, 12),  # Trup
            (11, 13), (13, 15), (12, 14), (14, 16)  # Nohy
        ]
        
        # Důležité úhly pro goniometrii
        self.goniometry_angles = {
            'left_elbow': (5, 7, 9),      # rameno-loket-zápěstí
            'right_elbow': (6, 8, 10),
            'left_shoulder': (7, 5, 11),   # loket-rameno-kyčel
            'right_shoulder': (8, 6, 12),
            'left_hip': (5, 11, 13),       # rameno-kyčel-koleno
            'right_hip': (6, 12, 14),
            'left_knee': (11, 13, 15),     # kyčel-koleno-kotník
            'right_knee': (12, 14, 16),
        }
    
    def _get_optimal_device(self) -> str:
        """Detekuje nejlepší dostupné zařízení"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def detect_pose(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """
        Detekuje pózu v rámci s vysokou přesností pro goniometrii
        
        Returns:
            keypoints: numpy array pro kompatibilitu s MediaPipe formátem
            raw_results: seznam detekovaných postav s keypoints a scores
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
            
            # Krok 2: Detekuj keypoints s MoE dataset indexem
            pose_results = self._detect_keypoints(image, person_boxes)
            
            if not pose_results:
                return None, None
            
            # Převeď na standardní numpy formát
            best_result = pose_results[0]
            keypoints_array = np.array(best_result['keypoints'])
            scores_array = np.array(best_result['scores'])
            
            # Přidej metadata
            for result in pose_results:
                result['confidence_threshold'] = self.confidence_threshold
                result['model_size'] = self.model_config['id']
            
            return keypoints_array, pose_results
            
        except Exception as e:
            print(f"⚠️  ViTPose detekce selhala: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _detect_persons(self, image: Image.Image) -> Optional[np.ndarray]:
        """Detekuje osoby v obrázku pomocí RT-DETR"""
        inputs = self.person_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.person_model(**inputs)
        
        results = self.person_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)]),
            threshold=0.3
        )
        
        if not results:
            return None
        
        result = results[0]
        person_mask = result["labels"] == 0  # COCO label 0 = person
        person_boxes = result["boxes"][person_mask]
        
        if len(person_boxes) == 0:
            return None
        
        # VOC (x1,y1,x2,y2) -> COCO (x1,y1,w,h)
        person_boxes = person_boxes.cpu().numpy()
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
        
        return person_boxes
    
    def _detect_keypoints(self, image: Image.Image, boxes: np.ndarray) -> List[Dict]:
        """Detekuje keypoints s použitím MoE dataset indexu"""
        inputs = self.pose_processor(
            image,
            boxes=[boxes],
            return_tensors="pt"
        ).to(self.device)
        
        # Přidání dataset_index pro MoE modely (ViTPose++)
        dataset_index_tensor = torch.tensor([self.dataset_index] * len(boxes), device=self.device)
        
        with torch.no_grad():
            outputs = self.pose_model(**inputs, dataset_index=dataset_index_tensor)
        
        pose_results = self.pose_processor.post_process_pose_estimation(
            outputs,
            boxes=[boxes]
        )
        
        return pose_results[0] if pose_results else []
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Vypočítá úhel mezi třemi body (vertex je p2)
        
        Args:
            p1, p2, p3: body jako [x, y]
            
        Returns:
            Úhel ve stupních (0-180°)
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def calculate_goniometry_angles(self, keypoints: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """
        Vypočítá všechny relevantní úhly pro goniometrii
        
        Returns:
            Dictionary s názvy úhlů a jejich hodnotami ve stupních
        """
        angles = {}
        
        for angle_name, (idx1, idx2, idx3) in self.goniometry_angles.items():
            # Kontrola, že všechny body mají dostatečnou confidence
            if (scores[idx1] > self.confidence_threshold and 
                scores[idx2] > self.confidence_threshold and 
                scores[idx3] > self.confidence_threshold):
                
                p1 = keypoints[idx1]
                p2 = keypoints[idx2]
                p3 = keypoints[idx3]
                
                angle = self.calculate_angle(p1, p2, p3)
                angles[angle_name] = angle
            else:
                angles[angle_name] = None
        
        return angles
    
    def draw_landmarks(self, frame: np.ndarray, detection_result: List[Dict], 
                      show_angles: bool = True) -> None:
        """
        Vykreslí keypoints a skeleton do snímku
        
        Args:
            frame: OpenCV frame
            detection_result: výsledky detekce
            show_angles: zobrazit vypočítané úhly
        """
        if detection_result is None or not detection_result:
            return
        
        try:
            result = detection_result[0]
            keypoints = np.array(result['keypoints'])
            scores = np.array(result['scores'])
            
            # Vykreslení skeletu (nejprve, aby byly pod body)
            for start_idx, end_idx in self.skeleton:
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    scores[start_idx] > self.confidence_threshold and
                    scores[end_idx] > self.confidence_threshold):
                    
                    start_point = tuple(keypoints[start_idx].astype(int))
                    end_point = tuple(keypoints[end_idx].astype(int))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            
            # Vykreslení bodů
            for i, (x, y) in enumerate(keypoints):
                if scores[i] > self.confidence_threshold:
                    color = (0, 0, 255) if scores[i] > 0.8 else (0, 165, 255)
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1)
            
            # Vykreslení úhlů
            if show_angles:
                angles = self.calculate_goniometry_angles(keypoints, scores)
                y_offset = 30
                
                for angle_name, angle_value in angles.items():
                    if angle_value is not None:
                        text = f"{angle_name}: {angle_value:.1f}°"
                        cv2.putText(frame, text, (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, text, (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        y_offset += 25
                    
        except Exception as e:
            print(f"⚠️  Chyba při vykreslování: {e}")
    
    def get_model_info(self) -> Dict:
        """Vrátí informace o aktuálním modelu"""
        return {
            'model_name': self.model_config['id'],
            'accuracy': self.model_config['accuracy'],
            'speed': self.model_config['speed'],
            'use_case': self.model_config['use_case'],
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'dataset_index': self.dataset_index
        }
    
    def close(self):
        """Uvolní zdroje"""
        self.person_model = None
        self.pose_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def test_vitpose_goniometry(image_path: str, model_size: str = "large"):
    """
    Testovací funkce pro goniometrický detektor
    
    Args:
        image_path: cesta k testovacímu obrázku
        model_size: 'base', 'large', nebo 'huge'
    """
    if not VITPOSE_HF_AVAILABLE:
        print("❌ ViTPose není k dispozici")
        return

    print(f"\n🧪 Test ViTPose Goniometry Detector")
    print(f"📁 Obrázek: {image_path}")
    print(f"📊 Model: {model_size}")

    try:
        # Načtení obrázku
        frame = None
        if image_path.startswith('http'):
            print(f"🌐 Stahuji obrázek z URL...")
            import requests
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            print(f"📁 Načítám lokální obrázek: {image_path}")
            import os
            
            # Kontrola existence souboru
            if not os.path.exists(image_path):
                print(f"❌ Soubor neexistuje: {image_path}")
                print(f"📍 Aktuální adresář: {os.getcwd()}")
                print(f"📂 Dostupné obrázky:")
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    import glob
                    files = glob.glob(ext)
                    if files:
                        for f in files[:5]:  # Show first 5
                            print(f"   - {f}")
                return
            
            # Pokus načíst pomocí Pillow (robustnější)
            try:
                with Image.open(image_path) as img:
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                print(f"✅ Načteno pomocí Pillow")
            except Exception as e:
                print(f"⚠️  Pillow selhalo: {e}")
                print("🔄 Zkouším cv2.imread...")
                frame = cv2.imread(image_path)
                if frame is not None:
                    print(f"✅ Načteno pomocí OpenCV")

        if frame is None:
            print("❌ Nepodařilo se načíst obrázek žádnou metodou")
            print("💡 Zkuste jiný obrázek nebo URL")
            return

        print(f"✅ Obrázek načten: {frame.shape}")

        # Inicializace detektoru
        detector = ViTPoseGoniometryDetector(
            model_size=model_size,
            confidence_threshold=0.6,
            dataset_index=0
        )

        # Detekce
        print("\n🔍 Provádím detekci pózy...")
        keypoints, results = detector.detect_pose(frame)

        if results:
            print(f"✅ Detekováno {len(results)} postav")
            
            # Vypočítat úhly
            result = results[0]
            angles = detector.calculate_goniometry_angles(
                np.array(result['keypoints']),
                np.array(result['scores'])
            )
            
            print("\n📐 Goniometrická měření:")
            print("-" * 40)
            for angle_name, angle_value in angles.items():
                if angle_value is not None:
                    print(f"  {angle_name:20s}: {angle_value:6.2f}°")
                else:
                    print(f"  {angle_name:20s}: N/A (nízká confidence)")
            
            # Vykreslení
            detector.draw_landmarks(frame, results, show_angles=True)
            
            # Uložení výsledku
            output_path = f"goniometry_output_{model_size}.jpg"
            cv2.imwrite(output_path, frame)
            print(f"\n✅ Výstup uložen: {output_path}")
            
            # Info o modelu
            model_info = detector.get_model_info()
            print(f"\n📊 Informace o modelu:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
        else:
            print("⚠️  Žádná postava nebyla detekována")

        detector.close()
        print("\n✅ Test dokončen")

    except Exception as e:
        print(f"❌ Test selhal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ViTPose Goniometry Detector - optimalizováno pro měření rozsahu pohybu"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="pose.jpg",
        help="Cesta nebo URL k testovacímu obrázku"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['base', 'large', 'huge'],
        default='large',
        help="Velikost modelu (default: large - doporučeno)"
    )
    
    args = parser.parse_args()
    test_vitpose_goniometry(args.image, args.model)