"""
Modulární pose detection systém s podporou MediaPipe a OpenPose
"""

import cv2
import numpy as np
import os
import sys

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe není k dispozici. Nainstalujte: pip install mediapipe")

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    MOVENET_AVAILABLE = True
    
    # MoveNet konstanty podle oficiální dokumentace
    KEYPOINT_DICT = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
    
    # Confidence score pro určení spolehlivosti keypoint predikce
    MIN_CROP_KEYPOINT_SCORE = 0.2
    
except ImportError:
    MOVENET_AVAILABLE = False
    print("MoveNet není k dispozici. Nainstalujte: pip install tensorflow tensorflow-hub")

try:
    # OpenPose import - může vyžadovat speciální instalaci
    sys.path.append('/usr/local/python')  # Typická cesta pro OpenPose
    from openpose import pyopenpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    print("OpenPose není k dispozici. Zkontrolujte instalaci OpenPose.")


class PoseDetector:
    """Abstraktní třída pro pose detection"""
    
    def __init__(self, detector_type="mediapipe"):
        self.detector_type = detector_type.lower()
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Inicializuje vybraný detektor"""
        if self.detector_type == "mediapipe":
            if not MEDIAPIPE_AVAILABLE:
                raise ImportError("MediaPipe není k dispozici")
            self._init_mediapipe()
        elif self.detector_type in ["movenet", "movenet_lightning", "movenet_thunder"]:
            if not MOVENET_AVAILABLE:
                raise ImportError("MoveNet není k dispozici")
            self._init_movenet()
        elif self.detector_type == "openpose":
            if not OPENPOSE_AVAILABLE:
                raise ImportError("OpenPose není k dispozici")
            self._init_openpose()
        else:
            raise ValueError(f"Neznámý typ detektoru: {self.detector_type}")
    
    def _init_mediapipe(self):
        """Inicializuje MediaPipe"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe inicializován")
    
    def _init_movenet(self):
        """Inicializuje MoveNet"""
        # Výběr modelu podle typu
        if self.detector_type == "movenet_thunder":
            model_name = "thunder"
            self.input_size = 256
        else:  # lightning nebo obecný movenet
            model_name = "lightning"
            self.input_size = 192
        
        model_url = f"https://tfhub.dev/google/movenet/singlepose/{model_name}/4"
        
        try:
            self.detector = hub.load(model_url)
            self.movenet = self.detector.signatures['serving_default']
            # Inicializace crop region pro první frame
            self.crop_region = None
            print(f"MoveNet inicializován ({model_name})")
        except Exception as e:
            print(f"Chyba při načítání MoveNet modelu: {e}")
            raise
    
    def _init_openpose(self):
        """Inicializuje OpenPose"""
        params = dict()
        params["model_folder"] = "/usr/local/share/OpenPose/models/"  # Upravte cestu podle vaší instalace
        params["model_pose"] = "BODY_25"  # Nebo "COCO", "MPI"
        params["net_resolution"] = "368x368"
        
        try:
            self.detector = op.WrapperPython()
            self.detector.configure(params)
            self.detector.start()
            print("OpenPose inicializován")
        except Exception as e:
            print(f"Chyba při inicializaci OpenPose: {e}")
            raise
    
    def detect_pose(self, frame):
        """Detekuje pose v rámci a vrací normalizované keypoints"""
        if self.detector_type == "mediapipe":
            return self._detect_mediapipe(frame)
        elif self.detector_type in ["movenet", "movenet_lightning", "movenet_thunder"]:
            return self._detect_movenet(frame)
        elif self.detector_type == "openpose":
            return self._detect_openpose(frame)
    
    def _detect_mediapipe(self, frame):
        """MediaPipe pose detection"""
        height, width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image_rgb)
        
        if results.pose_landmarks:
            # Normalizované keypoints [x, y, visibility] pro každý bod
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x * width
                y = landmark.y * height
                v = landmark.visibility
                keypoints.extend([x, y, v])
            
            return keypoints, results
        
        return None, None
    
    def _detect_movenet(self, frame):
        """MoveNet pose detection podle oficiální dokumentace"""
        height, width = frame.shape[:2]
        
        # Inicializace crop region pro první frame
        if self.crop_region is None:
            self.crop_region = self._init_crop_region(height, width)
        
        try:
            # Ořezání a změna velikosti podle crop region
            input_image = self._crop_and_resize(
                tf.expand_dims(frame, axis=0), 
                self.crop_region, 
                [self.input_size, self.input_size]
            )
            
            # Převod na správný formát pro MoveNet
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # Inference
            outputs = self.movenet(input_image)
            keypoints = outputs['output_0'].numpy()
            
            if keypoints is not None and len(keypoints) > 0:
                # MoveNet vrací keypoints ve formátu [y, x, confidence] normalizované 0-1
                pose_keypoints = keypoints[0, 0, :, :]  # [17, 3] array
                
                # Aktualizace koordinátů podle crop region
                self._update_keypoints_coordinates(pose_keypoints, height, width)
                
                # Určení nové crop region pro další frame
                self.crop_region = self._determine_crop_region(keypoints, height, width)
                
                # Převod na MediaPipe formát
                mediapipe_keypoints = self._convert_movenet_to_mediapipe_format(pose_keypoints, width, height)
                
                return mediapipe_keypoints, pose_keypoints
            
        except Exception as e:
            print(f"Chyba v MoveNet detekci: {e}")
            # Reset crop region při chybě
            self.crop_region = self._init_crop_region(height, width)
        
        return None, None
    
    def _detect_openpose(self, frame):
        """OpenPose detection"""
        height, width = frame.shape[:2]
        
        # OpenPose detection
        datum = op.Datum()
        datum.cvInputData = frame
        self.detector.emplaceAndPop(op.VectorDatum([datum]))
        
        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            # OpenPose vrací keypoints ve formátu [x, y, confidence]
            pose_keypoints = datum.poseKeypoints[0]  # První osoba
            
            # Převod na formát kompatibilní s MediaPipe
            # OpenPose BODY_25 -> MediaPipe mapping
            keypoints = self._convert_openpose_to_mediapipe_format(pose_keypoints, width, height)
            
            return keypoints, datum
        
        return None, None
    
    def _convert_movenet_to_mediapipe_format(self, movenet_keypoints, width, height):
        """Převádí MoveNet keypoints na MediaPipe formát"""
        # MoveNet má 17 bodů, MediaPipe má 33 bodů
        # Vytvoříme prázdný array pro MediaPipe formát
        mediapipe_keypoints = [0.0] * (33 * 3)  # 33 bodů × 3 (x, y, visibility)
        
        # MoveNet COCO keypoints -> MediaPipe pose landmarks mapping
        # MoveNet pořadí: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, 
        # right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, 
        # right_hip, left_knee, right_knee, left_ankle, right_ankle
        mapping = {
            0: 0,   # nose -> nose
            1: 2,   # left_eye -> left_eye 
            2: 5,   # right_eye -> right_eye
            3: 7,   # left_ear -> left_ear
            4: 8,   # right_ear -> right_ear
            5: 11,  # left_shoulder -> left_shoulder
            6: 12,  # right_shoulder -> right_shoulder
            7: 13,  # left_elbow -> left_elbow
            8: 14,  # right_elbow -> right_elbow
            9: 15,  # left_wrist -> left_wrist
            10: 16, # right_wrist -> right_wrist
            11: 23, # left_hip -> left_hip
            12: 24, # right_hip -> right_hip
            13: 25, # left_knee -> left_knee
            14: 26, # right_knee -> right_knee
            15: 27, # left_ankle -> left_ankle
            16: 28, # right_ankle -> right_ankle
        }
        
        for movenet_idx, mediapipe_idx in mapping.items():
            if movenet_idx < len(movenet_keypoints):
                y_norm, x_norm, confidence = movenet_keypoints[movenet_idx]
                
                if confidence > MIN_CROP_KEYPOINT_SCORE:  # Použití oficiálního threshold
                    # Převod normalizovaných souřadnic na pixely
                    x = x_norm * width
                    y = y_norm * height
                    
                    base_idx = mediapipe_idx * 3
                    mediapipe_keypoints[base_idx] = x
                    mediapipe_keypoints[base_idx + 1] = y
                    mediapipe_keypoints[base_idx + 2] = confidence
        
        return mediapipe_keypoints
    
    def _convert_openpose_to_mediapipe_format(self, openpose_keypoints, width, height):
        """Převádí OpenPose keypoints na MediaPipe formát"""
        # OpenPose BODY_25 model mapping na MediaPipe
        # Toto je zjednodušené mapování - můžete ho rozšířit
        
        # MediaPipe má 33 bodů, OpenPose BODY_25 má 25 bodů
        # Vytvoříme prázdný array pro MediaPipe formát
        mediapipe_keypoints = [0.0] * (33 * 3)  # 33 bodů × 3 (x, y, visibility)
        
        # Mapování klíčových bodů (zjednodušené)
        # OpenPose BODY_25 -> MediaPipe pose landmarks
        mapping = {
            0: 0,   # Nose -> Nose
            1: 2,   # Neck -> Right Eye
            2: 12,  # RShoulder -> Right Shoulder  
            3: 14,  # RElbow -> Right Elbow
            4: 16,  # RWrist -> Right Wrist
            5: 11,  # LShoulder -> Left Shoulder
            6: 13,  # LElbow -> Left Elbow
            7: 15,  # LWrist -> Left Wrist
            8: 24,  # MidHip -> Right Hip (aproximace)
            9: 26,  # RHip -> Right Hip
            10: 28, # RKnee -> Right Knee
            11: 30, # RAnkle -> Right Ankle
            12: 23, # LHip -> Left Hip
            13: 25, # LKnee -> Left Knee
            14: 27, # LAnkle -> Left Ankle
            # Další body lze mapovat podle potřeby
        }
        
        for openpose_idx, mediapipe_idx in mapping.items():
            if openpose_idx < len(openpose_keypoints):
                x, y, confidence = openpose_keypoints[openpose_idx]
                
                if confidence > 0.3:  # Práh confidence
                    base_idx = mediapipe_idx * 3
                    mediapipe_keypoints[base_idx] = x
                    mediapipe_keypoints[base_idx + 1] = y
                    mediapipe_keypoints[base_idx + 2] = confidence
        
        return mediapipe_keypoints
    
    def draw_landmarks(self, frame, detection_result):
        """Vykreslí pose landmarks do snímku"""
        if self.detector_type == "mediapipe" and detection_result is not None:
            self.mp_drawing.draw_landmarks(
                frame,
                detection_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        elif self.detector_type in ["movenet", "movenet_lightning", "movenet_thunder"] and detection_result is not None:
            # Jednoduché vykreslení MoveNet keypoints
            self._draw_movenet_keypoints(frame, detection_result)
        elif self.detector_type == "openpose" and detection_result is not None:
            # OpenPose už má pose vykreslené v detection_result.cvOutputData
            if hasattr(detection_result, 'cvOutputData') and detection_result.cvOutputData is not None:
                frame[:] = detection_result.cvOutputData
    
    def _draw_movenet_keypoints(self, frame, keypoints):
        """Vykreslí MoveNet keypoints do snímku"""
        height, width = frame.shape[:2]
        
        # MoveNet connections (podobné COCO formátu)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms  
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Vykreslení bodů
        for i, (y_norm, x_norm, confidence) in enumerate(keypoints):
            if confidence > 0.3:
                x = int(x_norm * width)
                y = int(y_norm * height)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        # Vykreslení spojnic
        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                
                start_x = int(keypoints[start_idx][1] * width)
                start_y = int(keypoints[start_idx][0] * height)
                end_x = int(keypoints[end_idx][1] * width)
                end_y = int(keypoints[end_idx][0] * height)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    
    def _init_crop_region(self, image_height, image_width):
        """Definuje výchozí crop region podle oficiální dokumentace"""
        if image_width > image_height:
            box_height = image_width / image_height
            box_width = 1.0
            y_min = (image_height / 2 - image_width / 2) / image_height
            x_min = 0.0
        else:
            box_height = 1.0
            box_width = image_height / image_width
            y_min = 0.0
            x_min = (image_width / 2 - image_height / 2) / image_width
        
        return {
            'y_min': y_min,
            'x_min': x_min,
            'y_max': y_min + box_height,
            'x_max': x_min + box_width,
            'height': box_height,
            'width': box_width
        }
    
    def _torso_visible(self, keypoints):
        """Kontroluje, zda jsou viditelné dostatečné torso keypoints"""
        return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] > MIN_CROP_KEYPOINT_SCORE or
                keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] > MIN_CROP_KEYPOINT_SCORE) and
               (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE or
                keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE))
    
    def _determine_crop_region(self, keypoints, image_height, image_width):
        """Určuje region pro ořezání podle oficiální dokumentace"""
        target_keypoints = {}
        for joint in KEYPOINT_DICT.keys():
            target_keypoints[joint] = [
                keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
                keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
            ]
        
        if self._torso_visible(keypoints):
            center_y = (target_keypoints['left_hip'][0] + target_keypoints['right_hip'][0]) / 2
            center_x = (target_keypoints['left_hip'][1] + target_keypoints['right_hip'][1]) / 2
            
            # Výpočet range pro torso a celé tělo
            torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            max_torso_yrange = max_torso_xrange = 0.0
            
            for joint in torso_joints:
                dist_y = abs(center_y - target_keypoints[joint][0])
                dist_x = abs(center_x - target_keypoints[joint][1])
                max_torso_yrange = max(max_torso_yrange, dist_y)
                max_torso_xrange = max(max_torso_xrange, dist_x)
            
            max_body_yrange = max_body_xrange = 0.0
            for joint in KEYPOINT_DICT.keys():
                if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
                    continue
                dist_y = abs(center_y - target_keypoints[joint][0])
                dist_x = abs(center_x - target_keypoints[joint][1])
                max_body_yrange = max(max_body_yrange, dist_y)
                max_body_xrange = max(max_body_xrange, dist_x)
            
            crop_length_half = max([
                max_torso_xrange * 1.9, max_torso_yrange * 1.9,
                max_body_yrange * 1.2, max_body_xrange * 1.2
            ])
            
            tmp = np.array([center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = min(crop_length_half, np.max(tmp))
            
            if crop_length_half > max(image_width, image_height) / 2:
                return self._init_crop_region(image_height, image_width)
            
            crop_length = crop_length_half * 2
            crop_corner = [center_y - crop_length_half, center_x - crop_length_half]
            
            return {
                'y_min': crop_corner[0] / image_height,
                'x_min': crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': crop_length / image_height,
                'width': crop_length / image_width
            }
        else:
            return self._init_crop_region(image_height, image_width)
    
    def _crop_and_resize(self, image, crop_region, crop_size):
        """Ořeže a změní velikost obrázku podle oficiální dokumentace"""
        boxes = [[crop_region['y_min'], crop_region['x_min'],
                 crop_region['y_max'], crop_region['x_max']]]
        output_image = tf.image.crop_and_resize(
            image, box_indices=[0], boxes=boxes, crop_size=crop_size)
        return output_image
    
    def _update_keypoints_coordinates(self, keypoints, image_height, image_width):
        """Aktualizuje koordináty keypoints podle crop region"""
        for idx in range(17):
            keypoints[idx, 0] = (
                self.crop_region['y_min'] * image_height +
                self.crop_region['height'] * image_height * keypoints[idx, 0]
            ) / image_height
            keypoints[idx, 1] = (
                self.crop_region['x_min'] * image_width +
                self.crop_region['width'] * image_width * keypoints[idx, 1]
            ) / image_width
    
    def close(self):
        """Uzavře detektor"""
        if self.detector_type == "mediapipe" and self.detector:
            self.detector.close()
        elif self.detector_type in ["movenet", "movenet_lightning", "movenet_thunder"]:
            # Reset crop region při zavření
            self.crop_region = None
        elif self.detector_type == "openpose" and self.detector:
            self.detector.stop()


def get_available_detectors():
    """Vrací seznam dostupných detektorů"""
    detectors = []
    if MEDIAPIPE_AVAILABLE:
        detectors.append("mediapipe")
    if MOVENET_AVAILABLE:
        detectors.extend(["movenet_lightning", "movenet_thunder"])
    if OPENPOSE_AVAILABLE:
        detectors.append("openpose")
    return detectors


def select_detector():
    """Interaktivní výběr detektoru"""
    available = get_available_detectors()
    
    if not available:
        print("❌ Žádný pose detektor není k dispozici!")
        print("Nainstalujte MediaPipe: pip install mediapipe")
        print("Nebo nainstalujte OpenPose podle oficiální dokumentace")
        return None
    
    print("🎯 Dostupné pose detektory:")
    for i, detector in enumerate(available, 1):
        print(f"  {i}. {detector.upper()}")
    
    while True:
        try:
            choice = input(f"Vyberte detektor (1-{len(available)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    return available[idx]
            print("❌ Neplatná volba, zkuste znovu.")
        except KeyboardInterrupt:
            print("\n🚫 Zrušeno uživatelem")
            return None


if __name__ == "__main__":
    # Test dostupnosti detektorů
    print("🔍 Kontrola dostupných pose detektorů...")
    detectors = get_available_detectors()
    
    if detectors:
        print(f"✅ Dostupné detektory: {', '.join(detectors)}")
        
        # Test inicializace
        for detector_name in detectors:
            try:
                detector = PoseDetector(detector_name)
                print(f"✅ {detector_name.upper()} úspěšně inicializován")
                detector.close()
            except Exception as e:
                print(f"❌ {detector_name.upper()} selhal: {e}")
    else:
        print("❌ Žádné pose detektory nejsou k dispozici")