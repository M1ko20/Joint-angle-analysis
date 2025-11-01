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
    
    # MoveNet konstanty
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
    
    # Confidence score MOVENET!! - Zvýšený threshold pro vyšší přesnost
    MIN_CROP_KEYPOINT_SCORE = 0.65
    
except ImportError:
    MOVENET_AVAILABLE = False
    print("MoveNet není k dispozici. Nainstalujte: pip install tensorflow tensorflow-hub")

try:
    # OpenPose
    sys.path.append('/usr/local/python') #url
    from openpose import pyopenpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    print("OpenPose není k dispozici. Zkontrolujte instalaci OpenPose.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    
    # YOLO11 COCO pose keypoints mapping
    YOLO_KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO není k dispozici. Nainstalujte: pip install ultralytics")

try:
    import torch
    import torch.nn as nn
    VITPOSE_AVAILABLE = True
    
    # ViTPose COCO pose keypoints mapping (stejné jako YOLO)
    VITPOSE_KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
except ImportError:
    VITPOSE_AVAILABLE = False
    print("ViTPose není k dispozici. Nainstalujte: pip install torch torchvision")


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
        elif self.detector_type in ["yolo11n", "yolo11x", "yolo"]:
            if not YOLO_AVAILABLE:
                raise ImportError("YOLO není k dispozici")
            self._init_yolo()
        elif self.detector_type == "vitpose":
            if not VITPOSE_AVAILABLE:
                raise ImportError("ViTPose není k dispozici")
            self._init_vitpose()
        else:
            raise ValueError(f"Neznámý typ detektoru: {self.detector_type}")
    
    def _init_mediapipe(self):
        """Inicializuje MediaPipe"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.detector = self.mp_pose.Pose( #nastaveni ktere by bylo fajn se kouknout
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5, #KONFIDENCE!
            min_tracking_confidence=0.5
        )
        print("MediaPipe inicializován")
    
    def _init_movenet(self):
        """Inicializuje MoveNet"""
        # thunder
        if self.detector_type == "movenet_thunder":
            model_name = "thunder"
            self.input_size = 256
        else:  # lightning
            model_name = "lightning"
            self.input_size = 192
        
        model_url = f"https://tfhub.dev/google/movenet/singlepose/{model_name}/4"
        
        try:
            self.detector = hub.load(model_url)
            self.movenet = self.detector.signatures['serving_default']
            self.crop_region = None
            print(f"MoveNet inicializován ({model_name})")
        except Exception as e:
            print(f"Chyba při načítání MoveNet modelu: {e}")
            raise
    
    def _init_openpose(self):
        """Inicializuje OpenPose"""
        params = dict()
        params["model_folder"] = "/usr/local/share/OpenPose/models/" #open pose url
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
    
    def _init_yolo(self):
        """Inicializuje YOLO11 pose model"""
        if self.detector_type == "yolo11x":
            model_name = "yolo11x-pose.pt"
        elif self.detector_type == "yolo11n":
            model_name = "yolo11n-pose.pt"
        else: 
            model_name = "yolo11n-pose.pt"
        
        try:
            # Hledej model v Analysis adresáři (kde je pose_detector.py)
            analysis_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(analysis_dir, model_name)
            
            # Zkus z app adresáře (fallback)
            if not os.path.exists(model_path):
                app_dir = os.path.join(analysis_dir, "app")
                model_path = os.path.join(app_dir, model_name)
            
            # Zkus z aktuální složky jako poslední fallback
            if not os.path.exists(model_path):
                model_path = model_name
            
            print(f"YOLO model hledám: {model_path}")
            print(f"Existuje: {os.path.exists(model_path)}")
            
            # Načti YOLO model
            self.detector = YOLO(model_path)
            
            # Konfigurace
            self.yolo_conf_threshold = 0.9  
            self.yolo_iou_threshold = 0.9   
            
            print(f"YOLO11 inicializován ({model_path})")
        except Exception as e:
            print(f"Chyba při načítání YOLO11 modelu: {e}")
            raise
    
    def _init_vitpose(self):
        """Inicializuje ViTPose model - pure PyTorch bez mmpose (Mac compatible)"""
        model_name = "vitpose-l.pth"
        
        try:
            # Hledej model v Analysis adresáři
            analysis_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(analysis_dir, model_name)
            
            # Zkus z app adresáři
            if not os.path.exists(model_path):
                app_dir = os.path.join(analysis_dir, "app")
                model_path = os.path.join(app_dir, model_name)
            
            # Zkus z aktuální složku
            if not os.path.exists(model_path):
                model_path = model_name
            
            print(f"ViTPose model hledám: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ViTPose model nebyl nalezen: {model_path}")
            
            # Načti checkpoint - podobně jako YOLO načítá model
            print(f"Načítám ViTPose checkpoint z: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # ViTPose používá YOLO11-like API, takže použijeme podobný přístup
            # Pro teď jen načteme checkpoint a připravíme ho pro budoucí použití
            self.detector = {
                'checkpoint': checkpoint,
                'model': None,  # Model by vyžadoval mmpose build
                'ready': False
            }
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.vitpose_device = device
            self.vitpose_input_size = (192, 256)  # width, height
            
            print(f"✓ ViTPose checkpoint načten ({model_path})")
            print(f"  Zařízení: {device}")
            print(f"  ⚠️  Poznámka: ViTPose vyžaduje mmpose framework pro plnou funkčnost")
            print(f"  ⚠️  Model je připraven, ale inference není k dispozici na Macu bez kompilace")
            
        except Exception as e:
            print(f"✗ Chyba při inicializaci ViTPose: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def detect_pose(self, frame):
        """Detekuje pose v rámci a vrací normalizované keypoints"""
        if self.detector_type == "mediapipe":
            return self._detect_mediapipe(frame)
        elif self.detector_type in ["movenet", "movenet_lightning", "movenet_thunder"]:
            return self._detect_movenet(frame)
        elif self.detector_type == "openpose":
            return self._detect_openpose(frame)
        elif self.detector_type in ["yolo11n", "yolo11x", "yolo"]:
            return self._detect_yolo(frame)
        elif self.detector_type == "vitpose":
            return self._detect_vitpose(frame)
    
    def _detect_mediapipe(self, frame):
        """MediaPipe pose detection"""
        height, width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Zkus s numpy array (obchází protobuf problém)
            import mediapipe as mp
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.detector.detect(image)
            
            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks[0]:
                    x = landmark.x * width
                    y = landmark.y * height
                    z = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    keypoints.extend([x, y, z])
                
                return np.array(keypoints), results
        except:
            # Fallback na starší API
            try:
                results = self.detector.process(image_rgb)
                
                if results.pose_landmarks:
                    keypoints = []
                    for landmark in results.pose_landmarks.landmark:
                        x = landmark.x * width
                        y = landmark.y * height
                        v = landmark.visibility
                        keypoints.extend([x, y, v])
                    
                    return np.array(keypoints), results
            except Exception as e:
                print(f"MediaPipe detection error: {e}")
        
        return None, None
    
    def _detect_movenet(self, frame):
        """MoveNet pose detection podle oficiální dokumentace"""
        height, width = frame.shape[:2]
        
        #Crop pro 1. frame
        if self.crop_region is None:
            self.crop_region = self._init_crop_region(height, width)
        
        try:
            input_image = self._crop_and_resize(
                tf.expand_dims(frame, axis=0), 
                self.crop_region, 
                [self.input_size, self.input_size]
            )
            input_image = tf.cast(input_image, dtype=tf.int32)
            
            # Inference
            outputs = self.movenet(input_image)
            keypoints = outputs['output_0'].numpy()
            
            if keypoints is not None and len(keypoints) > 0:
                pose_keypoints = keypoints[0, 0, :, :]  # [17, 3] array
            
                self._update_keypoints_coordinates(pose_keypoints, height, width)
                
                self.crop_region = self._determine_crop_region(keypoints, height, width)
                mediapipe_keypoints = self._convert_movenet_to_mediapipe_format(pose_keypoints, width, height)
                
                return mediapipe_keypoints, pose_keypoints
            
        except Exception as e:
            print(f"Chyba v MoveNet detekci: {e}")
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
            # OpenPose  [x, y, confidence]
            pose_keypoints = datum.poseKeypoints[0]  
            
            keypoints = self._convert_openpose_to_mediapipe_format(pose_keypoints, width, height)
            
            return keypoints, datum
        
        return None, None
    
    def _detect_yolo(self, frame):
        """YOLO11 pose detection"""
        height, width = frame.shape[:2]
        
        try:
            # Inference s YOLO11
            results = self.detector(frame, 
                                  conf=self.yolo_conf_threshold,
                                  iou=self.yolo_iou_threshold,
                                  verbose=False)  # Potlačení výstupu
            
            if results and len(results) > 0:
                result = results[0]  

                if result.keypoints is not None and len(result.keypoints.data) > 0:
                    best_detection_idx = 0
                    if len(result.boxes) > 1:
                        best_conf = 0
                        for i, box in enumerate(result.boxes):
                            if box.conf[0] > best_conf:
                                best_conf = box.conf[0]
                                best_detection_idx = i
                    
                    keypoints_data = result.keypoints.data[best_detection_idx]  # [17, 3] tensor
                    
                    if hasattr(keypoints_data, 'cpu'):
                        keypoints_array = keypoints_data.cpu().numpy()
                    else:
                        keypoints_array = keypoints_data.numpy() if hasattr(keypoints_data, 'numpy') else keypoints_data
                    
                    mediapipe_keypoints = self._convert_yolo_to_mediapipe_format(keypoints_array, width, height)
                    
                    return mediapipe_keypoints, result
                    
        except Exception as e:
            print(f"Chyba v YOLO11 detekci: {e}")
        
        return None, None
    
    def _detect_vitpose(self, frame):
        """ViTPose pose detection - Mac compatible placeholder"""
        height, width = frame.shape[:2]
        
        # ViTPose na Macu vyžaduje kompilovaný mmpose s C++ extensions
        # Pro plnou funkcionalitu je potřeba:
        # 1. Nainstalovat mmcv-full (vyžaduje kompilaci)
        # 2. Nainstalovat mmpose
        # 3. Build C++ extensions
        
        print("[ViTPose] ⚠️  Model není k dispozici na Macu bez kompilace mmpose")
        print("[ViTPose] Použijte jiný detektor (mediapipe, movenet, yolo)")
        
        return None, None
    
    def _heatmap_to_keypoints(self, heatmaps, img_w, img_h):
        """Konvertuje heatmapy na keypoints souřadnice"""
        num_joints = heatmaps.shape[0]
        keypoints = np.zeros((num_joints, 3))  # x, y, confidence
        
        for i in range(num_joints):
            heatmap = heatmaps[i]
            # Najdi maximum v heatmapě
            idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            keypoints[i, 1] = idx[0] * img_h / heatmap.shape[0]  # y
            keypoints[i, 0] = idx[1] * img_w / heatmap.shape[1]  # x
            keypoints[i, 2] = heatmap[idx]  # confidence
        
        return keypoints
    
    def _convert_vitpose_to_mediapipe_format(self, vitpose_keypoints, width, height):
        """Převádí ViTPose (COCO 17) keypoints na MediaPipe formát (33 bodů)"""
        # ViTPose má stejné keypoints jako YOLO (COCO format)
        # Použijeme stejnou konverzi
        return self._convert_yolo_to_mediapipe_format(vitpose_keypoints, width, height)
    
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
                
                if confidence > 0.5:  # Zvýšený threshold (bylo 0.3)
                    base_idx = mediapipe_idx * 3
                    mediapipe_keypoints[base_idx] = x
                    mediapipe_keypoints[base_idx + 1] = y
                    mediapipe_keypoints[base_idx + 2] = confidence
        
        return mediapipe_keypoints
    
    def _convert_yolo_to_mediapipe_format(self, yolo_keypoints, width, height):
        """Převádí YOLO11 keypoints na MediaPipe formát"""
        # YOLO11 má 17 bodů, MediaPipe má 33 bodů
        # Vytvoříme prázdný array pro MediaPipe formát
        mediapipe_keypoints = [0.0] * (33 * 3)  # 33 bodů × 3 (x, y, visibility)
        
        # YOLO11 COCO keypoints -> MediaPipe pose landmarks mapping
        # YOLO11 pořadí: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, 
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
        
        for yolo_idx, mediapipe_idx in mapping.items():
            if yolo_idx < len(yolo_keypoints):
                x, y, confidence = yolo_keypoints[yolo_idx]
                
                # YOLO11 keypoints jsou již v pixelech, nekontrolováno normalizované
                # Zkontrolujeme, zda jsou normalizované (0-1) nebo v pixelech
                if x <= 1.0 and y <= 1.0:
                    # Normalizované souřadnice -> převod na pixely
                    x = x * width
                    y = y * height
                
                if confidence > 0.6:  # Zvýšený threshold (bylo 0.3)
                    base_idx = mediapipe_idx * 3
                    mediapipe_keypoints[base_idx] = x
                    mediapipe_keypoints[base_idx + 1] = y
                    mediapipe_keypoints[base_idx + 2] = confidence
        
        return mediapipe_keypoints
    
    def draw_landmarks(self, frame, detection_result):
        """Vykresli pose landmarks do snímku"""
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
        elif self.detector_type in ["yolo11n", "yolo11x", "yolo"] and detection_result is not None:
            # YOLO11 vykreslení
            self._draw_yolo_keypoints(frame, detection_result)
    
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
    
    def _draw_yolo_keypoints(self, frame, yolo_result):
        """Vykresli YOLO11 keypoints do snímku"""
        try:
            if yolo_result.keypoints is not None and len(yolo_result.keypoints.data) > 0:
                # Najít nejlepší detekci
                best_detection_idx = 0
                if len(yolo_result.boxes) > 1:
                    best_conf = 0
                    for i, box in enumerate(yolo_result.boxes):
                        if box.conf[0] > best_conf:
                            best_conf = box.conf[0]
                            best_detection_idx = i
                
                keypoints_data = yolo_result.keypoints.data[best_detection_idx]
                
                if hasattr(keypoints_data, 'cpu'):
                    keypoints = keypoints_data.cpu().numpy()
                else:
                    keypoints = keypoints_data.numpy() if hasattr(keypoints_data, 'numpy') else keypoints_data
                
                height, width = frame.shape[:2]
                
                # YOLO11 COCO connections (podobné MoveNet)
                connections = [
                    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms  
                    (5, 11), (6, 12), (11, 12),  # Torso
                    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
                ]
                
                # Vykreslení bodů
                for i, (x, y, confidence) in enumerate(keypoints):
                    if confidence > 0.3:
                        # Zkontrolovat, zda jsou souřadnice normalizované
                        if x <= 1.0 and y <= 1.0:
                            x = int(x * width)
                            y = int(y * height)
                        else:
                            x = int(x)
                            y = int(y)
                        
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                        
                        # Přidání textu s názvem bodu (volitelné)
                        if i < len(YOLO_KEYPOINT_NAMES):
                            cv2.putText(frame, YOLO_KEYPOINT_NAMES[i][:3], (x+5, y-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Vykreslení spojnic
                for start_idx, end_idx in connections:
                    if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                        keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                        
                        # Start point
                        start_x, start_y = keypoints[start_idx][:2]
                        if start_x <= 1.0 and start_y <= 1.0:
                            start_x = int(start_x * width)
                            start_y = int(start_y * height)
                        else:
                            start_x = int(start_x)
                            start_y = int(start_y)
                        
                        # End point
                        end_x, end_y = keypoints[end_idx][:2]
                        if end_x <= 1.0 and end_y <= 1.0:
                            end_x = int(end_x * width)
                            end_y = int(end_y * height)
                        else:
                            end_x = int(end_x)
                            end_y = int(end_y)
                        
                        cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
                        
        except Exception as e:
            print(f"Chyba při vykreslování YOLO keypoints: {e}")
    
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
        elif self.detector_type in ["yolo11n", "yolo11x", "yolo"]:
            # YOLO modely nevyžadují speciální uzavření
            self.detector = None


def get_available_detectors():
    """Vrací seznam dostupných detektorů"""
    detectors = []
    if MEDIAPIPE_AVAILABLE:
        detectors.append("mediapipe")
    if MOVENET_AVAILABLE:
        detectors.extend(["movenet_lightning", "movenet_thunder"])
    if OPENPOSE_AVAILABLE:
        detectors.append("openpose")
    if YOLO_AVAILABLE:
        detectors.extend(["yolo11n", "yolo11x"])
    if VITPOSE_AVAILABLE:
        detectors.append("vitpose")
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