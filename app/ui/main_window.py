"""
Hlavní okno PyQt6 aplikace pro analýzu pohybu
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QScrollArea, QGroupBox, QGridLayout, QFileDialog,
    QProgressBar, QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QFrame
import cv2
import numpy as np
import math
from pathlib import Path
from core.analyzer import AnalysisWorker

JOINT_NAMES_CZ = {
    'left_elbow': 'Levý loket',
    'right_elbow': 'Pravý loket',
    'left_shoulder': 'Levé rameno',
    'right_shoulder': 'Pravé rameno',
    'left_knee': 'Levé koleno',
    'right_knee': 'Pravé koleno',
    'left_hip': 'Levý bok',
    'right_hip': 'Pravý bok',
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analýza Pohybu - Pose Detection")
        self.setGeometry(100, 100, 1400, 900)
        
        # Proměnné
        self.video_path = None
        self.video_capture = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.frames_per_second = 30
        self.analysis_results = None
        self.analysis_worker = None
        self.analysis_thread = None
        self.video_rotation = 0 
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        
        left_panel = self._create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        center_panel = self._create_video_preview()
        main_layout.addWidget(center_panel, 2)
        
        right_panel = self._create_info_panel()
        main_layout.addWidget(right_panel, 1)
        
    def _create_control_panel(self):
        """Vytvoří levý ovládací panel"""
        panel = QGroupBox("Ovládání")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Video:"))
        video_layout = QHBoxLayout()
        self.video_label = QLabel("Nevybráno")
        video_layout.addWidget(self.video_label)
        self.video_btn = QPushButton("Otevřít video")
        self.video_btn.clicked.connect(self._select_video)
        video_layout.addWidget(self.video_btn)
        layout.addLayout(video_layout)
        
        layout.addWidget(QLabel("Model detekce:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "MediaPipe - Image",
            "MediaPipe - Video",
            "MoveNet Lightning - Image",
            "MoveNet Lightning - Video",
            "MoveNet Thunder - Image",
            "MoveNet Thunder - Video",
            "YOLO11n - Image",
            "YOLO11x - Image",
            "ViTPose - Image"
        ])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(self.model_combo)
        
        # Info o režimu
        self.mode_info_label = QLabel()
        self.mode_info_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        self.mode_info_label.setWordWrap(True)
        layout.addWidget(self.mode_info_label)
        
        layout.addWidget(QLabel("Měřené klouby:"))
        joints_group = self._create_joints_checkboxes()
        layout.addWidget(joints_group)
        
        layout.addWidget(QLabel("Parametry:"))
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Confidence threshold:"), 0, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setSingleStep(0.05)
        params_layout.addWidget(self.confidence_spin, 0, 1)
        
        # Smooth factor (pouze pro Video režim)
        self.smooth_label = QLabel("Smooth factor:")
        params_layout.addWidget(self.smooth_label, 1, 0)
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 1.0)
        self.smooth_spin.setValue(0.3)
        self.smooth_spin.setSingleStep(0.1)
        self.smooth_spin.setToolTip("0.0 = bez vyhlazování, 1.0 = maximální vyhlazování")
        params_layout.addWidget(self.smooth_spin, 1, 1)
        
        layout.addLayout(params_layout)
        
        # Aktualizuj mode info (teď už existují všechny widgety)
        self._update_mode_info()
        
        self.analyze_btn = QPushButton("Spustit analýzu")
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.analyze_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        layout.addWidget(self.analyze_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        export_layout = QHBoxLayout()
        self.export_json_btn = QPushButton("JSON")
        self.export_json_btn.clicked.connect(self._export_json)
        self.export_json_btn.setEnabled(False)
        self.export_json_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover:!pressed {
                background-color: #0052a3;
            }
            QPushButton:pressed {
                background-color: #004080;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        export_layout.addWidget(self.export_json_btn)
        
        self.export_xml_btn = QPushButton("XML")
        self.export_xml_btn.clicked.connect(self._export_xml)
        self.export_xml_btn.setEnabled(False)
        self.export_xml_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover:!pressed {
                background-color: #0052a3;
            }
            QPushButton:pressed {
                background-color: #004080;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        export_layout.addWidget(self.export_xml_btn)
        layout.addLayout(export_layout)
        
        self.graph_btn = QPushButton("Zobrazit grafy")
        self.graph_btn.clicked.connect(self._show_graphs)
        self.graph_btn.setEnabled(False)
        self.graph_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6600;
                color: white;
                font-weight: bold;
                padding: 12px;
                border: none;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover:!pressed {
                background-color: #ff8c00;
            }
            QPushButton:pressed {
                background-color: #e55a00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
        """)
        self.graph_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.graph_btn)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def _create_joints_checkboxes(self):
        """Vytvoří checkboxy pro výběr kloubů"""
        group = QGroupBox()
        layout = QGridLayout()
        
        self.joint_checks = {}
        joints = [
            ("Levý loket", "left_elbow"),
            ("Pravý loket", "right_elbow"),
            ("Levé rameno", "left_shoulder"),
            ("Pravé rameno", "right_shoulder"),
            ("Levé koleno", "left_knee"),
            ("Pravé koleno", "right_knee"),
            ("Levý bok", "left_hip"),
            ("Pravý bok", "right_hip"),
        ]
        
        for i, (label, key) in enumerate(joints):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            self.joint_checks[key] = checkbox
            layout.addWidget(checkbox, i // 2, i % 2)
        
        group.setLayout(layout)
        return group
    
    def _create_video_preview(self):
        """Vytvoří náhled videa"""
        panel = QGroupBox("Náhled videa")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        layout.addStretch()
        
        # Label pro video - vhodná velikost
        self.video_display = QLabel()
        self.video_display.setMinimumSize(600, 450)
        self.video_display.setMaximumSize(800, 600)
        self.video_display.setStyleSheet("""
            background-color: #1a1a1a; 
            border: 3px solid #0066cc;
            border-radius: 8px;
            padding: 5px;
        """)
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setScaledContents(False)
        layout.addWidget(self.video_display, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addStretch()
        
        controls_layout = QHBoxLayout()
        controls_layout.addStretch()
        
        self.frame_slider = QSpinBox()
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setMaximumWidth(120)
        self.frame_slider.setStyleSheet("""
            QSpinBox {
                padding: 5px;
                border: 1px solid #0066cc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
        """)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        
        controls_layout.addWidget(QLabel("Frame:"))
        controls_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0/0")
        self.frame_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        controls_layout.addWidget(self.frame_label)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        panel.setLayout(layout)
        return panel
    
    def _create_info_panel(self):
        """Vytvoří pravý panel s informacemi"""
        panel = QGroupBox("Informace")
        layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setText("Žádná data k zobrazení")
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("""
            background-color: #ffffff;
            padding: 15px;
            border: none;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11px;
        """)
        self.info_text.setMinimumHeight(400)
        layout.addWidget(self.info_text)
        
        panel.setLayout(layout)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return panel
    
    def _select_video(self):
        """Otevře dialog pro výběr videa"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Vyberte video",
            "",
            "Video soubory (*.mp4 *.avi *.mov *.mkv);;Všechny soubory (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.video_label.setText(Path(file_path).name)
            
            self.video_capture = cv2.VideoCapture(file_path)
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frames_per_second = self.video_capture.get(cv2.CAP_PROP_FPS)
            
            self.frame_slider.setRange(0, self.total_frames - 1)
            self.frame_label.setText(f"0/{self.total_frames}")
            
            self.current_frame_index = 0
            self._display_frame(0)
            
            self.analyze_btn.setEnabled(True)
            
            self._detect_and_set_video_rotation()
    
    def _detect_and_set_video_rotation(self):
        """Detekuje orientaci videa z prvních snímků a nastaví rotaci"""
        from pose_detector import PoseDetector
        
        detector = PoseDetector('mediapipe')
        max_attempts = 10
        
        for frame_idx in range(min(max_attempts, self.total_frames)):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_capture.read()
            
            if not ret:
                continue
            
            keypoints, _ = detector.detect_pose(frame)
            
            if keypoints is not None and len(keypoints) > 0:
                if isinstance(keypoints, list):
                    keypoints = keypoints[0] if len(keypoints) > 0 else None
                
                if keypoints is not None:
                    rotation = self._calculate_rotation_from_keypoints(keypoints)
                    if rotation is not None:
                        self.video_rotation = rotation
                        print(f"Detekována orientace videa: {rotation}° (frame {frame_idx})")
                        break
        
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def _calculate_rotation_from_keypoints(self, keypoints):
        """
        Vypočítá potřebnou rotaci videa podle orientace těla
        Returns: 0, 90, 180, nebo 270 (stupně)
        """
        try:
            if isinstance(keypoints, list):
                keypoints = np.array(keypoints)
            
            if len(keypoints.shape) == 1:
                keypoints = keypoints.reshape(-1, 3)
            # TODO FIXME, tady to urcite musi byt jinak, nemuzeme tipovat
            # MediaPipe indexy: nose=0, left_hip=23, right_hip=24
            # MoveNet/YOLO mají jiné indexy, ale nos je většinou 0
            if len(keypoints) < 25:
                # Pravděpodobně MoveNet (17 bodů) nebo YOLO
                # nose=0, left_hip=11, right_hip=12
                nose_idx = 0
                left_hip_idx = 11
                right_hip_idx = 12
            else:
                # MediaPipe (33 bodů)
                nose_idx = 0
                left_hip_idx = 23
                right_hip_idx = 24
            
            if len(keypoints) <= max(nose_idx, left_hip_idx, right_hip_idx):
                return None
            
            nose = keypoints[nose_idx][:2]  
            left_hip = keypoints[left_hip_idx][:2]
            right_hip = keypoints[right_hip_idx][:2]
            
            # Zkontroluj confidence (pokud existuje)
            if keypoints.shape[1] > 2:
                nose_conf = keypoints[nose_idx][2]
                left_hip_conf = keypoints[left_hip_idx][2]
                right_hip_conf = keypoints[right_hip_idx][2]
                
                if nose_conf < 0.3 or left_hip_conf < 0.3 or right_hip_conf < 0.3: #TODO FIXME udelat lepsi confidence skore a ne hardcode asi 
                    return None
            
            hip_center = (left_hip + right_hip) / 2
            
            vector = nose - hip_center
            
            angle_rad = math.atan2(vector[1], vector[0])
            angle_deg = math.degrees(angle_rad)
            
            angle_deg = angle_deg % 360
            
            # Rotace:
            # - 0° (hlava nahoře) = úhel 90° ± 45° 
            # - 90° (hlava vpravo, leží vlevo) = úhel 0° ± 45°
            # - 180° (hlava dole, stojí na hlavě) = úhel 270° ± 45°
            # - 270° (hlava vlevo, leží vpravo) = úhel 180° ± 45°
            
            if 225 <= angle_deg < 315:  # Hlava nahoře (normální)
                return 0
            elif 315 <= angle_deg or angle_deg < 45:  # Hlava vpravo
                return 270  # Rotuj doleva
            elif 45 <= angle_deg < 135:  # Hlava dole
                return 180
            elif 135 <= angle_deg < 225:  # Hlava vlevo
                return 90  # Rotuj doprava
            else:
                return 0
                
        except Exception as e:
            print(f"⚠️ Chyba při výpočtu rotace: {e}")
            return None
    
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
    
    def _display_frame(self, frame_index):
        """Zobrazí frame ze studeného videa"""
        if not hasattr(self, 'video_capture') or self.video_capture is None:
            return
        
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.video_capture.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if hasattr(self, 'video_rotation') and self.video_rotation != 0:
                frame_rgb = self._rotate_frame(frame_rgb, self.video_rotation)
            
            if hasattr(self, 'analysis_results') and self.analysis_results and 'keypoints' in self.analysis_results:
                keypoints_data = self.analysis_results.get('keypoints', [])
                for kp_entry in keypoints_data:
                    if kp_entry['frame'] == frame_index:
                        keypoints = np.array(kp_entry['keypoints'])
                        frame_rgb = self._draw_keypoints_on_frame_display(frame_rgb, keypoints)
                        break
            
            # Zmenši frame aby se vešel do okna
            h, w = frame_rgb.shape[:2]
            scale = min(640 / w, 480 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            
            
            h, w, ch = frame_resized.shape
            bytes_per_line = 3 * w
            qt_image = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            self.video_display.setPixmap(pixmap)
            self.current_frame_index = frame_index
            self.frame_label.setText(f"{frame_index}/{self.total_frames}")
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_index)
            self.frame_slider.blockSignals(False)
    
    def _draw_keypoints_on_frame_display(self, frame_rgb, keypoints):
        """
        Kreslí keypoints na frame pro display - RGB formát
        Filtruje podle confidence thresholdu z UI
        """
        if keypoints is None:
            return frame_rgb
        
        annotated = frame_rgb.copy()
        
        if isinstance(keypoints, list):
            keypoints = np.array(keypoints)
        
        if len(keypoints.shape) == 1:
            keypoints = keypoints.reshape(-1, 3)
        
        # Použij threshold z UI
        confidence_threshold = self.confidence_spin.value()
        
        point_radius = 5
        line_thickness = 2
        
        for i, keypoint in enumerate(keypoints):
            try:
                x, y = int(keypoint[0]), int(keypoint[1])
                confidence = keypoint[2] if len(keypoint) > 2 else 1.0
                
                # Filtruj podle UI thresholdu - NEKRESLÍ keypoints pod threshold!
                if confidence >= confidence_threshold and 0 <= x < annotated.shape[1] and 0 <= y < annotated.shape[0]:
                    color = (int(255 * (1 - confidence)), int(255 * confidence), 0)
                    cv2.circle(annotated, (x, y), point_radius, color, -1)
            except (IndexError, ValueError):
                pass
        
        connections = [
            (11, 13), (13, 15),  # Pravá ruka
            (12, 14), (14, 16),  # Levá ruka
            (11, 12),            # Ramena
            (11, 23), (12, 24),  # Trup
            (23, 24),            # Bok
            (23, 25), (25, 27),  # Pravá noha
            (24, 26), (26, 28),  # Levá noha
        ]
        
        for start, end in connections:
            if start < len(keypoints) and end < len(keypoints):
                try:
                    start_kp = keypoints[start]
                    end_kp = keypoints[end]
                    
                    start_conf = start_kp[2] if len(start_kp) > 2 else 1.0
                    end_conf = end_kp[2] if len(end_kp) > 2 else 1.0
                    
                    # Filtruj podle UI thresholdu - NEKRESLÍ linky mezi body pod threshold!
                    if start_conf >= confidence_threshold and end_conf >= confidence_threshold:
                        start_pos = (int(start_kp[0]), int(start_kp[1]))
                        end_pos = (int(end_kp[0]), int(end_kp[1]))
                        
                        if (0 <= start_pos[0] < annotated.shape[1] and 0 <= start_pos[1] < annotated.shape[0] and
                            0 <= end_pos[0] < annotated.shape[1] and 0 <= end_pos[1] < annotated.shape[0]):
                            cv2.line(annotated, start_pos, end_pos, (255, 255, 0), line_thickness)  # Žlutá v RGB
                except (IndexError, ValueError):
                    pass
        
        return annotated
    
    def _on_frame_changed(self, value):
        """Obsluha změny frame slideru"""
        self._display_frame(value)
    
    def _start_analysis(self):
        """Spustí analýzu"""
        if self.video_path is None:
            QMessageBox.warning(self, "Chyba", "Nejprve si vyberte video!")
            return
        
        # Ukončit předchozí thread pokud existuje
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait(5000)  # Čekej max 5 sekund
        
        # Zakázat buttons během analýzy
        self.analyze_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        
        # Zobrazit progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Vytvoř worker thread
        selected_joints = [key for key, checkbox in self.joint_checks.items() if checkbox.isChecked()]
        model_name, mode = self._get_model_name()
        
        self.analysis_worker = AnalysisWorker(
            self.video_path,
            model_name,
            selected_joints,
            self.confidence_spin.value(),
            self.video_rotation,
            mode,  # Přidán parametr režimu
            self.smooth_spin.value()  # Přidán smooth factor
        )
        
        self.analysis_thread = QThread()
        self.analysis_worker.moveToThread(self.analysis_thread)
        
        self.analysis_worker.progress.connect(self._on_analysis_progress)
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)  # Quit thread po skončení
        
        self.analysis_thread.start()
    
    def _on_analysis_progress(self, progress, frame_data):
        """Obsluha progress signálu z analyzátoru"""
        self.progress_bar.setValue(progress)
    
    def _on_analysis_finished(self, results):
        """Obsluha finalizace analýzy"""
        # Kontrola na chyby
        if 'error' in results:
            QMessageBox.critical(self, "Chyba", f"Chyba během analýzy:\n{results['error']}")
        else:
            self.analysis_results = results
            
            # Aktivuj tlačítka exportu
            self.export_json_btn.setEnabled(True)
            self.export_xml_btn.setEnabled(True)
            self.graph_btn.setEnabled(True)
            
            # Aktualizuj informace
            self._update_info_panel()
        
        # Skryj progress bar
        self.progress_bar.setVisible(False)
        
        # Znovu aktivuj kontroly
        self.analyze_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
    
    def _update_info_panel(self):
        """Aktualizuje pravý panel s informacemi"""
        if self.analysis_results is None:
            return
        
        total_frames = self.analysis_results.get('total_frames', 0)
        
        # HTML formátování - vylepšené
        info_html = f"""
        <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; line-height: 1.8; color: #333;">
        <h2 style="color: #0066cc; margin: 0 0 15px 0; font-size: 18px; border-bottom: 3px solid #0066cc; padding-bottom: 10px;">
        VÝSLEDKY ANALÝZY</h2>
        <p style="font-size: 13px; color: #555; margin: 10px 0; background: #e8f4f8; padding: 10px; border-radius: 4px;">
        <b style="font-size: 14px;">Celkem snímků:</b> <span style="font-size: 14px; color: #0066cc;"><b>{total_frames}</b></span>
        </p>
        <hr style="border: none; border-top: 2px solid #e0e0e0; margin: 15px 0;">
        """
        
        # Zobraz statistiku pokud existuje
        if 'statistics' in self.analysis_results:
            stats = self.analysis_results['statistics']
            for joint, joint_stats in stats.items():
                joint_name = JOINT_NAMES_CZ.get(joint, joint)
                min_val = joint_stats.get('min', 0)
                min_frame = joint_stats.get('min_frame', 0)
                max_val = joint_stats.get('max', 0)
                max_frame = joint_stats.get('max_frame', 0)
                count = joint_stats.get('count', 0)
                
                info_html += f"""
                <div style="margin-bottom: 18px; padding: 12px; background: #f5f9fc; border-left: 5px solid #0066cc; border-radius: 3px;">
                <p style="margin: 0 0 10px 0; font-weight: bold; font-size: 14px; color: #0066cc;">{joint_name}</p>
                <table style="width: 100%; margin-top: 8px; font-size: 12px; border-collapse: collapse;">
                <tr style="background: #ffffff;">
                    <td style="padding: 6px 8px; font-weight: bold; width: 35%;">Min:</td>
                    <td style="text-align: right; padding: 6px 8px; color: #ff6600; font-weight: bold; font-size: 13px;">{min_val:.1f}° (snímek {min_frame}/{total_frames})</td>
                </tr>
                <tr style="background: #fafbfc;">
                    <td style="padding: 6px 8px; font-weight: bold;">Max:</td>
                    <td style="text-align: right; padding: 6px 8px; color: #ff6600; font-weight: bold; font-size: 13px;">{max_val:.1f}° (snímek {max_frame}/{total_frames})</td>
                </tr>
                <tr style="background: #e8f4f8; border-top: 2px solid #0066cc; border-bottom: 2px solid #0066cc;">
                    <td style="padding: 6px 8px; font-weight: bold; color: #0066cc;">Rozsah (ROM):</td>
                    <td style="text-align: right; padding: 6px 8px; color: #0066cc; font-weight: bold; font-size: 14px;">{max_val - min_val:.1f}°</td>
                </tr>
                <tr style="background: #ffffff; border-top: 2px solid #e0e0e0;">
                    <td style="padding: 6px 8px; font-weight: bold;">Snímků:</td>
                    <td style="text-align: right; padding: 6px 8px;">{count}</td>
                </tr>
                </table>
                </div>
                """
        
        info_html += """
        </body>
        </html>
        """
        
        self.info_text.setHtml(info_html)
    
    def _get_model_name(self):
        """
        Vrátí internalní název modelu a režim
        Returns: (model_name, mode) kde mode je 'image' nebo 'video'
        """
        text = self.model_combo.currentText()
        
        # Parse formát "ModelName - Mode"
        if " - " in text:
            model_part, mode_part = text.split(" - ", 1)
            mode = mode_part.lower()  # "image" nebo "video"
        else:
            # Fallback pro starý formát
            model_part = text
            mode = "image"
        
        model_map = {
            "MediaPipe": "mediapipe",
            "MoveNet Lightning": "movenet_lightning",
            "MoveNet Thunder": "movenet_thunder",
            "YOLO11n": "yolo11n",
            "YOLO11x": "yolo11x",
            "ViTPose": "vitpose",
        }
        
        model_name = model_map.get(model_part, "mediapipe")
        return model_name, mode
    
    def _on_model_changed(self):
        """Volá se při změně modelu"""
        self._update_mode_info()
    
    def _update_mode_info(self):
        """Aktualizuje info text o režimu a zobrazí/skryje smooth factor"""
        model_name, mode = self._get_model_name()
        
        if mode == "video":
            info = "📹 Video režim: Tracking a vyhlazování pro plynulejší výsledky"
            self.mode_info_label.setStyleSheet("color: #00aa00; font-style: italic; font-size: 10px; font-weight: bold;")
            # Zobraz smooth factor pro video režim
            self.smooth_label.setVisible(True)
            self.smooth_spin.setVisible(True)
        else:
            info = "🖼️ Image režim: Každý snímek zpracován nezávisle"
            self.mode_info_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
            # Skryj smooth factor pro image režim
            self.smooth_label.setVisible(False)
            self.smooth_spin.setVisible(False)
        
        self.mode_info_label.setText(info)
    
    def _export_json(self):
        """Exportuje výsledky do JSON - strukturováno po framech"""
        if self.analysis_results is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Uložit JSON",
            "",
            "JSON soubory (*.json)"
        )
        
        if file_path:
            import json
            
            # Vytvoř strukturu: každý frame s jeho úhly
            frames_data = []
            
            # Získej angles data
            angles = self.analysis_results.get('angles', {})
            total_frames = self.analysis_results.get('total_frames', 0)
            
            # Pro každý frame vytvoř záznam
            if angles:
                # Zjisti maximální počet měření (některé klouby mohou mít různý počet)
                max_measurements = max(len(v) for v in angles.values()) if angles else 0
                
                for frame_idx in range(max_measurements):
                    frame_data = {
                        'frame': frame_idx,
                        'angles': {}
                    }
                    
                    # Přidej úhly všech kloubů pro tento frame
                    for joint, values in angles.items():
                        if frame_idx < len(values):
                            angle_value = values[frame_idx]
                            # Konvertuj numpy typy na Python typy
                            if angle_value is not None:
                                if isinstance(angle_value, np.ndarray):
                                    angle_value = float(angle_value)
                                elif hasattr(angle_value, 'item'):  # numpy scalar
                                    angle_value = angle_value.item()
                                frame_data['angles'][joint] = angle_value
                            else:
                                frame_data['angles'][joint] = None
                    
                    frames_data.append(frame_data)
            
            # Vytvoř finální export strukturu
            export_data = {
                'video_info': {
                    'total_frames': total_frames,
                    'fps': self.analysis_results.get('fps', 30),
                    'analyzed_frames': len(frames_data)
                },
                'statistics': self.analysis_results.get('statistics', {}),
                'frames': frames_data
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "Export", f"Data exportována:\n{len(frames_data)} snímků uloženo do JSON")
    
    def _export_xml(self):
        """Exportuje výsledky do XML - strukturováno po framech"""
        if self.analysis_results is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Uložit XML",
            "",
            "XML soubory (*.xml)"
        )
        
        if file_path:
            import xml.etree.ElementTree as ET
            from xml.dom import minidom
            
            # Vytvoř root element
            root = ET.Element("pose_analysis")
            
            # Video info
            video_info = ET.SubElement(root, "video_info")
            ET.SubElement(video_info, "total_frames").text = str(self.analysis_results.get('total_frames', 0))
            ET.SubElement(video_info, "fps").text = str(self.analysis_results.get('fps', 30))
            
            # Získej angles data
            angles = self.analysis_results.get('angles', {})
            
            if angles:
                # Zjisti maximální počet měření
                max_measurements = max(len(v) for v in angles.values()) if angles else 0
                ET.SubElement(video_info, "analyzed_frames").text = str(max_measurements)
                
                # Statistics
                if 'statistics' in self.analysis_results:
                    stats_elem = ET.SubElement(root, "statistics")
                    for joint, joint_stats in self.analysis_results['statistics'].items():
                        joint_elem = ET.SubElement(stats_elem, "joint", name=joint)
                        ET.SubElement(joint_elem, "min").text = f"{joint_stats.get('min', 0):.1f}"
                        ET.SubElement(joint_elem, "min_frame").text = str(joint_stats.get('min_frame', 0))
                        ET.SubElement(joint_elem, "max").text = f"{joint_stats.get('max', 0):.1f}"
                        ET.SubElement(joint_elem, "max_frame").text = str(joint_stats.get('max_frame', 0))
                        ET.SubElement(joint_elem, "range_of_motion").text = f"{joint_stats.get('max', 0) - joint_stats.get('min', 0):.1f}"
                        ET.SubElement(joint_elem, "count").text = str(joint_stats.get('count', 0))
                
                # Frames data
                frames_elem = ET.SubElement(root, "frames")
                
                for frame_idx in range(max_measurements):
                    frame_elem = ET.SubElement(frames_elem, "frame", index=str(frame_idx))
                    angles_elem = ET.SubElement(frame_elem, "angles")
                    
                    # Přidej úhly všech kloubů pro tento frame
                    for joint, values in angles.items():
                        if frame_idx < len(values):
                            angle_value = values[frame_idx]
                            if angle_value is not None:
                                # Konvertuj numpy typy na Python typy
                                if isinstance(angle_value, np.ndarray):
                                    angle_value = float(angle_value)
                                elif hasattr(angle_value, 'item'):
                                    angle_value = angle_value.item()
                                ET.SubElement(angles_elem, "joint", name=joint).text = f"{angle_value:.2f}"
                            else:
                                ET.SubElement(angles_elem, "joint", name=joint).text = "null"
            
            # Vytvoř hezky formátovaný XML
            xml_str = ET.tostring(root, encoding='unicode')
            dom = minidom.parseString(xml_str)
            pretty_xml = dom.toprettyxml(indent="  ")
            
            # Ulož do souboru
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            
            QMessageBox.information(self, "Export", f"Data exportována:\n{max_measurements if angles else 0} snímků uloženo do XML")
    
    
    def _show_graphs(self):
        """Zobrazí grafy analýzy - v novém PyQt okně"""
        print("DEBUG: _show_graphs() zavolána")
        
        if self.analysis_results is None:
            print("DEBUG: analysis_results je None")
            QMessageBox.warning(self, "Varování", "Nejprve proveďte analýzu")
            return
        
        print(f"DEBUG: analysis_results = {list(self.analysis_results.keys())}")
        
        try:
            from matplotlib.figure import Figure
            # PyQt6 backend!
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
            
            print("DEBUG: Matplotlib importy OK")
            
            # Vytvoř nové PyQt okno - ulož jako instanční proměnnou!
            self.graph_window = QMainWindow()
            self.graph_window.setWindowTitle("Grafy úhlů - Analýza pohybu")
            self.graph_window.setGeometry(100, 100, 1400, 800)
            
            # Hlavní widget
            main_widget = QWidget()
            self.graph_window.setCentralWidget(main_widget)
            layout = QVBoxLayout(main_widget)
            layout.setContentsMargins(5, 5, 5, 5)
            
            # Počet vybraných kloubů
            angles_dict = {j: v for j, v in self.analysis_results.get('angles', {}).items() if v}
            num_joints = len(angles_dict)
            
            if num_joints == 0:
                QMessageBox.warning(self, "Varování", "Žádné úhly k zobrazení")
                return
            
            # Vypočítej optimální grid pro subploty
            if num_joints == 1:
                rows, cols = 1, 1
            elif num_joints == 2:
                rows, cols = 1, 2
            elif num_joints == 3:
                rows, cols = 1, 3
            elif num_joints == 4:
                rows, cols = 2, 2
            elif num_joints == 5 or num_joints == 6:
                rows, cols = 2, 3
            elif num_joints == 7 or num_joints == 8:
                rows, cols = 2, 4
            elif num_joints == 9 or num_joints == 10:
                rows, cols = 2, 5
            else:
                cols = 4
                rows = (num_joints + cols - 1) // cols
            
            # Vytvoř figure s subploty
            fig = Figure(figsize=(14, 8), dpi=100)
            
            plot_idx = 1
            # Vykresli data
            for joint, values in angles_dict.items():
                ax = fig.add_subplot(rows, cols, plot_idx)
                
                # Filtruj None hodnoty
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    ax.plot(valid_values, linewidth=2, color='#1f77b4')
                    ax.fill_between(range(len(valid_values)), valid_values, alpha=0.3, color='#1f77b4')
                    
                    avg = sum(valid_values) / len(valid_values)
                    min_val = min(valid_values)
                    max_val = max(valid_values)
                    joint_name = JOINT_NAMES_CZ.get(joint, joint)
                    ax.set_title(f"{joint_name}\nØ: {avg:.1f}° | Min: {min_val:.1f}° | Max: {max_val:.1f}°", 
                               fontsize=8, fontweight='bold')
                    ax.set_xlabel("Frame", fontsize=8)
                    ax.set_ylabel("Úhel (°)", fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=7)
                
                plot_idx += 1
            
            fig.tight_layout()
            
            # Vytvoř PyQt6 canvas
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
            # Přidej toolbar pro zoom/pan
            toolbar = NavigationToolbar(canvas, main_widget)
            layout.addWidget(toolbar)
            
            print("DEBUG: Zobrazuji okno...")
            
            # Zobraz okno - self.graph_window!
            self.graph_window.show()
            self.graph_window.raise_()
            self.graph_window.activateWindow()
            
            print("DEBUG: Okno zobrazeno!")
            
        except Exception as e:
            print(f"DEBUG: CHYBA! {e}")
            QMessageBox.critical(self, "Chyba", f"Chyba při zobrazení grafů:\n{str(e)}")
            import traceback
            traceback.print_exc()
