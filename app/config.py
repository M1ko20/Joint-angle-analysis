"""
Konfigurační soubor aplikace
"""

# Dostupné modely detekce
AVAILABLE_MODELS = {
    'mediapipe': 'MediaPipe',
    'movenet_lightning': 'MoveNet Lightning',
    'movenet_thunder': 'MoveNet Thunder',
    'yolo11n': 'YOLO11n',
    'yolo11x': 'YOLO11x',
}

# Výchozí nastavení
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MODEL = 'mediapipe'
DEFAULT_SMOOTHING = 30  # fps

# Dostupné klouby
AVAILABLE_JOINTS = {
    'left_elbow': 'Levý loket',
    'right_elbow': 'Pravý loket',
    'left_shoulder': 'Levé rameno',
    'right_shoulder': 'Pravé rameno',
    'left_knee': 'Levé koleno',
    'right_knee': 'Pravé koleno',
    'left_hip': 'Levý bok',
    'right_hip': 'Pravý bok',
}

# GUI Rozměry
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
VIDEO_PREVIEW_WIDTH = 640
VIDEO_PREVIEW_HEIGHT = 480

# Exportní formáty
EXPORT_FORMATS = {
    'json': 'JSON',
    'xml': 'XML',
    'csv': 'CSV',
}
