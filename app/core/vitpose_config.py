"""
ViTPose (Vision Transformer Pose) - Config
Model: ViTPose-Large trained on COCO dataset
Keypoints: 17 (COCO format - stejný jako YOLO11)
"""

# Model architektura
VITPOSE_CONFIG = {
    'name': 'vitpose_large_coco_256x192',
    'backbone': {
        'type': 'ViT',
        'img_size': (256, 192),  # Input size (height, width)
        'patch_size': 16,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'drop_path_rate': 0.5,
    },
    'keypoint_head': {
        'type': 'TopdownHeatmapSimpleHead',
        'in_channels': 1024,
        'num_deconv_layers': 2,
        'num_deconv_filters': (256, 256),
        'num_deconv_kernels': (4, 4),
        'final_conv_kernel': 1,
        'num_output_channels': 17,  # COCO format
    },
}

# Data pipeline
DATA_CONFIG = {
    'image_size': [192, 256],  # [width, height]
    'heatmap_size': [48, 64],
    'num_output_channels': 17,
    'num_joints': 17,
    'target_type': 'GaussianHeatmap',
}

# Normalizace - ImageNet stats
NORMALIZE_CONFIG = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

# Keypoints mapping (COCO format - 17 keypoints)
KEYPOINT_NAMES = [
    'nose',              # 0
    'left_eye',          # 1
    'right_eye',         # 2
    'left_ear',          # 3
    'right_ear',         # 4
    'left_shoulder',     # 5
    'right_shoulder',    # 6
    'left_elbow',        # 7
    'right_elbow',       # 8
    'left_wrist',        # 9
    'right_wrist',       # 10
    'left_hip',          # 11
    'right_hip',         # 12
    'left_knee',         # 13
    'right_knee',        # 14
    'left_ankle',        # 15
    'right_ankle',       # 16
]

# Skeleton connections (COCO format)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),           # nose -> eyes
    (1, 3), (2, 4),           # eyes -> ears
    (0, 5), (0, 6),           # nose -> shoulders
    (5, 7), (6, 8),           # shoulders -> elbows
    (7, 9), (8, 10),          # elbows -> wrists
    (5, 11), (6, 12),         # shoulders -> hips
    (11, 13), (12, 14),       # hips -> knees
    (13, 15), (14, 16),       # knees -> ankles
]

# Model file path
MODEL_FILE = 'vitpose-l.pth'

# Post-processing
POST_PROCESS_CONFIG = {
    'flip_test': True,
    'post_process': 'default',
    'shift_heatmap': False,
    'modulate_kernel': 11,
    'use_udp': True,  # Unbiased Data Processing
}

# Thresholds
THRESHOLD_CONFIG = {
    'visibility_threshold': 0.2,  # Visibility threshold for keypoints
    'nms_threshold': 1.0,
    'oks_threshold': 0.9,
}

# Training config (pro referenci)
TRAINING_CONFIG = {
    'optimizer': 'AdamW',
    'lr': 5e-4,
    'betas': (0.9, 0.999),
    'weight_decay': 0.1,
    'grad_clip_max_norm': 1.0,
    'total_epochs': 210,
}

# Inference settings
INFERENCE_CONFIG = {
    'batch_size': 1,
    'num_workers': 0,
    'device': 'cuda',  # or 'cpu'
}
