# Joint Angle Analysis - Pose Detection Models

Analýza úhlů kloubů pomocí různých pose detection modelů (MediaPipe, MoveNet, YOLO, ViTPose, MMPose).

## 📋 Obsah

- [Požadavky](#požadavky)
- [Instalace](#instalace)
- [Stažení Model Vah](#stažení-model-vah)
- [Použití](#použití)
- [Modely](#modely)

## 🔧 Požadavky

- Python 3.12+ (venv modely) / Python 3.13+ (conda modely)
- CUDA 12.1+ (pro GPU akceleraci)
- 8GB+ GPU paměť (doporučeno pro RTMPose3D)

## 📦 Instalace

### 1. VENV Prostředí (MediaPipe, MoveNet, YOLO, ViTPose)

```bash
# Vytvoř virtuální prostředí
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# nebo: venv\Scripts\activate  # Windows

# Instaluj závislosti
pip install -r requirementVenv.txt
```

### 2. CONDA Prostředí (MMPose modely: HRNet, RTMPose, RTMPose3D)

```bash
# Vytvoř conda prostředí
conda create -n openmmlab python=3.13
conda activate openmmlab

# Instaluj PyTorch s CUDA
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Instaluj ostatní závislosti
pip install -r requirementsConda.txt
```

## 🤖 Stažení Model Vah

Model váhy nejsou součástí repozitáře (jsou příliš velké). Stáhni je následovně:

### YOLO11 (venv)

```bash
# YOLOv11 Nano Pose
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt

# YOLOv11 X-Large Pose
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
```

### HRNet (conda)

```bash
# HRNet-w48 COCO
wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    -O td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth

# Nebo použij mim (MMPose tool)
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest .
```

### RTMPose (conda)

```bash
# Vytvoř složku RTMPose
mkdir -p RTMPose

# RTMPose-L
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth \
    -O RTMPose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth

# Config soubor
wget https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-384x288.py \
    -O RTMPose/rtmpose-l_8xb256-420e_coco-384x288.py
```

### RTMPose3D (conda)

```bash
# Naklonuj MMPose repozitář (obsahuje configs)
git clone https://github.com/open-mmlab/mmpose.git

# Stáhni RTMPose3D váhy
cd mmpose
wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
```

### ViTPose (venv)

```bash
# ViT-Base
wget https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-b.pth

# ViT-Large  
wget https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-l.pth

# ViT-Huge
wget https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-h.pth
```

**Poznámka:** MediaPipe a MoveNet se stahují automaticky při prvním použití.

## 🚀 Použití

### Batch Analýza - VENV Modely

```bash
source venv/bin/activate
python3 batch_analysis_venv.py --videos video --output outputvenv
```

### Batch Analýza - CONDA Modely (s GPU)

```bash
# GPU je výchozí
python3 batch_analysis_conda.py --videos video --output output

# Pro CPU použij:
python3 batch_analysis_conda.py --videos video --output output --cpu
```

### Generování Grafů a Reportů

```bash
python3 graphs_from_output.py
```

Více informací v [README_BATCH_SPLIT.md](README_BATCH_SPLIT.md)

## 🎯 Modely

### VENV Modely (batch_analysis_venv.py)

| Model | Typ | Rychlost | Přesnost |
|-------|-----|----------|----------|
| MediaPipe | 2D/3D | ⚡⚡⚡ | ⭐⭐⭐ |
| MoveNet Lightning | 2D | ⚡⚡⚡ | ⭐⭐ |
| MoveNet Thunder | 2D | ⚡⚡ | ⭐⭐⭐ |
| YOLO11n | 2D | ⚡⚡⚡ | ⭐⭐⭐ |
| YOLO11x | 2D | ⚡⚡ | ⭐⭐⭐⭐ |
| ViTPose Base | 2D | ⚡⚡ | ⭐⭐⭐⭐ |
| ViTPose Large | 2D | ⚡ | ⭐⭐⭐⭐⭐ |
| ViTPose Huge | 2D | ⚡ | ⭐⭐⭐⭐⭐ |

### CONDA Modely (batch_analysis_conda.py)

| Model | Typ | Rychlost | Přesnost | GPU RAM |
|-------|-----|----------|----------|---------|
| HRNet-w48 | 2D | ⚡⚡ | ⭐⭐⭐⭐ | 4GB |
| RTMPose-L | 2D | ⚡⚡⚡ | ⭐⭐⭐⭐ | 4GB |
| RTMPose3D-L | 3D | ⚡⚡ | ⭐⭐⭐⭐⭐ | 8GB |

## 📁 Struktura Projektu

```
.
├── batch_analysis_venv.py      # Venv modely
├── batch_analysis_conda.py     # Conda modely (MMPose)
├── graphs_from_output.py       # Generování grafů
├── pose_detector.py            # Venv detector wrapper
├── video_pose_detector.py      # Video mode wrapper
├── mmpose_detector.py          # MMPose wrapper (HRNet, RTMPose)
├── rtmpose3d_detector.py       # RTMPose3D wrapper
├── vitPosedetector.py          # ViTPose wrapper
├── pose_analysis_unified.py    # 2D angle calculations
├── pose_analysis_3d.py         # 3D angle calculations
├── requirementVenv.txt         # Venv dependencies
└── requirementsConda.txt       # Conda dependencies
```

## 🐛 Troubleshooting

### CUDA není dostupná

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Pokud `False`, přeinstaluj PyTorch s CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### MMPose Import Error

```bash
conda activate openmmlab
pip install mmpose mmdet mmcv
```

## 📝 Licence

MIT License

## 👤 Autor

Adam Miko

