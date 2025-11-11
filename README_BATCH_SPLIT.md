# Batch Analysis - Oddělené Skripty

Batch analýza byla rozdělena na dva samostatné skripty pro lepší správu prostředí:

## 📁 Struktura

- **`batch_analysis_venv.py`** - Venv modely (MediaPipe, MoveNet, YOLO, ViTPose)
- **`batch_analysis_conda.py`** - Conda modely (HRNet, RTMPose, RTMPose3D)
- **`batch_analysis.py`** - *DEPRECATED* - Původní kombinovaný skript

## 🚀 Použití

### VENV Modely

```bash
# Aktivuj venv
source venv/bin/activate  # Linux/Mac
# nebo: venv\Scripts\activate  # Windows

# Spusť analýzu
python3 batch_analysis_venv.py

# Parametry:
python3 batch_analysis_venv.py \
    --videos video \
    --output outputvenv \
    --confidence 0.5 \
    --model yolo11x  # Volitelně: spustit jen jeden model
```

**Dostupné venv modely:**
- `mediapipe` / `mediapipe_video`
- `movenet_lightning` / `movenet_lightning_video`
- `movenet_thunder` / `movenet_thunder_video`
- `yolo11n` / `yolo11x`
- `vitpose_base` / `vitpose_large` / `vitpose_huge`
- `MediaPipe3D`

### CONDA Modely (MMPose)

```bash
# Aktivuj conda prostředí (NENÍ nutné - skript to dělá automaticky)
# conda activate openmmlab

# Spusť analýzu s GPU
python3 batch_analysis_conda.py

# Spusť analýzu s CPU (pokud nemáš GPU)
python3 batch_analysis_conda.py --cpu

# Parametry:
python3 batch_analysis_conda.py \
    --videos video \
    --output output \
    --confidence 0.5 \
    --conda-env openmmlab \
    --model hrnet  # Volitelně: spustit jen jeden model
```

**Dostupné conda modely:**
- `hrnet` - HRNet-w48 (COCO 256x192)
- `rtmpose` - RTMPose-L (384x288)
- `rtmpose3d` - RTMPose3D-L (3D pose estimation)

## 🎮 GPU vs CPU

### VENV (batch_analysis_venv.py)
- **Automatická detekce GPU** - PyTorch použije CUDA pokud je dostupná
- Závislosti v `requirementVenv.txt` již obsahují CUDA podporu

### CONDA (batch_analysis_conda.py)
- **GPU je výchozí** - pro CPU použij `--cpu` flag
- Závislosti v `requirementsConda.txt` obsahují komentáře pro instalaci PyTorch s CUDA

## 📦 Instalace Závislostí

### VENV Prostředí

```bash
# Vytvoř venv (pokud ještě neexistuje)
python3 -m venv venv
source venv/bin/activate

# Instaluj závislosti
pip install -r requirementVenv.txt

# Poznámka: requirementVenv.txt už obsahuje PyTorch s CUDA supportem
```

### CONDA Prostředí (pro GPU na serveru)

```bash
# Vytvoř conda prostředí
conda create -n openmmlab python=3.13

# Aktivuj prostředí
conda activate openmmlab

# DŮLEŽITÉ: Nejprve nainstaluj PyTorch s CUDA
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Potom instaluj ostatní závislosti
pip install -r requirementsConda.txt

# Nebo pro pip instalaci PyTorch s CUDA:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Pro 4 grafické karty na serveru:**
```bash
# Zkontroluj dostupné GPU
nvidia-smi

# PyTorch automaticky detekuje všechny GPU
# Pro využití konkrétní GPU použij:
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 batch_analysis_conda.py
```

## 🔧 Rozdíly mezi Skripty

| Vlastnost | batch_analysis_venv.py | batch_analysis_conda.py |
|-----------|------------------------|-------------------------|
| Prostředí | Python venv | Conda (openmmlab) |
| Modely | MediaPipe, MoveNet, YOLO, ViTPose | HRNet, RTMPose, RTMPose3D |
| Výchozí output | `outputvenv/` | `output/` |
| GPU podpora | Automatická (PyTorch) | Výchozí (--cpu pro vypnutí) |
| Subprocess | Ne (přímé volání) | Ano (conda run) |

## 📊 Výstupy

Oba skripty vytvářejí stejnou strukturu výstupů:

```
output/
├── batch_summary_venv.json     # Souhrn venv analýzy
├── batch_summary_conda.json    # Souhrn conda analýzy
└── {model}/
    └── {view}/
        └── {condition}/
            ├── analyzed_video.mp4
            ├── data.json
            ├── results.txt
            ├── angles_timeline.json
            └── frames/
                ├── 00000.jpg
                ├── 00001.jpg
                └── ...
```

## 🐛 Troubleshooting

### CUDA není dostupná

```bash
# Zkontroluj CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.cuda.device_count())"

# Pokud False, přeinstaluj PyTorch s CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Conda prostředí nenalezeno

```bash
# Seznam conda prostředí
conda env list

# Pokud openmmlab chybí, vytvoř ho:
conda create -n openmmlab python=3.13
conda activate openmmlab
pip install -r requirementsConda.txt
```

### Import Error v conda skriptu

```bash
# Ujisti se, že máš nainstalované MMPose/MMDet
conda activate openmmlab
pip install mmpose mmdet mmcv
```

## 📝 Poznámky

1. **Rotace videa** - Oba skripty automaticky detekují a rotují videa pokud jsou špatně orientovaná
2. **Dočasné soubory** - Rotovaná videa jsou ukládána do `/tmp/batch_analysis_rotated/` a automaticky mazána
3. **GPU Memory** - Pro RTMPose3D může být potřeba více GPU paměti (doporučeno >=8GB)
4. **Paralelizace** - Modely běží sekvenčně, pro paralelní běh spusť více instancí s `--model` parametrem

## 🎯 Doporučený Workflow

```bash
# 1. Spusť venv modely (rychlejší, bez conda overhead)
python3 batch_analysis_venv.py

# 2. Spusť conda modely s GPU
python3 batch_analysis_conda.py

# 3. Vygeneruj grafy a reporty
python3 graphs_from_output.py
```
