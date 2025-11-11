#!/bin/bash
# Script pro stažení všech model vah pro Joint Angle Analysis
# Spusť tento skript po klonování repozitáře z GitHubu

set -e  # Exit on error

echo "=================================================="
echo "🤖 Stahování Model Vah"
echo "=================================================="

# Vytvoř složky
mkdir -p RTMPose

echo ""
echo "📥 Stahuji YOLO11 modely..."
if [ ! -f "yolo11n-pose.pt" ]; then
    wget -q --show-progress https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
    echo "✅ yolo11n-pose.pt stažen"
else
    echo "⏭️  yolo11n-pose.pt již existuje"
fi

if [ ! -f "yolo11x-pose.pt" ]; then
    wget -q --show-progress https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
    echo "✅ yolo11x-pose.pt stažen"
else
    echo "⏭️  yolo11x-pose.pt již existuje"
fi

echo ""
echo "📥 Stahuji HRNet model..."
if [ ! -f "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth" ]; then
    wget -q --show-progress https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
        -O td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth
    echo "✅ HRNet stažen"
else
    echo "⏭️  HRNet model již existuje"
fi

if [ ! -f "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py" ]; then
    wget -q --show-progress https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py
    echo "✅ HRNet config stažen"
else
    echo "⏭️  HRNet config již existuje"
fi

echo ""
echo "📥 Stahuji RTMPose model..."
if [ ! -f "RTMPose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth" ]; then
    wget -q --show-progress https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth \
        -O RTMPose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth
    echo "✅ RTMPose stažen"
else
    echo "⏭️  RTMPose model již existuje"
fi

if [ ! -f "RTMPose/rtmpose-l_8xb256-420e_coco-384x288.py" ]; then
    wget -q --show-progress https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-384x288.py \
        -O RTMPose/rtmpose-l_8xb256-420e_coco-384x288.py
    echo "✅ RTMPose config stažen"
else
    echo "⏭️  RTMPose config již existuje"
fi

echo ""
echo "📥 Stahuji RTMPose3D..."
if [ ! -d "mmpose" ]; then
    echo "⏬ Klonuji MMPose repozitář..."
    git clone --depth 1 https://github.com/open-mmlab/mmpose.git
    echo "✅ MMPose repozitář naklonován"
else
    echo "⏭️  MMPose repozitář již existuje"
fi

if [ ! -f "mmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth" ]; then
    wget -q --show-progress https://download.openmmlab.com/mmpose/v1/projects/rtmpose3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth \
        -O mmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
    echo "✅ RTMPose3D stažen"
else
    echo "⏭️  RTMPose3D model již existuje"
fi

echo ""
echo "📥 Stahuji ViTPose modely (volitelné)..."
read -p "Stáhnout ViTPose modely? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -f "vitpose-b.pth" ]; then
        wget -q --show-progress https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-b.pth
        echo "✅ ViTPose-Base stažen"
    fi
    
    if [ ! -f "vitpose-l.pth" ]; then
        wget -q --show-progress https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-l.pth
        echo "✅ ViTPose-Large stážen"
    fi
    
    if [ ! -f "vitpose-h.pth" ]; then
        wget -q --show-progress https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-h.pth
        echo "✅ ViTPose-Huge stážen"
    fi
else
    echo "⏭️  Přeskakuji ViTPose modely"
fi

echo ""
echo "=================================================="
echo "✅ HOTOVO!"
echo "=================================================="
echo ""
echo "📊 Stažené modely:"
ls -lh *.pt *.pth 2>/dev/null || echo "  (žádné .pt/.pth v root)"
ls -lh RTMPose/*.pth 2>/dev/null || echo "  (žádné RTMPose modely)"
ls -lh mmpose/*.pth 2>/dev/null || echo "  (žádné RTMPose3D modely)"
echo ""
echo "ℹ️  MediaPipe a MoveNet se stáhnou automaticky při prvním použití"
echo ""
echo "🚀 Další kroky:"
echo "   1. Vytvoř venv: python3 -m venv venv"
echo "   2. Aktivuj venv: source venv/bin/activate"
echo "   3. Instaluj závislosti: pip install -r requirementVenv.txt"
echo "   4. Pro conda: conda create -n openmmlab python=3.13"
echo "   5. Spusť analýzu: python3 batch_analysis_venv.py"
echo ""
