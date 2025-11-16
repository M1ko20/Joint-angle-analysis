#!/bin/bash
###############################################################################
# Master test script - spustí všechny pose detection modely
# Struktura: ~/BP-MIK0542/Joint-angle-analysis/model_test/
###############################################################################

set -e

BASE_DIR="$HOME/BP-MIK0542/Joint-angle-analysis"
MODEL_TEST_DIR="$BASE_DIR/model_test"
VIDEO_PATH="$BASE_DIR/video/side/zero.mp4"
OUTPUT_DIR="$BASE_DIR/output_$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES=1

echo ""
echo "================================================================================"
echo "🎯 TESTOVÁNÍ VŠECH POSE DETECTION MODELŮ"
echo "================================================================================"
echo "📹 Video: $VIDEO_PATH"
echo "📁 Output: $OUTPUT_DIR"
echo "🎮 GPU: cuda:1"
echo ""

# Kontrola videa
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Video nenalezeno: $VIDEO_PATH"
    exit 1
fi

# Kontrola rotace
echo "🔄 Kontrola rotace..."
cd "$BASE_DIR"
if [ -f "check_rotation.py" ]; then
    python3 check_rotation.py --videos "$(dirname "$(dirname "$VIDEO_PATH")")" >/dev/null 2>&1
    ROTATED="$(dirname "$VIDEO_PATH")/zero_rotated.mp4"
    if [ -f "$ROTATED" ]; then
        VIDEO_PATH="$ROTATED"
        echo "   ✓ Používám rotované video"
    else
        echo "   ✓ Video je OK"
    fi
fi

mkdir -p "$OUTPUT_DIR"

# === MEDIAPIPE (4 testy) ===
echo ""
echo "================================================================================"
echo "1/6 MEDIAPIPE (venvMediapipe)"
echo "================================================================================"
source "$BASE_DIR/venvMediapipe/bin/activate"
python3 "$MODEL_TEST_DIR/run_mediapipe.py" \
    --video "$VIDEO_PATH" \
    --output-base "$OUTPUT_DIR" \
    --confidence 0.5
deactivate

# === MOVENET (4 testy) ===
echo ""
echo "================================================================================"
echo "2/6 MOVENET (venvMovenet)"
echo "================================================================================"
source "$BASE_DIR/venvMovenet/bin/activate"
python3 "$MODEL_TEST_DIR/run_movenet.py" \
    --video "$VIDEO_PATH" \
    --output-base "$OUTPUT_DIR" \
    --confidence 0.5
deactivate

# === YOLO (4 testy) ===
echo ""
echo "================================================================================"
echo "3/6 YOLO (venvYolo)"
echo "================================================================================"
source "$BASE_DIR/venvYolo/bin/activate"
python3 "$MODEL_TEST_DIR/run_yolo.py" \
    --video "$VIDEO_PATH" \
    --output-base "$OUTPUT_DIR" \
    --confidence 0.5
deactivate

# === VITPOSE (2 testy) ===
echo ""
echo "================================================================================"
echo "4/6 VITPOSE (venvVitpose)"
echo "================================================================================"
source "$BASE_DIR/venvVitpose/bin/activate"
python3 "$MODEL_TEST_DIR/run_vitpose.py" \
    --video "$VIDEO_PATH" \
    --output-base "$OUTPUT_DIR" \
    --confidence 0.5
deactivate

# === MMPOSE (3-4 testy) ===
echo ""
echo "================================================================================"
echo "5/6 MMPOSE (venvMmpose)"
echo "================================================================================"
source "$BASE_DIR/venvMmpose/bin/activate"
python3 "$MODEL_TEST_DIR/run_mmpose.py" \
    --video "$VIDEO_PATH" \
    --output-base "$OUTPUT_DIR"
deactivate

# === POSEFORMER (1 test) ===
echo ""
echo "================================================================================"
echo "6/6 POSEFORMER V2 (venvPoseFormerv2)"
echo "================================================================================"
source "$BASE_DIR/venvPoseFormerv2/bin/activate"
python3 "$MODEL_TEST_DIR/run_poseformer.py" \
    --video "$VIDEO_PATH" \
    --output-base "$OUTPUT_DIR"
deactivate

# === SROVNÁVACÍ REPORT ===
echo ""
echo "================================================================================"
echo "📊 VYTVÁŘENÍ SROVNÁVACÍHO REPORTU"
echo "================================================================================"
source "$BASE_DIR/venvMediapipe/bin/activate"
python3 "$MODEL_TEST_DIR/create_comparison_report.py" --output "$OUTPUT_DIR"
deactivate

# === FINÁLNÍ SOUHRN ===
echo ""
echo "================================================================================"
echo "✅ VŠECHNY TESTY DOKONČENY!"
echo "================================================================================"
echo ""
echo "📁 Výsledky: $OUTPUT_DIR"
echo ""
echo "📊 Celkem testů: ~19"
echo "   - MediaPipe: 4 (2D/3D × video/image)"
echo "   - MoveNet: 4 (lightning/thunder × video/image)"
echo "   - YOLO: 4 (11n/11x × video/image)"
echo "   - ViTPose: 2 (large/huge)"
echo "   - MMPose: 3-4 (rtmpose3d/rtmpose/hrnet)"
echo "   - PoseFormer: 1"
echo ""
echo "📄 Reporty:"
echo "   - comparison_report.json"
echo "   - comparison_summary.txt"
echo "   - comparison_graphs/"
echo ""
