#!/bin/bash
###############################################################################
# Test script - pouze MMPose a PoseFormer
# FIXED: Bez CUDA_VISIBLE_DEVICES, přímo cuda:1
###############################################################################
set -e

BASE_DIR="$HOME/BP-MIK0542/Joint-angle-analysis"
MODEL_TEST_DIR="$BASE_DIR/model_test"
VIDEO_PATH="$BASE_DIR/video/side/zero.mp4"
OUTPUT_DIR="$BASE_DIR/output_test_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "================================================================================"
echo "🎯 TEST MMPOSE & POSEFORMER"
echo "================================================================================"
echo "📹 Video: $VIDEO_PATH"
echo "📁 Output: $OUTPUT_DIR"
echo "🎮 GPU: cuda:1 (NVIDIA GeForce GTX 1070 - 8GB free)"
echo ""

# Kontrola videa
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Video nenalezeno: $VIDEO_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# === MMPOSE (3-4 testy) ===
echo ""
echo "================================================================================"
echo "🚀 MMPOSE (venvMmpose)"
echo "================================================================================"
source "$BASE_DIR/venvMmpose/bin/activate"
python3 "$MODEL_TEST_DIR/run_mmpose.py" \
    --video "$VIDEO_PATH" \
    --output-base "$OUTPUT_DIR"
deactivate

# === POSEFORMER (1 test) ===
echo ""
echo "================================================================================"
echo "🚀 POSEFORMER V2 (venvPoseFormerv2)"
echo "================================================================================"
source "$BASE_DIR/venvPoseFormerv2/bin/activate"
python3 "$MODEL_TEST_DIR/run_poseformer.py" \
    --video "$VIDEO_PATH" \
    --output-base "$OUTPUT_DIR"
deactivate

# === SOUHRN ===
echo ""
echo "================================================================================"
echo "✅ TESTY DOKONČENY!"
echo "================================================================================"
echo ""
echo "📁 Výsledky: $OUTPUT_DIR"
echo ""
echo "Testované modely:"
echo "   - MMPose (rtmpose3d, rtmpose, hrnet)"
echo "   - PoseFormer V2"
echo ""
