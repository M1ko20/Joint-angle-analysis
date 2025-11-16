#!/usr/bin/env python3
"""
MoveNet Runner - 4 varianty:
- Lightning Video/Image, Thunder Video/Image
"""

import cv2
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(__file__))
from angle_utils import *

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError:
    print("❌ TensorFlow není nainstalován")
    sys.exit(1)

# Import z pose_detector.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pose_detector import PoseDetector


def process_movenet(video_path, output_dir, model="lightning", mode="video", confidence=0.5):
    """Zpracuje video pomocí MoveNet"""
    
    detector_name = f"movenet_{model}_{mode}"
    detector_type = f"movenet_{model}"
    
    set_detector_type(detector_type)
    set_confidence_threshold(confidence)
    
    print(f"\n{'='*80}")
    print(f"🚀 {detector_name.upper()}")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializace
    try:
        pose_detector = PoseDetector(detector_type, confidence_threshold=confidence)
        print(f"✅ MoveNet {model} inicializován (confidence={confidence})")
    except Exception as e:
        print(f"❌ Chyba: {e}")
        return False
    
    # Pro image mód: resetuj crop region pro každý frame
    if mode == "image":
        original_detect = pose_detector.detect_pose
        def detect_with_reset(frame):
            pose_detector.crop_region = None  # Reset pro každý frame
            return original_detect(frame)
        pose_detector.detect_pose = detect_with_reset
    
    # Otevři video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Nelze otevřít: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Output
    output_video = os.path.join(output_dir, "analyzed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    all_keypoints = []
    angles_data = {k: [] for k in ["right_elbow", "left_elbow", "right_shoulder", "left_shoulder",
                                     "right_hip", "left_hip", "right_knee", "left_knee"]}
    
    frame_id = 0
    
    print(f"🎬 Zpracovávám...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints, detection_result = pose_detector.detect_pose(frame)
            
            frame_data = {"frame_id": frame_id, "timestamp": frame_id/fps, "keypoints": []}
            
            if keypoints is not None:
                for i in range(0, len(keypoints), 3):
                    frame_data["keypoints"].append({
                        "x": float(keypoints[i]),
                        "y": float(keypoints[i+1]),
                        "visibility": float(keypoints[i+2])
                    })
                
                angles = calculate_all_angles(keypoints)
                for joint, angle in angles.items():
                    if angle is not None:
                        angles_data[joint].append((angle, frame_id))
                
                pose_detector.draw_landmarks(frame, detection_result)
                cv2.putText(frame, f"{detector_name.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            all_keypoints.append(frame_data)
            out.write(frame)
            
            if frame_id % 30 == 0:
                print(f"   {frame_id}/{total_frames} ({100*frame_id/total_frames:.1f}%)")
            
            frame_id += 1
    except KeyboardInterrupt:
        print("\n🚫 Přerušeno")
    finally:
        cap.release()
        out.release()
        pose_detector.close()
    
    print(f"✅ Zpracováno {frame_id} frames")
    save_all_results(all_keypoints, angles_data, output_dir, detector_name, fps)
    print(f"✅ {detector_name} dokončen!")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    
    variants = [
        ("lightning", "video"),
        ("lightning", "image"),
        ("thunder", "video"),
        ("thunder", "image")
    ]
    
    results = {}
    for model, mode in variants:
        output_dir = os.path.join(args.output_base, f"movenet_{model}_{mode}")
        success = process_movenet(args.video, output_dir, model, mode, args.confidence)
        results[f"movenet_{model}_{mode}"] = success
    
    print(f"\n{'='*80}\n📊 MOVENET SOUHRN\n{'='*80}\n")
    for name, success in results.items():
        print(f"{'✅' if success else '❌'} {name}")
    
    successful = sum(1 for v in results.values() if v)
    print(f"\n✅ Úspěšných: {successful}/4")
    return 0 if successful == 4 else 1

if __name__ == "__main__":
    sys.exit(main())
