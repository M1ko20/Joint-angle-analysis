#!/usr/bin/env python3
"""
===== run_yolo.py =====
YOLO Runner - 4 varianty: 11n/11x × Video/Image
"""
# ULOŽ JAKO: run_yolo.py

import cv2
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(__file__))
from angle_utils import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pose_detector import PoseDetector

def process_yolo(video_path, output_dir, model="yolo11n", mode="video", confidence=0.5):
    detector_name = f"{model}_{mode}"
    set_detector_type(model)
    set_confidence_threshold(confidence)
    
    print(f"\n{'='*80}\n🚀 {detector_name.upper()}\n{'='*80}\n")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        pose_detector = PoseDetector(model, confidence_threshold=confidence)
        print(f"✅ {model.upper()} inicializován")
    except Exception as e:
        print(f"❌ {e}")
        return False
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_video = os.path.join(output_dir, "analyzed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    all_keypoints = []
    angles_data = {k: [] for k in ["right_elbow", "left_elbow", "right_shoulder", "left_shoulder",
                                     "right_hip", "left_hip", "right_knee", "left_knee"]}
    frame_id = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints, detection_result = pose_detector.detect_pose(frame)
            frame_data = {"frame_id": frame_id, "timestamp": frame_id/fps, "keypoints": []}
            
            if keypoints is not None:
                for i in range(0, len(keypoints), 3):
                    frame_data["keypoints"].append({"x": float(keypoints[i]), "y": float(keypoints[i+1]), "visibility": float(keypoints[i+2])})
                
                angles = calculate_all_angles(keypoints)
                for joint, angle in angles.items():
                    if angle is not None:
                        angles_data[joint].append((angle, frame_id))
                
                pose_detector.draw_landmarks(frame, detection_result)
                cv2.putText(frame, detector_name.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            all_keypoints.append(frame_data)
            out.write(frame)
            if frame_id % 30 == 0:
                print(f"   {frame_id}/{total_frames}")
            frame_id += 1
    finally:
        cap.release()
        out.release()
        pose_detector.close()
    
    save_all_results(all_keypoints, angles_data, output_dir, detector_name, fps)
    print(f"✅ {detector_name} dokončen!")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    
    variants = [("yolo11n", "video"), ("yolo11n", "image"), ("yolo11x", "video"), ("yolo11x", "image")]
    results = {}
    for model, mode in variants:
        output_dir = os.path.join(args.output_base, f"{model}_{mode}")
        results[f"{model}_{mode}"] = process_yolo(args.video, output_dir, model, mode, args.confidence)
    
    print(f"\n{'='*80}\n📊 YOLO SOUHRN\n{'='*80}\n")
    for name, success in results.items():
        print(f"{'✅' if success else '❌'} {name}")
    print(f"\n✅ Úspěšných: {sum(1 for v in results.values() if v)}/4")
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())

