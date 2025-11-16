#!/usr/bin/env python3
"""
MediaPipe Runner - 4 varianty:
- 2D Video, 2D Image, 3D Video, 3D Image
"""

import cv2
import os
import sys
import argparse
import numpy as np

# Import angle utils
sys.path.insert(0, os.path.dirname(__file__))
from angle_utils import *

try:
    import mediapipe as mp
except ImportError:
    print("❌ MediaPipe není nainstalován: pip install mediapipe")
    sys.exit(1)


def process_mediapipe(video_path, output_dir, mode="video", dimension="2d", confidence=0.5):
    """
    Zpracuje video pomocí MediaPipe
    
    Args:
        video_path: Cesta k videu
        output_dir: Výstupní složka
        mode: "video" nebo "image"
        dimension: "2d" nebo "3d"
        confidence: Confidence threshold
    """
    
    detector_name = f"mediapipe_{dimension}_{mode}"
    set_detector_type("mediapipe")
    set_confidence_threshold(confidence)
    
    print(f"\n{'='*80}")
    print(f"🚀 {detector_name.upper()}")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializace MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    static_image_mode = (mode == "image")
    
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence
    )
    
    print(f"✅ MediaPipe inicializován:")
    print(f"   Mode: {'IMAGE' if static_image_mode else 'VIDEO'}")
    print(f"   Dimension: {dimension.upper()}")
    print(f"   Confidence: {confidence}")
    
    # Otevři video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Nelze otevřít video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Output video
    output_video = os.path.join(output_dir, "analyzed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Data storage
    all_keypoints = []
    angles_data = {
        "right_elbow": [], "left_elbow": [],
        "right_shoulder": [], "left_shoulder": [],
        "right_hip": [], "left_hip": [],
        "right_knee": [], "left_knee": []
    }
    
    frame_id = 0
    
    print(f"🎬 Zpracovávám...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detekce
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            frame_data = {
                "frame_id": frame_id,
                "timestamp": frame_id / fps,
                "keypoints": []
            }
            
            if results.pose_landmarks:
                # Extrahuj keypoints
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    x = landmark.x * width
                    y = landmark.y * height
                    v = landmark.visibility
                    
                    # Ulož do frame_data
                    kp_data = {
                        "x": float(x),
                        "y": float(y),
                        "visibility": float(v)
                    }
                    
                    # Pro 3D přidej Z
                    if dimension == "3d":
                        kp_data["z"] = float(landmark.z)
                    
                    frame_data["keypoints"].append(kp_data)
                    
                    # Pro výpočet úhlů (vždy 2D keypoints formát)
                    keypoints.extend([x, y, v])
                
                keypoints = np.array(keypoints)
                
                # Vypočítej úhly (vždy v 2D - ignoruj Z pro úhly)
                angles = calculate_all_angles(keypoints)
                
                for joint, angle in angles.items():
                    if angle is not None:
                        angles_data[joint].append((angle, frame_id))
                
                # Vykresli landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Info text
                cv2.putText(frame, f"{detector_name.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
        pose.close()
    
    print(f"✅ Zpracováno {frame_id} frames")
    
    # Ulož výsledky
    print(f"💾 Ukládám výsledky...")
    save_all_results(all_keypoints, angles_data, output_dir, detector_name, fps)
    
    print(f"✅ {detector_name} dokončen!")
    return True


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose Detection")
    parser.add_argument("--video", required=True, help="Video path")
    parser.add_argument("--output-base", required=True, help="Base output directory")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return 1
    
    # Spusť všechny 4 varianty
    variants = [
        ("video", "2d"),
        ("image", "2d"),
        ("video", "3d"),
        ("image", "3d")
    ]
    
    results = {}
    
    for mode, dimension in variants:
        output_dir = os.path.join(args.output_base, f"mediapipe_{dimension}_{mode}")
        os.makedirs(output_dir, exist_ok=True)
        
        success = process_mediapipe(
            args.video,
            output_dir,
            mode=mode,
            dimension=dimension,
            confidence=args.confidence
        )
        
        results[f"mediapipe_{dimension}_{mode}"] = success
    
    # Souhrn
    print(f"\n{'='*80}")
    print(f"📊 MEDIAPIPE SOUHRN")
    print(f"{'='*80}\n")
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    successful = sum(1 for v in results.values() if v)
    print(f"\n✅ Úspěšných: {successful}/4")
    
    return 0 if successful == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
