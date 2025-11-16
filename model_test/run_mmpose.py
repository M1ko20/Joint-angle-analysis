#!/usr/bin/env python3
"""
===== run_mmpose.py =====
MMPose Runner - RTMPOSE3D, RTMPOSE, HRNet
POZN: Tyto modely nejsou v pose_detector.py, musí se volat přes MMPose API
"""
# ULOŽ JAKO: run_mmpose.py

import cv2
import os
import sys
import subprocess
import argparse
import shutil
sys.path.insert(0, os.path.dirname(__file__))
from angle_utils import *

def run_rtmpose3d(video_path, output_dir):
    """RTMPOSE3D - pouze image (první frame)"""
    print(f"\n{'='*80}\n🚀 RTMPOSE3D (IMAGE)\n{'='*80}\n")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrahuj první frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ Nelze načíst frame")
        return False
    
    temp_img = os.path.join(output_dir, "temp_frame.jpg")
    cv2.imwrite(temp_img, frame)
    
    # Spusť RTMPOSE3D
    mmpose_dir = os.path.join(os.path.dirname(__file__), "..", "mmpose")
    cmd = [
        "python",
        os.path.join(mmpose_dir, "projects/rtmpose3d/demo/body3d_img2pose_demo.py"),
        "--input", temp_img,
        "--output-root", output_dir,
        "--device", "cuda:1",
        os.path.join(mmpose_dir, "demo/rtmdet_m_640-8xb32_coco-person.py"),
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
        os.path.join(mmpose_dir, "projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py"),
        "https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ RTMPOSE3D dokončen")
            # TODO: Parsuj výstup a vypočítej úhly
            return True
        else:
            print(f"❌ Selhalo: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {e}")
        return False

def run_rtmpose(video_path, output_dir):
    """RTMPOSE 2D - pouze video"""
    print(f"\n{'='*80}\n🚀 RTMPOSE (VIDEO)\n{'='*80}\n")
    os.makedirs(output_dir, exist_ok=True)
    
    mmpose_dir = os.path.join(os.path.dirname(__file__), "..", "mmpose")
    cmd = [
        "python",
        os.path.join(mmpose_dir, "demo/topdown_demo_with_mmdet.py"),
        "--input", str(video_path),
        "--output-root", output_dir,
        "--device", "cuda:1",
        "--draw-heatmap",
        os.path.join(mmpose_dir, "projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py"),
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
        os.path.join(mmpose_dir, "projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-384x288.py"),
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("✅ RTMPOSE dokončen")
            return True
        else:
            print(f"❌ Selhalo: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {e}")
        return False

def run_hrnet(video_path, output_dir, mode="video"):
    """HRNet - video nebo image"""
    print(f"\n{'='*80}\n🚀 HRNET ({mode.upper()})\n{'='*80}\n")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from mmpose.apis import inference_topdown, init_model
        from mmpose.utils import register_all_modules
        register_all_modules()
        
        mmpose_dir = os.path.join(os.path.dirname(__file__), "..", "mmpose")
        config = os.path.join(mmpose_dir, "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py")
        checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
        
        model = init_model(config, checkpoint, device='cuda:1')
        print("✅ HRNet model načten")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_keypoints = []
        angles_data = {k: [] for k in ["right_elbow", "left_elbow", "right_shoulder", "left_shoulder",
                                        "right_hip", "left_hip", "right_knee", "left_knee"]}
        frame_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = inference_topdown(model, frame)
            frame_data = {"frame_id": frame_id, "timestamp": frame_id/fps, "keypoints": []}
            
            if results and len(results) > 0:
                pred_instances = results[0].pred_instances
                if hasattr(pred_instances, 'keypoints'):
                    kpts = pred_instances.keypoints[0]
                    scores = pred_instances.keypoint_scores[0]
                    
                    # Převeď na flat array
                    keypoints = []
                    for (kp, score) in zip(kpts, scores):
                        frame_data["keypoints"].append({"x": float(kp[0]), "y": float(kp[1]), "visibility": float(score)})
                        keypoints.extend([float(kp[0]), float(kp[1]), float(score)])
                    
                    keypoints = np.array(keypoints)
                    angles = calculate_all_angles(keypoints)
                    for joint, angle in angles.items():
                        if angle is not None:
                            angles_data[joint].append((angle, frame_id))
            
            all_keypoints.append(frame_data)
            if frame_id % 30 == 0:
                print(f"   {frame_id}/{total_frames}")
            frame_id += 1
        
        cap.release()
        save_all_results(all_keypoints, angles_data, output_dir, f"hrnet_{mode}", fps)
        print(f"✅ HRNet {mode} dokončen!")
        return True
        
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-base", required=True)
    args = parser.parse_args()
    
    results = {
        "rtmpose3d": run_rtmpose3d(args.video, os.path.join(args.output_base, "rtmpose3d")),
        "rtmpose": run_rtmpose(args.video, os.path.join(args.output_base, "rtmpose")),
        "hrnet_video": run_hrnet(args.video, os.path.join(args.output_base, "hrnet_video"), "video"),
        "hrnet_image": run_hrnet(args.video, os.path.join(args.output_base, "hrnet_image"), "image")
    }
    
    print(f"\n{'='*80}\n📊 MMPOSE SOUHRN\n{'='*80}\n")
    for name, success in results.items():
        print(f"{'✅' if success else '❌'} {name}")
    print(f"\n✅ Úspěšných: {sum(1 for v in results.values() if v)}/4")
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())


