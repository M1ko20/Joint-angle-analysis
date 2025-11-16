

"""
===== run_poseformer.py =====
PoseFormerV2 Runner - pouze video
"""
# ULOŽ JAKO: run_poseformer.py

import os
import sys
import subprocess
import argparse
import shutil

def run_poseformer(video_path, output_dir):
    print(f"\n{'='*80}\n🚀 POSEFORMER V2\n{'='*80}\n")
    os.makedirs(output_dir, exist_ok=True)
    
    poseformer_dir = os.path.join(os.path.dirname(__file__), "..", "PoseFormerV2")
    vis_script = os.path.join(poseformer_dir, "demo/vis.py")
    
    if not os.path.exists(vis_script):
        print(f"❌ vis.py nenalezen: {vis_script}")
        return False
    
    # Zkopíruj video do demo/video/
    demo_video_dir = os.path.join(poseformer_dir, "demo/video")
    os.makedirs(demo_video_dir, exist_ok=True)
    
    video_name = os.path.basename(video_path)
    target_video = os.path.join(demo_video_dir, video_name)
    shutil.copy2(video_path, target_video)
    print(f"📹 Video zkopírováno")
    
    # Spusť PoseFormer
    original_cwd = os.getcwd()
    os.chdir(poseformer_dir)
    
    cmd = ["python", str(vis_script), "--video", video_name, "--gpu", "1"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print("✅ PoseFormerV2 dokončen")
            
            # Zkopíruj výsledky
            pf_output = os.path.join(poseformer_dir, "demo/output", os.path.splitext(video_name)[0])
            if os.path.exists(pf_output):
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                shutil.copytree(pf_output, output_dir)
                print(f"📦 Výsledky zkopírovány")
                # TODO: Parsuj keypoints a vypočítej úhly
                return True
            else:
                print(f"⚠️  Výstup nenalezen: {pf_output}")
                return False
        else:
            print(f"❌ Selhalo: {result.stderr}")
            return False
    except Exception as e:
        os.chdir(original_cwd)
        print(f"❌ {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-base", required=True)
    args = parser.parse_args()
    
    success = run_poseformer(args.video, os.path.join(args.output_base, "poseformerv2"))
    
    print(f"\n{'='*80}\n📊 POSEFORMER - {'✅ ÚSPĚCH' if success else '❌ SELHALO'}\n{'='*80}\n")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
