#!/usr/bin/env python3
"""
Batch analýza CONDA (MMPose) modelů na všech videích
Spouští pouze conda modely (HRNet, RTMPose, RTMPose3D) přes subprocess
Pro venv modely použij batch_analysis_venv.py
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile


class BatchAnalyzerConda:
    """Batch analýza CONDA (MMPose) modelů na všech videích"""
    
    def __init__(self, videos_root="video", output_root="output", confidence_threshold=0.5, 
                 conda_env="openmmlab", use_gpu=True):
        self.videos_root = Path(videos_root)
        self.output_root = Path(output_root)
        self.confidence_threshold = confidence_threshold
        self.conda_env = conda_env
        self.use_gpu = use_gpu
        self.temp_videos = []  # Pro sledování dočasných rotovaných videí
        
        # Seznam conda modelů
        self.conda_models = self._get_conda_models()
        
        print(f"\n{'='*80}")
        print(f"🎯 BATCH POSE ANALYSIS - CONDA (MMPose) MODELY")
        print(f"{'='*80}")
        print(f"📁 Videa: {self.videos_root}")
        print(f"📁 Output: {self.output_root}")
        print(f"🎚️ Confidence: {self.confidence_threshold}")
        print(f"🐍 Conda env: {self.conda_env}")
        print(f"🎮 GPU: {'✅ Zapnuto' if self.use_gpu else '❌ Vypnuto (CPU)'}")
        print(f"🔧 Dostupné modely: {len(self.conda_models)}")
        for model in self.conda_models:
            print(f"   • {model}")
        print(f"{'='*80}\n")
    
    def _get_conda_models(self):
        """Seznam conda modelů (MMPose)"""
        # Kontrola conda prostředí
        try:
            result = subprocess.run(
                ['conda', 'env', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if self.conda_env in result.stdout:
                return ["hrnet", "rtmpose", "rtmpose3d"]
            else:
                print(f"⚠️ Conda prostředí '{self.conda_env}' nenalezeno!")
                return []
        except Exception as e:
            print(f"⚠️ Nelze zkontrolovat conda prostředí: {e}")
            return []
    
    def _detect_video_rotation(self, video_path):
        """
        Detekuje potřebnou rotaci videa pomocí metadat a analýzy rozměrů
        
        Returns:
            int: Rotace ve stupních (90, 180, 270) nebo None
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        # Zkus získat rotation metadata
        try:
            rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            if rotation in [90, 180, 270]:
                cap.release()
                print(f"      📐 Detekována rotace z metadat: {rotation}°")
                return int(rotation)
        except:
            pass
        
        # Analýza rozměrů videa
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = width / height
        
        cap.release()
        
        # Heuristika: video s lidmi by mělo být spíše na šířku nebo čtvercové
        if aspect_ratio < 0.8:
            # Video je výrazně na výšku → pravděpodobně potřebuje rotaci
            print(f"      📐 Video je na výšku (aspect ratio: {aspect_ratio:.2f}), aplikuji rotaci 90°")
            return 90
        elif aspect_ratio > 1.5:
            # Video je velmi široké, možná je otočené o 90° a mělo by být na výšku
            print(f"      📐 Video je velmi široké (aspect ratio: {aspect_ratio:.2f}), možná rotace 270°")
            return 270
        
        print(f"      📐 Video se zdá správně orientované (aspect ratio: {aspect_ratio:.2f})")
        return None
    
    def _rotate_video(self, video_path, rotation_degrees):
        """
        Rotuje video a vrací cestu k dočasnému souboru
        
        Args:
            video_path: Cesta k původnímu videu
            rotation_degrees: Stupně rotace (90, 180, 270)
        
        Returns:
            Path k rotovanému videu
        """
        if rotation_degrees is None or rotation_degrees == 0:
            return video_path
        
        print(f"      🔄 Rotuji video o {rotation_degrees}°...")
        
        # Mapování rotace
        rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        
        if rotation_degrees not in rotation_map:
            print(f"      ⚠️ Neplatná rotace: {rotation_degrees}°")
            return video_path
        
        rotation_code = rotation_map[rotation_degrees]
        
        # Vytvoř dočasný soubor
        temp_dir = Path(tempfile.gettempdir()) / "batch_analysis_rotated"
        temp_dir.mkdir(exist_ok=True)
        
        temp_video = temp_dir / f"{video_path.stem}_rotated_{rotation_degrees}.mp4"
        
        # Načti video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"      ❌ Nelze otevřít video")
            return video_path
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Zjisti nové rozměry
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return video_path
        
        rotated_first = cv2.rotate(first_frame, rotation_code)
        new_height, new_width = rotated_first.shape[:2]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Vytvoř výstup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, fps, (new_width, new_height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rotated_frame = cv2.rotate(frame, rotation_code)
            out.write(rotated_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"      ✅ Video rotováno ({frame_count} framů) → {temp_video.name}")
        
        # Zapamatuj pro cleanup
        self.temp_videos.append(temp_video)
        
        return temp_video
    
    def get_videos(self):
        """Najde všechna videa v side/front složkách"""
        videos = []
        for view in ['side', 'front']:
            view_path = self.videos_root / view
            if view_path.exists():
                for video_file in view_path.glob("*.mp4"):
                    # Název složky bez přípony, oprava překlepů
                    video_name = video_file.stem
                    if video_name == "minustwetny":
                        video_name = "minustwenty"
                    
                    videos.append({
                        'path': video_file,
                        'view': view,
                        'name': video_name
                    })
        return videos
    
    def run_all(self):
        """Spustí analýzu všech conda modelů na všech videích"""
        if not self.conda_models:
            print("❌ Žádné conda modely dostupné!")
            return
        
        videos = self.get_videos()
        
        if not videos:
            print("❌ Žádná videa nenalezena!")
            return
        
        print(f"📹 Nalezeno {len(videos)} videí:")
        for v in videos:
            print(f"   • {v['view']}/{v['name']}.mp4")
        print()
        
        # Sekvenční zpracování
        total = len(self.conda_models) * len(videos)
        current = 0
        
        results_summary = []
        
        for model in self.conda_models:
            print(f"\n{'='*80}")
            print(f"🔧 MODEL: {model.upper()}")
            print(f"{'='*80}\n")
            
            for video in videos:
                current += 1
                print(f"[{current}/{total}] {model} → {video['view']}/{video['name']}")
                
                try:
                    result = self.analyze_video(model, video)
                    results_summary.append(result)
                    
                    if result['success']:
                        print(f"   ✅ Úspěch ({result['time']:.1f}s)")
                    else:
                        print(f"   ❌ Selhání: {result['error']}")
                
                except Exception as e:
                    print(f"   ❌ Kritická chyba: {e}")
                    results_summary.append({
                        'model': model,
                        'video': video['name'],
                        'view': video['view'],
                        'success': False,
                        'error': str(e),
                        'time': 0
                    })
        
        # Uložení souhrnu
        self._save_summary(results_summary)
        self._print_summary(results_summary)
        
        # Cleanup dočasných videí
        self._cleanup_temp_videos()
    
    def _cleanup_temp_videos(self):
        """Smaže dočasná rotovaná videa"""
        if not self.temp_videos:
            return
        
        print(f"\n🧹 Čištění dočasných videí...")
        for temp_video in self.temp_videos:
            try:
                if temp_video.exists():
                    temp_video.unlink()
                    print(f"   • Smazáno: {temp_video.name}")
            except Exception as e:
                print(f"   ⚠️ Nelze smazat {temp_video.name}: {e}")
        
        # Pokus o smazání složky
        try:
            temp_dir = Path(tempfile.gettempdir()) / "batch_analysis_rotated"
            if temp_dir.exists() and not list(temp_dir.iterdir()):
                temp_dir.rmdir()
                print(f"   • Smazána složka: {temp_dir}")
        except:
            pass
    
    def analyze_video(self, model, video_info):
        """Analyzuje jedno video jedním modelem"""
        import time
        start_time = time.time()
        
        video_path = video_info['path']
        view = video_info['view']
        video_name = video_info['name']
        
        # Detekce a aplikace rotace videa
        rotation = self._detect_video_rotation(video_path)
        if rotation:
            video_path = self._rotate_video(video_path, rotation)
        
        # Výstupní složka
        output_dir = self.output_root / model / view / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            success = self._analyze_conda(model, video_path, output_dir)
            
            return {
                'model': model,
                'video': video_name,
                'view': view,
                'success': success,
                'error': None if success else "Unknown error",
                'time': time.time() - start_time,
                'output_dir': str(output_dir)
            }
        
        except Exception as e:
            return {
                'model': model,
                'video': video_name,
                'view': view,
                'success': False,
                'error': str(e),
                'time': time.time() - start_time,
                'output_dir': str(output_dir)
            }
    
    def _analyze_conda(self, model, video_path, output_dir):
        """Analýza pomocí conda modelu (subprocess)"""
        # Vytvoř dočasný skript
        temp_script = self._create_conda_script(model, video_path, output_dir)
        
        try:
            # Spusť v conda
            cmd = [
                'conda', 'run', '-n', self.conda_env, '--no-capture-output',
                'python', str(temp_script)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=600)  # 10 min timeout
            
            # Print výstup z conda procesu
            if stdout:
                print(f"      {stdout.strip()}")
            
            if process.returncode != 0:
                raise RuntimeError(f"Conda process failed: {stderr[:500]}")
            
            return True
        
        finally:
            # Smaž dočasný skript
            if temp_script.exists():
                temp_script.unlink()
    
    def _create_conda_script(self, model, video_path, output_dir):
        """Vytvoří dočasný Python skript pro conda - přímo importuje MMPose"""
        analysis_dir = Path(__file__).parent.absolute()
        
        # Vytvoř správný skript podle modelu
        if model in ['hrnet', 'rtmpose']:
            script_content = self._generate_mmpose_script(model, video_path, output_dir, analysis_dir)
        elif model == 'rtmpose3d':
            script_content = self._generate_rtmpose3d_script(video_path, output_dir, analysis_dir)
        else:
            raise ValueError(f"Neznámý conda model: {model}")
        
        temp_file = output_dir / "_temp_conda_script.py"
        with open(temp_file, 'w') as f:
            f.write(script_content)
        
        return temp_file
    
    def _generate_mmpose_script(self, model, video_path, output_dir, analysis_dir):
        """Generuje skript pro HRNet/RTMPose - přímo importuje MMPose"""
        # Cesty k config a checkpoint souborům
        if model == 'hrnet':
            config_file = analysis_dir / 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
            checkpoint_file = analysis_dir / 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
        else:  # rtmpose
            config_file = analysis_dir / 'RTMPose' / 'rtmpose-l_8xb256-420e_coco-384x288.py'
            checkpoint_file = analysis_dir / 'RTMPose' / 'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth'
        
        device = 'cuda' if self.use_gpu else 'cpu'
        
        return f"""import cv2
import json
import numpy as np
from pathlib import Path

# Import MMPose přímo
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
register_all_modules()

# Inicializace modelu
print("🔧 Inicializuji {model} na {device.upper()}...")
model = init_model(r'{config_file}', r'{checkpoint_file}', device='{device}')
print("✅ Model načten")

# Zpracování videa
cap = cv2.VideoCapture(r'{video_path}')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_dir = Path(r'{output_dir}')
frames_dir = output_dir / "frames"
frames_dir.mkdir(exist_ok=True)

output_video = output_dir / "analyzed_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

# COCO skeleton
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

keypoints_data = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detekce
    results = inference_topdown(model, frame)
    
    if results and len(results) > 0:
        result = results[0]
        data = result.pred_instances.to_dict()
        keypoints = data.get('keypoints', [])
        scores = data.get('keypoint_scores', [])
        
        if len(keypoints) > 0 and len(scores) > 0:
            kps = keypoints[0]
            scr = scores[0]
            
            # Vykreslení
            for i, (x, y) in enumerate(kps):
                if scr[i] > {self.confidence_threshold}:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            for start_idx, end_idx in connections:
                if scr[start_idx] > {self.confidence_threshold} and scr[end_idx] > {self.confidence_threshold}:
                    pt1 = (int(kps[start_idx][0]), int(kps[start_idx][1]))
                    pt2 = (int(kps[end_idx][0]), int(kps[end_idx][1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            
            # Uložení
            keypoints_data.append({{'frame': frame_idx, 'keypoints': kps.tolist(), 'scores': scr.tolist()}})
    
    # Zápis
    cv2.imwrite(str(frames_dir / f"{{frame_idx:05d}}.jpg"), frame)
    out.write(frame)
    frame_idx += 1
    
    # Progress
    if frame_idx % 30 == 0:
        progress = (frame_idx / total_frames) * 100
        print(f"Zpracováno: {{progress:.1f}}% ({{frame_idx}}/{{total_frames}})")

cap.release()
out.release()

# Uložení dat
with open(output_dir / "data.json", 'w') as f:
    json.dump(keypoints_data, f, indent=2)

print("✅ Hotovo")
"""
    
    def _generate_rtmpose3d_script(self, video_path, output_dir, analysis_dir):
        """Generuje skript pro RTMPose3D - přímo importuje MMPose/MMDet"""
        mmpose_dir = analysis_dir / "mmpose"
        det_config = mmpose_dir / "projects/rtmpose3d/demo/rtmdet_m_640-8xb32_coco-person.py"
        det_checkpoint = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
        pose_config = mmpose_dir / "projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py"
        pose_checkpoint = mmpose_dir / "rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
        
        device = 'cuda' if self.use_gpu else 'cpu'
        
        return f"""import cv2
import json
import numpy as np
from pathlib import Path

# Import MMPose a MMDet přímo
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmdet.apis import inference_detector, init_detector
register_all_modules()

# Inicializace
print("🔧 Inicializuji RTMPose3D na {device.upper()}...")
print("   Loading detector...")
detector = init_detector(r'{det_config}', r'{det_checkpoint}', device='{device}')
print("   Loading pose model...")
pose_model = init_model(r'{pose_config}', r'{pose_checkpoint}', device='{device}')
print("✅ Modely načteny")

# Zpracování videa
cap = cv2.VideoCapture(r'{video_path}')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_dir = Path(r'{output_dir}')
frames_dir = output_dir / "frames"
frames_dir.mkdir(exist_ok=True)

output_video = output_dir / "analyzed_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

# COCO skeleton
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

keypoints_data = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Person detection
    det_result = inference_detector(detector, frame)
    pred_inst = det_result.pred_instances.cpu().numpy()
    
    bboxes = pred_inst.bboxes
    scores = pred_inst.scores
    labels = pred_inst.labels
    
    mask = np.logical_and(labels == 0, scores > {self.confidence_threshold})
    bboxes = bboxes[mask]
    
    if len(bboxes) > 0:
        # Pose estimation
        pose_results = inference_topdown(pose_model, frame, bboxes)
        
        if pose_results and len(pose_results) > 0:
            result = pose_results[0]
            data = result.pred_instances.to_dict()
            keypoints = data.get('keypoints', [])
            kp_scores = data.get('keypoint_scores', [])
            
            if len(keypoints) > 0 and len(kp_scores) > 0:
                kps = keypoints[0]  # (17, 3) - x, y, z
                scr = kp_scores[0]
                
                # Vykreslení (2D projekce)
                for i, (x, y, z) in enumerate(kps):
                    if scr[i] > {self.confidence_threshold}:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                
                for start_idx, end_idx in connections:
                    if scr[start_idx] > {self.confidence_threshold} and scr[end_idx] > {self.confidence_threshold}:
                        pt1 = (int(kps[start_idx][0]), int(kps[start_idx][1]))
                        pt2 = (int(kps[end_idx][0]), int(kps[end_idx][1]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                
                # Uložení
                keypoints_data.append({{'frame': frame_idx, 'keypoints': kps.tolist(), 'scores': scr.tolist()}})
    
    # Zápis
    cv2.imwrite(str(frames_dir / f"{{frame_idx:05d}}.jpg"), frame)
    out.write(frame)
    frame_idx += 1
    
    # Progress
    if frame_idx % 30 == 0:
        progress = (frame_idx / total_frames) * 100
        print(f"Zpracováno: {{progress:.1f}}% ({{frame_idx}}/{{total_frames}})")

cap.release()
out.release()

# Uložení dat
with open(output_dir / "data.json", 'w') as f:
    json.dump(keypoints_data, f, indent=2)

print("✅ Hotovo")
"""
    
    def _save_summary(self, results):
        """Uloží souhrn všech analýz"""
        summary_file = self.output_root / "batch_summary_conda.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'confidence_threshold': self.confidence_threshold,
            'environment': f'conda ({self.conda_env})',
            'gpu_enabled': self.use_gpu,
            'total_runs': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Souhrn uložen: {summary_file}")
    
    def _print_summary(self, results):
        """Vypíše souhrn"""
        print(f"\n{'='*80}")
        print(f"📊 SOUHRN BATCH ANALÝZY - CONDA")
        print(f"{'='*80}\n")
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"✅ Úspěšných: {len(successful)}/{len(results)}")
        print(f"❌ Selhání: {len(failed)}/{len(results)}")
        
        if successful:
            total_time = sum(r['time'] for r in successful)
            avg_time = total_time / len(successful)
            print(f"⏱️ Celkový čas: {total_time:.1f}s")
            print(f"⏱️ Průměrný čas: {avg_time:.1f}s")
        
        if failed:
            print(f"\nSelhaná zpracování:")
            for r in failed:
                print(f"  • {r['model']} → {r['view']}/{r['video']}: {r['error'][:80]}")
        
        print(f"\n{'='*80}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch analýza CONDA (MMPose) modelů")
    parser.add_argument("--videos", "-v", type=str, default="video",
                       help="Cesta k složce s videi (default: video)")
    parser.add_argument("--output", "-o", type=str, default="output",
                       help="Výstupní složka (default: output)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--conda-env", type=str, default="openmmlab",
                       help="Conda environment (default: openmmlab)")
    parser.add_argument("--cpu", action="store_true",
                       help="Použít CPU místo GPU")
    parser.add_argument("--model", "-m", type=str,
                       help="Spustit pouze konkrétní model")
    
    args = parser.parse_args()
    
    analyzer = BatchAnalyzerConda(
        videos_root=args.videos,
        output_root=args.output,
        confidence_threshold=args.confidence,
        conda_env=args.conda_env,
        use_gpu=not args.cpu
    )
    
    # Filtrování podle argumentů
    if args.model:
        analyzer.conda_models = [m for m in analyzer.conda_models if m == args.model]
        if not analyzer.conda_models:
            print(f"❌ Model '{args.model}' není dostupný!")
            return 1
    
    analyzer.run_all()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
