#!/usr/bin/env python3
"""
Univerzální test všech pose detektorů
Automaticky detekuje prostředí a spouští správné detektory
"""

import os
import sys
import subprocess
import cv2
import json
from datetime import datetime
import argparse


class DetectorTester:
    """Manager pro testování všech detektorů"""
    
    def __init__(self, test_image_path, output_folder="detector_test_output"):
        self.test_image = test_image_path
        self.output_folder = output_folder
        self.results = {}
        
        # Vytvoření výstupní složky
        os.makedirs(output_folder, exist_ok=True)
        
        # Detekce prostředí
        self.venv_python = self._find_venv_python()
        self.conda_env = "openmmlab"
        
        print(f"\n{'='*80}")
        print(f"🎯 POSE DETECTOR TEST SUITE")
        print(f"{'='*80}")
        print(f"📷 Testovací obrázek: {test_image_path}")
        print(f"📁 Výstupní složka: {output_folder}")
        print(f"🐍 venv Python: {self.venv_python if self.venv_python else 'nenalezen'}")
        print(f"🐍 Conda prostředí: {self.conda_env}")
        print(f"{'='*80}\n")
    
    def _find_venv_python(self):
        """Najde Python z venv"""
        # Zkus venv ve stejné složce
        venv_paths = [
            os.path.join(os.path.dirname(__file__), "venv", "bin", "python"),
            os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe"),  # Windows
        ]
        
        for path in venv_paths:
            if os.path.exists(path):
                return path
        
        # Fallback na aktuální Python (pokud je spuštěn z venv)
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            return sys.executable
        
        return None
    
    def test_venv_detectors(self, detectors=None):
        """Testuje detektory z venv prostředí"""
        if self.venv_python is None:
            print("⚠️ venv Python nenalezen, přeskakuji venv detektory")
            return
        
        # Dostupné venv detektory
        available = ["mediapipe", "movenet_lightning", "movenet_thunder", "yolo11n", "yolo11x", "vitpose_base", "vitpose_large", "vitpose_huge"]
        
        if detectors is None:
            detectors = available
        
        print(f"\n{'='*80}")
        print(f"🔧 VENV DETEKTORY")
        print(f"{'='*80}")
        
        for detector in detectors:
            if detector not in available:
                continue
            
            print(f"\n▶️  Testování: {detector.upper()}")
            print(f"{'-'*80}")
            
            result = self._run_venv_detector(detector)
            self.results[detector] = result
            
            if result['success']:
                print(f"✅ {detector.upper()} - ÚSPĚCH")
                print(f"   Keypoints: {result['keypoints_count']}/33")
                print(f"   Čas: {result['time']:.2f}s")
            else:
                print(f"❌ {detector.upper()} - SELHÁNÍ")
                print(f"   Chyba: {result['error']}")
    
    def _run_venv_detector(self, detector):
        """Spustí detektor ve venv"""
        import time
        
        start_time = time.time()
        
        try:
            # Import zde (ne nahoře) aby to fungovalo i když venv není aktivní
            from pose_detector import PoseDetector
            
            # Inicializace
            det = PoseDetector(detector, confidence_threshold=0.5)
            
            # Načtení obrázku
            frame = cv2.imread(self.test_image)
            if frame is None:
                raise FileNotFoundError(f"Nelze načíst: {self.test_image}")
            
            # Detekce
            keypoints, result = det.detect_pose(frame)
            
            if keypoints is None:
                det.close()
                return {
                    'success': False,
                    'error': 'Žádná detekce',
                    'time': time.time() - start_time
                }
            
            # Vykreslení
            det.draw_landmarks(frame, result)
            
            # Uložení
            output_path = os.path.join(self.output_folder, f"{detector}_result.jpg")
            cv2.imwrite(output_path, frame)
            
            # Počet detekovaných bodů
            keypoints_count = sum(1 for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0.5)
            
            det.close()
            
            return {
                'success': True,
                'keypoints_count': keypoints_count,
                'output_image': output_path,
                'time': time.time() - start_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
    
    def test_mmpose_detectors(self, detectors=None):
        """Testuje MMPose detektory (HRNet, RTMPose) v conda"""
        available = ["hrnet", "rtmpose"]
        
        if detectors is None:
            detectors = available
        
        print(f"\n{'='*80}")
        print(f"🔧 MMPOSE DETEKTORY (conda)")
        print(f"{'='*80}")
        
        for detector in detectors:
            if detector not in available:
                continue
            
            print(f"\n▶️  Testování: {detector.upper()}")
            print(f"{'-'*80}")
            
            result = self._run_mmpose_detector(detector)
            self.results[detector] = result
            
            if result['success']:
                print(f"✅ {detector.upper()} - ÚSPĚCH")
                print(f"   Keypoints: {result['keypoints_count']}/33")
                print(f"   Čas: {result['time']:.2f}s")
            else:
                print(f"❌ {detector.upper()} - SELHÁNÍ")
                print(f"   Chyba: {result['error']}")
    
    def _run_mmpose_detector(self, detector):
        """Spustí MMPose detektor v conda prostředí pomocí subprocess"""
        import time

        # --- OPRAVA: Nalezení absolutní cesty k Pythonu ---
        try:
            base_proc = subprocess.run(
                ['conda', 'info', '--base'], 
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            conda_base_path = base_proc.stdout.strip()
            
            # Sestavení cesty pro Linux/macOS
            python_exe_path = os.path.join(conda_base_path, 'envs', self.conda_env, 'bin', 'python3')
            
            if not os.path.exists(python_exe_path):
                # Fallback pro Windows
                python_exe_path_win = os.path.join(conda_base_path, 'envs', self.conda_env, 'python.exe')
                if os.path.exists(python_exe_path_win):
                    python_exe_path = python_exe_path_win
                else:
                    # Fallback na 'python3' pokud cesta selže (a doufat v PATH)
                    python_exe_path = "python3"
                    if not os.path.exists(os.path.join(conda_base_path, 'envs', self.conda_env, 'bin')):
                         print(f"Warning: Nenalezen adresář bin v {conda_base_path}/envs/{self.conda_env}. Spoléhám na PATH.")

        except Exception as e:
            return {
                'success': False,
                'error': f"Chyba při hledání conda python3: {e}",
                'time': 0
            }
        # --- KONEC OPRAVY ---

        start_time = time.time()
        
        # Absolutní cesta k Analysis adresáři
        analysis_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # Vytvoř dočasný test skript
            test_script = f"""
import sys
import os

# KRITICKÉ: Přidej Analysis do PYTHONPATH
sys.path.insert(0, r'{analysis_dir}')

import cv2
import json

try:
    from mmpose_detector import MMPoseDetector
except ImportError as e:
    print(json.dumps({{'success': False, 'error': f'Import failed: {{e}}'}}))
    sys.exit(1)

try:
    detector = MMPoseDetector('{detector}', confidence_threshold=0.5)
    frame = cv2.imread(r'{self.test_image}')
    
    if frame is None:
        print(json.dumps({{'success': False, 'error': 'Cannot load image'}}))
        sys.exit(1)
    
    keypoints, result = detector.detect_pose(frame)
    
    if keypoints is None:
        print(json.dumps({{'success': False, 'error': 'No detection'}}))
        detector.close()
        sys.exit(1)
    
    # Vykreslení
    detector.draw_landmarks(frame, result)
    output_path = r'{os.path.join(self.output_folder, f"{detector}_result.jpg")}'
    cv2.imwrite(output_path, frame)
    
    # Počet keypoints
    keypoints_count = sum(1 for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0.5)
    
    print(json.dumps({{
        'success': True,
        'keypoints_count': keypoints_count,
        'output_image': output_path
    }}))
    
    detector.close()
    
except Exception as e:
    import traceback
    print(json.dumps({{
        'success': False, 
        'error': f'Runtime error: {{str(e)}}',
        'traceback': traceback.format_exc()
    }}))
    sys.exit(1)
"""
            
            # Uložení dočasného skriptu
            temp_script = os.path.join(self.output_folder, f"_temp_{detector}_test.py")
            with open(temp_script, 'w') as f:
                f.write(test_script)
            
            # Spuštění v conda
            cmd = [
                'conda', 'run', '-n', self.conda_env, '--no-capture-output',
                python_exe_path, temp_script  # <-- OPRAVA: Použití proměnné
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            
            stdout, stderr = process.communicate(timeout=60)
            
            # Smazání dočasného skriptu
            try:
                os.remove(temp_script)
            except:
                pass
            
            if process.returncode != 0:
                return {
                    'success': False,
                    'error': f"Process failed: {stderr[:200]}",
                    'time': time.time() - start_time
                }
            
            # Parsování výstupu
            try:
                # Najdi JSON v outputu (může být tam i jiný text)
                lines = stdout.strip().split('\n')
                json_line = None
                for line in reversed(lines):  # Hledej od konce
                    if line.strip().startswith('{'):
                        json_line = line
                        break
                
                if json_line is None:
                    raise ValueError("No JSON output found")
                
                result_data = json.loads(json_line)
                result_data['time'] = time.time() - start_time
                return result_data
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Failed to parse output: {str(e)}\nOutput: {stdout[:200]}",
                    'time': time.time() - start_time
                }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Timeout (60s)',
                'time': 60.0
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }
    
    def save_results(self):
        """Uloží výsledky do JSON"""
        results_file = os.path.join(self.output_folder, "test_results.json")
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'test_image': self.test_image,
            'results': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Výsledky uloženy: {results_file}")
    
    def print_summary(self):
        """Vypíše souhrn výsledků"""
        print(f"\n{'='*80}")
        print(f"📊 SOUHRN VÝSLEDKŮ")
        print(f"{'='*80}\n")
        
        # Rozdělení na úspěšné a neúspěšné
        successful = {k: v for k, v in self.results.items() if v.get('success', False)}
        failed = {k: v for k, v in self.results.items() if not v.get('success', False)}
        
        print(f"✅ Úspěšné: {len(successful)}/{len(self.results)}")
        print(f"❌ Selhání: {len(failed)}/{len(self.results)}\n")
        
        if successful:
            print("Úspěšné detektory:")
            print(f"{'Detektor':<20} {'Keypoints':<12} {'Čas (s)':<10}")
            print('-' * 45)
            for name, result in sorted(successful.items(), key=lambda x: x[1]['time']):
                kp = result.get('keypoints_count', 0)
                time_val = result.get('time', 0)
                print(f"{name:<20} {kp:>3}/33       {time_val:>6.2f}")
        
        if failed:
            print(f"\n{'='*80}")
            print("Selhané detektory:")
            for name, result in failed.items():
                error = result.get('error', 'Unknown error')
                print(f"  • {name}: {error[:60]}...")
        
        print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Test všech pose detektorů")
    parser.add_argument("--image", "-i", type=str, default="pose.jpg",
                       help="Cesta k testovacímu obrázku")
    parser.add_argument("--output", "-o", type=str, default="detector_test_output",
                       help="Výstupní složka")
    parser.add_argument("--venv-only", action="store_true",
                       help="Testovat pouze venv detektory")
    parser.add_argument("--mmpose-only", action="store_true",
                       help="Testovat pouze MMPose detektory")
    parser.add_argument("--detector", "-d", type=str,
                       help="Testovat pouze konkrétní detektor")
    
    args = parser.parse_args()
    
    # Kontrola testovacího obrázku
    if not os.path.exists(args.image):
        print(f"❌ Testovací obrázek neexistuje: {args.image}")
        return 1
    
    # Inicializace testeru
    tester = DetectorTester(args.image, args.output)
    
    # Výběr testů
    if args.detector:
        # Konkrétní detektor
        mmpose_detectors = ["hrnet", "rtmpose"]
        if args.detector in mmpose_detectors:
            tester.test_mmpose_detectors([args.detector])
        else:
            tester.test_venv_detectors([args.detector])
    elif args.mmpose_only:
        tester.test_mmpose_detectors()
    elif args.venv_only:
        tester.test_venv_detectors()
    else:
        # Všechny detektory
        tester.test_venv_detectors()
        tester.test_mmpose_detectors()
    
    # Uložení a výpis
    tester.save_results()
    tester.print_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())