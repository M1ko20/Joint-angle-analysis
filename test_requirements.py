#!/usr/bin/env python3
"""
Test skript pro kontrolu dostupnosti požadovaných knihoven
"""

def test_imports():
    """Testuje dostupnost všech požadovaných knihoven"""
    results = {}
    
    # Základní knihovny
    try:
        import cv2
        results['opencv'] = f"✅ OpenCV {cv2.__version__}"
    except ImportError as e:
        results['opencv'] = f"❌ OpenCV nedostupné: {e}"
    
    try:
        import numpy as np
        results['numpy'] = f"✅ NumPy {np.__version__}"
    except ImportError as e:
        results['numpy'] = f"❌ NumPy nedostupné: {e}"
    
    try:
        import matplotlib
        results['matplotlib'] = f"✅ Matplotlib {matplotlib.__version__}"
    except ImportError as e:
        results['matplotlib'] = f"❌ Matplotlib nedostupné: {e}"
    
    # Pose detection knihovny
    try:
        import mediapipe as mp
        results['mediapipe'] = f"✅ MediaPipe {mp.__version__}"
    except ImportError as e:
        results['mediapipe'] = f"❌ MediaPipe nedostupné: {e}"
    
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        results['movenet'] = f"✅ MoveNet (TensorFlow {tf.__version__})"
    except ImportError as e:
        results['movenet'] = f"❌ MoveNet nedostupné: {e}"
    
    try:
        from openpose import pyopenpose as op
        results['openpose'] = "✅ OpenPose dostupné"
    except ImportError as e:
        results['openpose'] = f"❌ OpenPose nedostupné: {e}"
    
    return results


def print_installation_instructions():
    """Vypíše instrukce pro instalaci"""
    print("\n📦 INSTRUKCE PRO INSTALACI:")
    print("="*50)
    print()
    print("1. Aktivujte venv:")
    print("   source venv/bin/activate")
    print()
    print("2. Nainstalujte základní knihovny:")
    print("   pip install opencv-python")
    print("   pip install numpy")
    print("   pip install matplotlib")
    print("   pip install mediapipe")
    print()
    print("3. Pro MoveNet (doporučeno pro rychlost):")
    print("   pip install tensorflow")
    print("   pip install tensorflow-hub")
    print()
    print("4. Pro OpenPose (volitelné, složitější instalace):")
    print("   - Stáhněte a nainstalujte OpenPose podle oficiální dokumentace")
    print("   - https://github.com/CMU-Perceptual-Computing-Lab/openpose")
    print()


if __name__ == "__main__":
    print("🔍 KONTROLA DOSTUPNOSTI KNIHOVEN")
    print("="*40)
    
    results = test_imports()
    
    for library, status in results.items():
        print(status)
    
    # Počet úspěšných importů
    successful = sum(1 for status in results.values() if status.startswith("✅"))
    total = len(results)
    
    print(f"\n📊 Celkem: {successful}/{total} knihoven dostupných")
    
    if successful < total:
        print_installation_instructions()
    else:
        print("\n🎉 Všechny knihovny jsou dostupné!")
        print("\n🚀 Můžete spustit:")
        print("   python pose_analysis_unified.py --help")
        print("   python pose_analysis_unified.py --interactive")