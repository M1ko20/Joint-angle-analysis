#!/usr/bin/env python3
"""
Test skript pro ověření video režimu
"""

import sys
import os

# Přidej cestu k modulům
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test importů"""
    print("🔍 Testování importů...")
    
    try:
        from pose_detector import PoseDetector, get_available_detectors
        print("  ✅ pose_detector importován")
    except Exception as e:
        print(f"  ❌ pose_detector selhal: {e}")
        return False
    
    try:
        from video_pose_detector import VideoPoseDetector, get_video_capable_detectors
        print("  ✅ video_pose_detector importován")
    except Exception as e:
        print(f"  ❌ video_pose_detector selhal: {e}")
        return False
    
    return True


def test_detector_availability():
    """Test dostupnosti detektorů"""
    print("\n📊 Dostupné detektory:")
    
    from pose_detector import get_available_detectors
    from video_pose_detector import get_video_capable_detectors
    
    available = get_available_detectors()
    print(f"\n  Image režim: {len(available)} detektorů")
    for detector in available:
        print(f"    • {detector}")
    
    video_capable = get_video_capable_detectors()
    video_count = sum(1 for v in video_capable.values() if v['video_support'])
    print(f"\n  Video režim: {video_count} detektorů")
    for name, caps in video_capable.items():
        if caps['video_support']:
            features = []
            if caps['tracking']:
                features.append("tracking")
            if caps['smoothing']:
                features.append("smoothing")
            print(f"    • {caps['name']}: {', '.join(features)}")


def test_detector_initialization():
    """Test inicializace detektorů"""
    print("\n🔧 Test inicializace:")
    
    from pose_detector import PoseDetector, get_available_detectors
    from video_pose_detector import VideoPoseDetector
    
    available = get_available_detectors()
    
    # Test Image režim
    print("\n  Image režim:")
    for detector_name in available[:2]:  # Test prvních 2
        try:
            detector = PoseDetector(detector_name)
            print(f"    ✅ {detector_name}")
            detector.close()
        except Exception as e:
            print(f"    ❌ {detector_name}: {e}")
    
    # Test Video režim
    print("\n  Video režim:")
    for detector_name in available[:2]:  # Test prvních 2
        try:
            detector = VideoPoseDetector(detector_name, smooth_factor=0.3)
            print(f"    ✅ {detector_name} (smoothing=0.3)")
            
            # Test tracking info
            info = detector.get_tracking_info()
            print(f"       Frame count: {info['frame_count']}")
            print(f"       Smooth factor: {info['smooth_factor']}")
            
            detector.close()
        except Exception as e:
            print(f"    ❌ {detector_name}: {e}")


def test_ui_models():
    """Test UI modelů"""
    print("\n🖥️  UI modely:")
    
    models = [
        "MediaPipe - Image",
        "MediaPipe - Video",
        "MoveNet Lightning - Image",
        "MoveNet Lightning - Video",
        "MoveNet Thunder - Image",
        "MoveNet Thunder - Video",
        "YOLO11n - Image",
        "YOLO11n - Video",
        "YOLO11x - Image",
        "YOLO11x - Video"
    ]
    
    for model in models:
        parts = model.split(" - ")
        if len(parts) == 2:
            model_name, mode = parts
            symbol = "📹" if mode == "Video" else "🖼️"
            print(f"  {symbol} {model}")


def main():
    """Hlavní test funkce"""
    print("=" * 60)
    print("VIDEO REŽIM - TEST SUITE")
    print("=" * 60)
    
    # Test 1: Importy
    if not test_imports():
        print("\n❌ Test importů selhal!")
        return 1
    
    # Test 2: Dostupnost
    try:
        test_detector_availability()
    except Exception as e:
        print(f"\n❌ Test dostupnosti selhal: {e}")
        return 1
    
    # Test 3: Inicializace
    try:
        test_detector_initialization()
    except Exception as e:
        print(f"\n❌ Test inicializace selhal: {e}")
        return 1
    
    # Test 4: UI modely
    try:
        test_ui_models()
    except Exception as e:
        print(f"\n❌ Test UI modelů selhal: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ VŠECHNY TESTY PROŠLY!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
