#!/usr/bin/env python3
"""
Test script - ověřuje, že všechny dependencies jsou nainstalovány
"""

import sys
import subprocess

def test_imports():
    """Testuje import všech potřebných modulů"""
    modules = [
        'PyQt6',
        'cv2',
        'numpy',
        'matplotlib',
        'scipy',
        'PIL',
        'mediapipe',
        'tensorflow',
        'ultralytics',
    ]
    
    missing = []
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - CHYBÍ")
            missing.append(module)
    
    return missing

def main():
    print("=" * 60)
    print("TEST ZÁVISLOSTÍ APLIKACE")
    print("=" * 60)
    
    missing = test_imports()
    
    print("\n" + "=" * 60)
    if missing:
        print(f"Chybí {len(missing)} modulů: {', '.join(missing)}")
        print("\nInstaluj je příkazem:")
        print(f"pip install {' '.join(missing)}")
        return 1
    else:
        print("✓ Všechny moduly jsou nainstalovány!")
        print("\nAplikaci spustíš příkazem:")
        print("python3 main.py")
        return 0

if __name__ == '__main__':
    sys.exit(main())
