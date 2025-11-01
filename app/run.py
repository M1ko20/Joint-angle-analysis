#!/usr/bin/env python3
"""
Spuštěný helper - Ověřuje prostředí a spouští aplikaci
"""
import sys
import os
from pathlib import Path

def check_environment():
    """Ověřuje, že je prostředí správně nastaveno"""
    print("Ověřuji prostředí...")
    
    required_modules = ['PyQt6', 'cv2', 'numpy', 'matplotlib', 'scipy', 'mediapipe']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module}")
            missing.append(module)
    
    if missing:
        print(f"\nChybí moduly: {', '.join(missing)}")
        print("\nNaprav to spuštěním:")
        print("  cd /Users/adammikolas/Analysis")
        print("  bash setup_dependencies.sh")
        return False
    
    print("\n✓ Prostředí je OK")
    return True

def main():
    if not check_environment():
        return 1
    
    try:
        from ui.main_window import MainWindow
        from PyQt6.QtWidgets import QApplication
        
        print("\nSpouštím aplikaci...")
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Chyba při spuštění: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
