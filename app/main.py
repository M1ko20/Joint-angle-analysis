"""
Hlavní PyQt6 aplikace pro analýzu pohybu těla
"""
import sys
import os
from pathlib import Path

# Přidej parent directory do path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
