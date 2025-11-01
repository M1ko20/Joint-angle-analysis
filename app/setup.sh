#!/bin/bash
# Setup script pro PyQt6 aplikaci analýzy pohybu

set -e  # Exit on error

echo "======================================"
echo "Instalace PyQt6 aplikace"
echo "======================================"

# Zkontroluj Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 není nainstalován!"
    exit 1
fi

echo "✓ Python3 nalezen: $(python3 --version)"

# Vytvoř virtuální prostředí
# if [ ! -d "venv" ]; then
#     echo ""
#     echo "Vytváření virtuálního prostředí..."
#     python3 -m venv venv
#     echo "✓ Virtuální prostředí vytvořeno"
# else
#     echo "✓ Virtuální prostředí již existuje"
# fi

# Aktivuj virtuální prostředí
echo ""
echo "Aktivuji virtuální prostředí..."
cd ..
source venv/bin/activate
cd app

# Aktualizuj pip
echo ""
echo "Aktualizuji pip..."
pip install --upgrade pip setuptools wheel > /dev/null

# Instaluj závislosti
echo ""
echo "Instaluji závislosti..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Závislosti nainstalovány"
else
    echo "❌ requirements.txt nenalezen!"
    exit 1
fi

# Testuj imports
echo ""
echo "Testuji importy..."
python3 test_dependencies.py

echo ""
echo "======================================"
echo "✓ Instalace byla úspěšná!"
echo "======================================"
echo ""
echo "Aplikaci spustíš příkazem:"
echo "  source venv/bin/activate"
echo "  python3 main.py"
echo ""
