#!/usr/bin/env python3
"""
Jednoduchý skript pro rychlou analýzu kloubů
Používá váš stávající skript myma.py jako základ
"""

import os
from pose_analysis_complete import analyze_video

def main():
    """Hlavní funkce pro rychlé spuštění"""
    
    # Kontrola existence video souboru
    video_files = [
        "video/RLelb_RLshou_RLknee.mp4",
        "RLelb_RLshou_RLknee.mp4",
        "video.mp4"
    ]
    
    video_path = None
    for path in video_files:
        if os.path.exists(path):
            video_path = path
            break
    
    if video_path is None:
        print("❌ Video soubor nebyl nalezen!")
        print("Hledal jsem v těchto umístěních:")
        for path in video_files:
            print(f"  - {path}")
        print("\n📝 Upravte cestu k video souboru níže:")
        video_path = input("Zadejte cestu k video souboru: ").strip()
        
        if not os.path.exists(video_path):
            print(f"❌ Soubor {video_path} neexistuje!")
            return
    
    print(f"✅ Nalezen video soubor: {video_path}")
    print("🚀 Spouštím analýzu...")
    
    try:
        # Spuštění analýzy
        results = analyze_video(video_path, "quick_analysis_output")
        
        print("\n" + "="*60)
        print("✅ ANALÝZA DOKONČENA!")
        print("="*60)
        
        # Rychlý přehled výsledků
        interesting_joints = ["Pravý loket", "Pravé rameno", "Pravé koleno"]
        
        for joint_name in interesting_joints:
            if joint_name in results:
                valid_data = [angle for angle, _ in results[joint_name] if angle is not None]
                if valid_data:
                    min_angle = min(valid_data)
                    max_angle = max(valid_data)
                    range_angle = max_angle - min_angle
                    
                    print(f"\n📊 {joint_name}:")
                    print(f"   Rozsah pohybu: {range_angle:.1f}° ({min_angle:.1f}° - {max_angle:.1f}°)")
                    print(f"   Počet měření: {len(valid_data)}")
        
        print(f"\n📁 Výsledky uloženy v: quick_analysis_output/")
        print(f"🎬 Video s analýzou: quick_analysis_output/analyzed_video.mp4")
        print(f"📈 Grafy: quick_analysis_output/graphs/")
        
    except Exception as e:
        print(f"❌ Chyba při analýze: {e}")
        print("💡 Zkontrolujte, zda máte nainstalované všechny závislosti:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()