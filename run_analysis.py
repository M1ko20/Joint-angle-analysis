#!/usr/bin/env python3
"""
Rychlý skript pro analýzu pohybu s MediaPipe
Použije existující video ze složky video/
"""

from pose_analyzer import PoseAnalyzer
import os

def main():
    # Automaticky najde video ve složce video/
    video_folder = "video"
    video_files = []
    
    if os.path.exists(video_folder):
        for file in os.listdir(video_folder):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(video_folder, file))
    
    if not video_files:
        print("Nebyl nalezen žádný video soubor ve složce 'video/'!")
        print("Prosím, umístěte video soubor do složky 'video/' nebo upravte cestu.")
        return
    
    # Použije první nalezené video
    video_path = video_files[0]
    print(f"Analyzuji video: {video_path}")
    
    # Spuštění analýzy
    output_dir = "pose_analysis_results"
    analyzer = PoseAnalyzer(video_path, output_dir)
    analyzer.run_analysis()
    
    print(f"\n🎉 Analýza dokončena!")
    print(f"📁 Výsledky najdete ve složce: {output_dir}/")
    print(f"📊 Grafy: {output_dir}/*.png")
    print(f"📋 Statistiky: {output_dir}/angles_statistics.txt")
    print(f"📄 JSON data: {output_dir}/angles_timeline.json")
    print(f"🎬 Video s úhly: {output_dir}/output_with_angles.mp4")

if __name__ == "__main__":
    main()