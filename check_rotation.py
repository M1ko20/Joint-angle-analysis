#!/usr/bin/env python3
"""
Diagnostický skript pro kontrolu rotace videí
Zjišťuje jestli jsou input videa otočená nebo se otáčejí během zpracování
"""

import cv2
import json
import os
from pathlib import Path
from datetime import datetime


def get_video_metadata(video_path):
    """Získá metadata videa včetně možné rotace"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    metadata = {
        'path': str(video_path),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    
    # Zkus získat rotaci (ne všechny verze OpenCV to podporují)
    try:
        rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        metadata['rotation_meta'] = rotation
    except:
        metadata['rotation_meta'] = 'N/A'
    
    # Přečti první frame
    success, frame = cap.read()
    if success:
        metadata['first_frame_shape'] = frame.shape  # (height, width, channels)
        metadata['first_frame_readable'] = True
    else:
        metadata['first_frame_shape'] = None
        metadata['first_frame_readable'] = False
    
    cap.release()
    return metadata, frame if success else None


def check_all_videos(videos_root="video", output_folder="rotation_check_output"):
    """Zkontroluje všechna videa v side/front složkách"""
    videos_root = Path(videos_root)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"🔍 KONTROLA ROTACE VIDEÍ")
    print(f"{'='*80}")
    print(f"📁 Videa: {videos_root}")
    print(f"📁 Output: {output_folder}\n")
    
    # Projdi všechna videa v side/front
    for view in ['side', 'front']:
        view_path = videos_root / view
        
        if not view_path.exists():
            print(f"⚠️  Složka {view} neexistuje")
            continue
        
        print(f"\n{'='*80}")
        print(f"📂 Pohled: {view.upper()}")
        print(f"{'='*80}\n")
        
        for video_file in sorted(view_path.glob("*.mp4")):
            video_name = video_file.stem
            
            print(f"📹 Zpracovávám: {view}/{video_name}.mp4")
            
            # Získej metadata
            metadata, first_frame = get_video_metadata(video_file)
            
            if metadata is None:
                print(f"   ❌ Nelze otevřít video!")
                results.append({
                    'view': view,
                    'video': video_name,
                    'error': 'Cannot open video'
                })
                continue
            
            # Vytiskni info
            print(f"   📐 Rozměry: {metadata['width']}x{metadata['height']}")
            print(f"   🎬 FPS: {metadata['fps']:.2f}")
            print(f"   🔢 Framy: {metadata['frame_count']}")
            print(f"   📊 Frame shape: {metadata['first_frame_shape']}")
            print(f"   🔄 Rotation meta: {metadata['rotation_meta']}")
            
            # Kontrola: je video portrét nebo landscape?
            if metadata['width'] < metadata['height']:
                orientation = "PORTRAIT (možná otočené!)"
                print(f"   ⚠️  {orientation}")
            else:
                orientation = "LANDSCAPE (normální)"
                print(f"   ✅ {orientation}")
            
            metadata['orientation'] = orientation
            
            # Ulož první frame pro vizuální kontrolu
            if first_frame is not None:
                frame_output = output_folder / f"{view}_{video_name}_frame0.jpg"
                cv2.imwrite(str(frame_output), first_frame)
                print(f"   💾 První frame uložen: {frame_output.name}")
                metadata['first_frame_saved'] = str(frame_output)
            
            results.append({
                'view': view,
                'video': video_name,
                'metadata': metadata
            })
            
            print()
    
    # Ulož výsledky do JSON
    report_file = output_folder / "rotation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'videos': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"📊 SOUHRN")
    print(f"{'='*80}\n")
    
    # Počet videí podle orientace
    portrait_count = sum(1 for r in results 
                        if 'metadata' in r and 'PORTRAIT' in r['metadata'].get('orientation', ''))
    landscape_count = sum(1 for r in results 
                         if 'metadata' in r and 'LANDSCAPE' in r['metadata'].get('orientation', ''))
    
    print(f"📹 Celkem videí: {len(results)}")
    print(f"📐 Landscape (normální): {landscape_count}")
    print(f"⚠️  Portrait (možná otočené): {portrait_count}")
    print(f"\n💾 Report uložen: {report_file}")
    print(f"🖼️  První framy uloženy v: {output_folder}/")
    
    if portrait_count > 0:
        print(f"\n⚠️  POZOR: Našel jsem {portrait_count} video/videí v portrait orientaci!")
        print(f"   To znamená, že videa jsou pravděpodobně otočená o 90°.")
        print(f"   Zkontroluj první framy v {output_folder}/ složce.")
    
    print(f"\n{'='*80}\n")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Kontrola rotace videí - zjišťuje jestli jsou input videa otočená"
    )
    parser.add_argument("--videos", "-v", type=str, default="video",
                       help="Cesta k složce s videi (default: video)")
    parser.add_argument("--output", "-o", type=str, default="rotation_check_output",
                       help="Výstupní složka (default: rotation_check_output)")
    
    args = parser.parse_args()
    
    results = check_all_videos(args.videos, args.output)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())