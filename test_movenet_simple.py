#!/usr/bin/env python3
"""
Jednoduchý test MoveNet
"""

def test_movenet_only():
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        print("✅ TensorFlow a TensorFlow Hub importovány úspěšně")
        
        # Test načtení MoveNet modelu
        print("🔄 Načítám MoveNet Lightning model...")
        model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        model = hub.load(model_url)
        print("✅ MoveNet Lightning model načten úspěšně")
        
        return True
        
    except Exception as e:
        print(f"❌ Chyba při testování MoveNet: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Test pouze MoveNet...")
    success = test_movenet_only()
    
    if success:
        print("\n🎉 MoveNet je připravený k použití!")
    else:
        print("\n❌ MoveNet není funkční")