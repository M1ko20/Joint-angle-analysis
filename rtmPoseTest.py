# rtmPoseTest_B.py
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import json

register_all_modules()

# Změněné cesty na tvé RTMPose soubory
config_file = 'RTMPose/rtmpose-l_8xb256-420e_coco-384x288.py'
checkpoint_file = 'RTMPose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth'

# Používáme "low-level" API
model = init_model(config_file, checkpoint_file, device='cpu')

input_img_path = '/home/adam/Downloads/Analysis/Analysis/pose.jpg'

# Provedení inference
results = inference_topdown(model, input_img_path)

# Výpis (stejný jako v tvém mmposetest.py)
if not results:
    print("❌ Nebyla detekována žádná osoba.")
else:
    print(f"✅ Detekováno {len(results)} osob(a). Surová data klíčových bodů:")
    
    for i, person_result in enumerate(results):
        print(f"\n--- Osoba {i+1} ---")
        try:
            data = person_result.pred_instances.to_dict()
            print(f"BBox (Oblast): {data.get('bboxes', 'N/A')}")
            print("Klíčové body (X, Y, Skóre):")
            print(json.dumps(data.get('keypoints', 'N/A').tolist(), indent=2))
            
        except AttributeError:
            print(person_result)
