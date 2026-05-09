import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
from stage3_vlm.inference import VLMInference

img_path = "data/fail/images/tsfabric_T1/000003.jpeg"
print(f"Test image: {img_path}")

inf = VLMInference()
image = np.array(Image.open(img_path).convert("RGB"))
print(f"Image shape: {image.shape}")

result = inf.analyze(image, mask=None, anomaly_score=0.85)

print("\n=== Stage 3 Result ===")
print(f"Caption     : {result['caption']}")
print(f"Defect type : {result['defect_type']}")
print(f"Severity    : {result['severity']}")
print(f"Location    : {result['location']}")
print(f"Pass/Fail   : {result['pass_fail']}")
print(f"Has defect  : {result['has_defect']}")
print(f"Confidence  : {result['confidence']}")
print("\nVQA raw answers:")
for k, v in result['vqa_raw'].items():
    print(f"  {k}: {v}")
