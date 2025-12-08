import os
import json
import gc
import torch
from PIL import Image
from .utils import expand_box
from .detector import ObjectDetector
from .captioner import LocalCaptioner
from .reasoner import QwenReasoner

class ScenePipeline:
    def __init__(self, base_path="./data"):
        self.base_path = base_path
        self.dirs = {
            'input': os.path.join(base_path, 'input_images'),
            'p1': os.path.join(base_path, 'phase1_detections'),
            'p2': os.path.join(base_path, 'phase2_patches'),
            'p3': os.path.join(base_path, 'phase3_captions'),
            'p4': os.path.join(base_path, 'phase4_reasoning')
        }
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)

    def run_phase1_detection(self):
        print("\n--- PHASE 1: DETECTION ---")
        detector = ObjectDetector()
        images = [f for f in os.listdir(self.dirs['input']) if f.endswith('.jpg')]
        for img_file in images:
            detections = detector.detect(os.path.join(self.dirs['input'], img_file))
            # Add IDs (Cell 4 logic)
            for i, obj in enumerate(detections): obj['id'] = f"Obj{i+1}"
            
            with open(os.path.join(self.dirs['p1'], img_file.replace('.jpg', '.json')), 'w') as f:
                json.dump({"image_id": img_file, "objects": detections}, f, indent=4)
        
        del detector; torch.cuda.empty_cache()

    def run_phase2_patching(self):
        print("\n--- PHASE 2: PATCH EXTRACTION ---")
        # Configs from Cell 6
        PADDING = 0.10
        MIN_SIZE = 25
        TARGET_SIZE = (224, 224)

        for js_file in os.listdir(self.dirs['p1']):
            with open(os.path.join(self.dirs['p1'], js_file), 'r') as f: data = json.load(f)
            img_path = os.path.join(self.dirs['input'], js_file.replace('.json', '.jpg'))
            if not os.path.exists(img_path): continue
            
            original_image = Image.open(img_path).convert("RGB")
            
            for obj in data['objects']:
                bbox = obj['bbox']
                if (bbox[2]-bbox[0] < MIN_SIZE) or (bbox[3]-bbox[1] < MIN_SIZE): continue

                expanded_box = expand_box(bbox, original_image.size, PADDING)
                patch = original_image.crop(tuple(expanded_box))
                patch = patch.resize(TARGET_SIZE, resample=Image.BICUBIC)
                
                p_name = f"{js_file.replace('.json', '')}_{obj['id']}_{obj['label']}.jpg"
                patch.save(os.path.join(self.dirs['p2'], p_name))

    def run_phase3_captioning(self):
        print("\n--- PHASE 3: CAPTIONING ---")
        captioner = LocalCaptioner()
        
        # Group by scene (Cell 8 logic)
        for json_file in os.listdir(self.dirs['p1']):
            image_id = json_file.replace('.json', '')
            with open(os.path.join(self.dirs['p1'], json_file), 'r') as f: det_data = json.load(f)
            
            scene_caps = {}
            for obj in det_data['objects']:
                patch_name = f"{image_id}_{obj['id']}_{obj['label']}.jpg"
                patch_path = os.path.join(self.dirs['p2'], patch_name)
                
                if os.path.exists(patch_path):
                    cap = captioner.generate(Image.open(patch_path).convert('RGB'))
                    scene_caps[obj['id']] = {"label_from_detector": obj['label'], "visual_description": cap}
            
            with open(os.path.join(self.dirs['p3'], f"{image_id}_captions.json"), 'w') as f:
                json.dump({"image_id": image_id, "object_captions": scene_caps}, f, indent=4)
        
        del captioner; torch.cuda.empty_cache()

    def run_phase4_reasoning(self):
        print("\n--- PHASE 4: REASONING ---")
        reasoner = QwenReasoner()
        
        for cap_file in os.listdir(self.dirs['p3']):
            with open(os.path.join(self.dirs['p3'], cap_file), 'r') as f: cap_data = json.load(f)
            
            # Load confidences from Phase 1
            det_file = cap_file.replace('_captions.json', '.json')
            with open(os.path.join(self.dirs['p1'], det_file), 'r') as f: det_data = json.load(f)
            conf_map = {obj['id']: obj['confidence'] for obj in det_data['objects']}

            # Construct Context
            context = ""
            for oid, data in cap_data['object_captions'].items():
                conf = conf_map.get(oid, "N/A")
                context += f"- ID: {oid}, Label: {data['label_from_detector']} (Conf: {conf})\n"
                context += f"  Desc: {data['visual_description']}\n"

            result = reasoner.reason(context)
            with open(os.path.join(self.dirs['p4'], cap_file.replace('_captions', '_reasoning')), 'w') as f:
                json.dump(result, f, indent=4)
