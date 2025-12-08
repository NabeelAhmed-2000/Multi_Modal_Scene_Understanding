import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

class ObjectDetector:
    def __init__(self, confidence_threshold=0.7):
        print("Loading Object Detector (DETR-ResNet-50)...")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.threshold = confidence_threshold

    def detect(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.threshold)[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_name = self.model.config.id2label[label.item()]
            
            detections.append({
                "label": label_name,
                "confidence": round(score.item(), 3),
                "bbox": box
            })
        
        return detections
