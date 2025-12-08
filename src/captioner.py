import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

class LocalCaptioner:
    def __init__(self, model_id="Salesforce/blip2-opt-2.7b"):
        print(f"Loading Captioning Model ({model_id})...")
        self.processor = Blip2Processor.from_pretrained(model_id)
        # Using float16 as per Cell 8
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, image):
        """Source: Cell 8 Prompt"""
        prompt = "Question: Describe this image in detail, focusing on appearance, color, and material. Answer:"
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
