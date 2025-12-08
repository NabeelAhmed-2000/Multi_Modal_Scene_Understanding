import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

class QwenReasoner:
    """Implements Phase 4 Logic (Cell 11)"""
    def __init__(self):
        print("Loading Qwen2.5-7B-Instruct...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

    def reason(self, context_str):
        system_prompt = """You are an advanced Visual Reasoning Agent.
Your goal is to synthesize a global description from conflicting object reports.
CRITICAL INSTRUCTIONS:
1. INCLUDE DETAILS: Do not summarize generic categories.
2. RESOLVE CONFLICTS: Use confidence scores.
3. OUTPUT FORMAT: Generate a natural, descriptive paragraph."""

        user_prompt = f"""Here are the object observations:\n{context_str}\n\nPlease output your analysis in this JSON format:
{{
 "conflicts_detected": [],
 "reasoning_steps": "...",
 "final_global_caption": "..."
}}"""
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512, temperature=0.7)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._parse_json(response)

    def _parse_json(self, response):
        try:
            if "```json" in response: json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response: json_str = response.split("```")[1].strip()
            else: json_str = response
            return json.loads(json_str)
        except:
            return {"raw_response": response}
