import argparse
import json
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

class DeepSeekGenerator:
    def __init__(self, 
                 model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                 optimize_memory: bool = False):
        """
        Initialize the DeepSeek prompt generator.
        
        Args:
            model_name: HuggingFace model identifier
            optimize_memory: If True, enables 8-bit quantization for 4GB+ VRAM
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        model_config = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16
        }
        
        if optimize_memory:
            try:
                import bitsandbytes as bnb
                model_config["load_in_8bit"] = True
                self.max_length = 256  # Reduced for memory optimization
            except ImportError:
                print("Warning: bitsandbytes not found. Running in standard mode.")
                self.max_length = 512
        else:
            self.max_length = 512
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
        
        if optimize_memory:
            self.model.gradient_checkpointing_enable()
        
        # Style templates
        self.style_templates = {
            "cinematic": "Create a cinematic scene with dramatic lighting and composition",
            "anime": "Design an anime-style illustration with vibrant colors",
            "photorealistic": "Generate a photorealistic image with high detail and natural lighting",
            "fantasy": "Create a fantasy-themed illustration with magical elements",
            "abstract": "Design an abstract artistic composition"
        }
        
        # Quality modifiers
        self.quality_modifiers = [
            "highly detailed", "8k uhd", "masterpiece", "professional lighting",
            "sharp focus", "high resolution", "stunning", "beautiful lighting"
        ]

    def generate_prompt(self, description: str, style: str = "cinematic") -> str:
        user_prompt = f"""Create a detailed image generation prompt based on this description: "{description}"
        Style reference: {self.style_templates.get(style, "Create a high-quality image")}
        Include specific details about:
        - Composition
        - Lighting
        - Colors
        - Atmosphere
        - Technical qualities
        Format the response as a single, detailed prompt."""

        inputs = self.tokenizer(user_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_length,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                use_cache=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def enhance_prompt(self, base_prompt: str) -> str:
        selected_modifiers = random.sample(self.quality_modifiers, 3)
        return f"{base_prompt}, {', '.join(selected_modifiers)}"

    def generate_variations(self, prompt: str, num_variations: int = 3) -> List[str]:
        variations = []
        for _ in range(num_variations):
            with torch.inference_mode():
                temp = random.uniform(0.5, 0.7)
                variation = self.generate_prompt(prompt)
                variations.append(self.enhance_prompt(variation))
            
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
        return variations

def main():
    parser = argparse.ArgumentParser(description="Generate prompts using DeepSeek-R1")
    parser.add_argument("description", type=str, help="Image description")
    parser.add_argument("--style", type=str, default="cinematic", 
                        choices=list(DeepSeekGenerator().style_templates.keys()))
    parser.add_argument("--variations", type=int, default=3)
    parser.add_argument("--optimize", action="store_true", help="Enable memory optimization")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    generator = DeepSeekGenerator(optimize_memory=args.optimize)
    base_prompt = generator.generate_prompt(args.description, args.style)
    variations = generator.generate_variations(base_prompt, args.variations)
    
    result = {
        "description": args.description,
        "style": args.style,
        "base_prompt": base_prompt,
        "variations": variations
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print("\nGenerated Prompts:")
        print("================")
        print(f"\nBase Prompt:\n{base_prompt}")
        print("\nVariations:")
        for i, var in enumerate(variations, 1):
            print(f"\n{i}. {var}")

if __name__ == "__main__":
    main()