import argparse
import json
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random
import os

class DeepSeekGenerator:
    def __init__(self, 
                 model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                 optimize_memory: bool = False,
                 device: str = None,
                 max_length: int = 2048):  # Increased default max_length
        """
        Initialize the DeepSeek prompt generator with memory optimizations.
        
        Args:
            model_name: HuggingFace model identifier
            optimize_memory: If True, enables memory optimizations
            device: Specify 'cuda' or 'cpu' (if None, will auto-detect)
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configure model caching and settings
        os.environ['TRANSFORMERS_CACHE'] = './models'
        self.max_length = max_length
        
        # Initialize tokenizer with low memory footprint
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Base model configuration
        model_config = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "quantization_config": quantization_config,
            "device_map": "auto"
        }
        
        # Additional memory optimizations
        if optimize_memory:
            model_config.update({
                "max_memory": {0: "7GiB"},  # Reserve 1GB for system
                "offload_folder": "offload",
                "low_cpu_mem_usage": True
            })
            max_new_tokens = 1024  # Allow for reasonable response length
        else:
            max_new_tokens = 2048  # More tokens when not memory constrained
        self.max_new_tokens = max_new_tokens
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_config
        )
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Load style templates from file
        self.load_style_templates()
        
    def load_style_templates(self):
        """Load style templates from JSON file"""
        try:
            with open('data/style_templates.json', 'r') as f:
                self.style_templates = json.load(f)
        except FileNotFoundError:
            # Fallback to default templates
            self.style_templates = {
                "cinematic": "Create a cinematic scene with dramatic lighting and composition",
                "anime": "Design an anime-style illustration with vibrant colors",
                "photorealistic": "Generate a photorealistic image with high detail",
                "fantasy": "Create a fantasy-themed illustration with magical elements",
                "abstract": "Design an abstract artistic composition"
            }

    def generate_prompt(self, description: str, style: str = "cinematic") -> str:
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        user_prompt = f"""<think>\nCreate a detailed image generation prompt based on this description: "{description}"
        Style reference: {self.style_templates.get(style, "Create a high-quality image")}
        Include specific details about:
        - Composition
        - Lighting
        - Colors
        - Atmosphere
        - Technical qualities
        Format the response as a single, detailed prompt."""

        inputs = self.tokenizer(
            user_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,  # Use max_new_tokens instead of max_length
                temperature=0.6,  # As recommended in docs
                top_p=0.95,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=torch.ones_like(inputs["input_ids"])  # Explicitly set attention mask
            )
            
        # Clear memory after generation
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_variations(self, prompt: str, num_variations: int = 3) -> List[str]:
        variations = []
        
        for _ in range(num_variations):
            # Generate variation with memory management
            try:
                with torch.inference_mode():
                    variation = self.generate_prompt(prompt)
                    variations.append(self.enhance_prompt(variation))
                
                # Clear memory after each variation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                print(f"Memory error during generation: {e}")
                # Reduce batch size or complexity if needed
                break
                
        return variations

    def enhance_prompt(self, base_prompt: str) -> str:
        """Enhance prompt with quality modifiers"""
        quality_modifiers = [
            "highly detailed", "8k uhd", "masterpiece",
            "professional lighting", "sharp focus"
        ]
        selected_modifiers = random.sample(quality_modifiers, 2)  # Reduced from 3
        return f"{base_prompt}, {', '.join(selected_modifiers)}"

def main():
    parser = argparse.ArgumentParser(description="Generate prompts using DeepSeek-R1")
    parser.add_argument("description", type=str, help="Image description")
    parser.add_argument("--style", type=str, default="cinematic")
    parser.add_argument("--variations", type=int, default=2)  # Reduced from 3
    parser.add_argument("--optimize", action="store_true", help="Enable memory optimization")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], help="Device to use")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum length for generation")
    
    args = parser.parse_args()
    
    try:
        generator = DeepSeekGenerator(
            optimize_memory=args.optimize,
            device=args.device,
            max_length=args.max_length
        )
        
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
                
    except Exception as e:
        print(f"Error during execution: {e}")
        
if __name__ == "__main__":
    main()