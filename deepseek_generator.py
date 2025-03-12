#!/usr/bin/env python3
"""
DeepSeek Prompt Generator
Generate high-quality prompts for Stable Diffusion using DeepSeek-R1-Distill-Llama-8B
"""

import argparse
import json
import os
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random


class DeepSeekGenerator:
    def __init__(self, 
                 model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                 optimize_memory: bool = False,
                 device: str = None,
                 max_length: int = 2048,
                 model_dir: str = './models'):
        """
        Initialize the DeepSeek prompt generator with memory optimizations.
        
        Args:
            model_name: HuggingFace model identifier or local path
            optimize_memory: If True, enables memory optimizations
            device: Specify 'cuda' or 'cpu' (if None, will auto-detect)
            max_length: Maximum token length for generation
            model_dir: Directory to store/load model files (prevents re-downloading)
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Check if model_name is a local path or a HuggingFace model ID
        # A path containing a slash that exists on the filesystem is treated as a local path
        is_local_path = os.path.exists(model_name) and os.path.isdir(model_name)
        
        # Configure model caching and settings
        if not is_local_path:
            # Using a HuggingFace model ID
            os.environ['TRANSFORMERS_CACHE'] = model_dir
            self.model_path = model_name  # Just use the model ID directly
            print(f"Using HuggingFace model: {model_name}")
            print(f"Model files will be cached in: {model_dir}")
        else:
            # Using a local path - convert to absolute path
            self.model_path = os.path.abspath(model_name)
            print(f"Using local model from: {self.model_path}")
            # Verify it contains required files
            if not os.path.exists(os.path.join(self.model_path, "config.json")):
                print(f"Warning: config.json not found in {self.model_path}")
                print("This may not be a valid model directory.")
            
        self.max_length = max_length
        
        # Initialize tokenizer with appropriate settings
        print("Loading tokenizer...")
        try:
            if is_local_path:
                # Use direct local path
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
            else:
                # Use HuggingFace model ID
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    cache_dir=model_dir
                )
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("\nIf you're trying to use a local model, make sure:")
            print("1. The path exists and contains model files")
            print("2. You have permission to access the directory")
            print("3. The directory contains tokenizer_config.json or tokenizer.json")
            print("\nIf you're using a HuggingFace model ID, verify the internet connection")
            raise
        
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
        print(f"Loading model {self.model_path}...")
        try:
            if is_local_path:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_config,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_config,
                    cache_dir=model_dir
                )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nIf you're trying to use a local model, make sure:")
            print("1. The directory contains pytorch_model.bin or model.safetensors files")
            print("2. The directory contains config.json")
            print("3. You have enough disk space and RAM/VRAM")
            print("\nIf you're using a HuggingFace model ID, verify the internet connection")
            raise
        
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
                "abstract": "Design an abstract artistic composition",
                "cyberpunk": "Create a cyberpunk-themed image with neon lights, high technology, and urban dystopia",
                "sci-fi": "Generate a science fiction scene with futuristic technology"
            }

    def generate_prompt(self, description: str, style: str = "cinematic") -> str:
        """
        Generate a detailed image generation prompt based on the description.
        
        Args:
            description: Brief description to expand into a detailed prompt
            style: Style template to use (cinematic, anime, etc.)
            
        Returns:
            A detailed prompt for image generation
        """
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
    parser.add_argument("--style", type=str, default="cinematic", help="Style preset to use")
    parser.add_argument("--variations", type=int, default=2, help="Number of variations to generate")
    parser.add_argument("--optimize", action="store_true", help="Enable memory optimization")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], help="Device to use")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum length for generation")
    parser.add_argument("--model-dir", type=str, default="./models", 
                        help="Directory to store/load model files (prevents re-downloading)")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model to use (HuggingFace model ID or local path)")
    
    args = parser.parse_args()
    
    try:
        print(f"DeepSeek Prompt Generator - Generating prompts for: {args.description}")
        print(f"Style: {args.style}, Variations: {args.variations}")
        
        generator = DeepSeekGenerator(
            model_name=args.model_name,
            optimize_memory=args.optimize,
            device=args.device,
            max_length=args.max_length,
            model_dir=args.model_dir
        )
        
        base_prompt = generator.generate_prompt(args.description, args.style)
        variations = generator.generate_variations(args.description, args.variations)
        
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
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()