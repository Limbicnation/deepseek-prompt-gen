#!/usr/bin/env python3
"""
DeepSeek Prompt Generator
Generate high-quality prompts for Stable Diffusion using DeepSeek-R1-Distill-Llama-8B
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random


def check_local_model_exists(model_name: str, local_model_dir: str = './local_model_dir') -> Optional[str]:
    """
    Check if a local model exists for the given model name.
    
    Args:
        model_name: The HuggingFace model identifier or local path
        local_model_dir: Directory to check for local models
        
    Returns:
        Path to local model if exists, None otherwise
        
    Raises:
        None - All exceptions are caught and handled gracefully
    """
    try:
        local_dir_path = Path(local_model_dir)
        if not local_dir_path.exists():
            return None
        
        def check_model_files_in_dir(dir_path: Path) -> bool:
            """
            Check if a directory contains valid model files.
            
            Args:
                dir_path: Path to directory to check
                
            Returns:
                True if directory contains valid model files, False otherwise
            """
            try:
                if not dir_path.exists() or not dir_path.is_dir():
                    return False
                
                # Check for config.json (required)
                if not (dir_path / 'config.json').exists():
                    return False
                
                # Check for at least one tokenizer file
                tokenizer_files = ['tokenizer.json', 'tokenizer_config.json']
                has_tokenizer = any((dir_path / f).exists() for f in tokenizer_files)
                if not has_tokenizer:
                    return False
                
                # Check for at least one model file
                model_files = ['pytorch_model.bin', 'model.safetensors']
                has_model = any((dir_path / f).exists() for f in model_files)
                if not has_model:
                    # Also check for model files with different patterns
                    try:
                        model_files_in_dir = [
                            f.name for f in dir_path.iterdir() 
                            if f.is_file() and f.suffix in ('.bin', '.safetensors')
                        ]
                        if not model_files_in_dir:
                            return False
                    except (OSError, PermissionError) as e:
                        print(f"Warning: Could not list files in {dir_path}: {e}")
                        return False
                
                return True
                
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not access directory {dir_path}: {e}")
                return False
        
        # Check if model files are directly in local_model_dir
        if check_model_files_in_dir(local_dir_path):
            print(f"Found local model at: {local_model_dir}")
            return local_model_dir
        
        # If not found directly, try looking in subdirectory based on model name
        # Extract model name from HuggingFace identifier
        # Handle various separator patterns (/, \, etc.)
        if '/' in model_name:
            model_folder = model_name.split('/')[-1]
        elif '\\' in model_name:
            model_folder = model_name.split('\\')[-1]
        else:
            model_folder = model_name
        
        potential_path = local_dir_path / model_folder
        if check_model_files_in_dir(potential_path):
            print(f"Found local model at: {potential_path}")
            return str(potential_path)
        
        return None
        
    except Exception as e:
        print(f"Warning: Error checking for local model: {e}")
        return None


class DeepSeekGenerator:
    def __init__(self, 
                 model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                 optimize_memory: bool = False,
                 device: str = None,
                 max_length: int = 32768,
                 model_dir: str = './models',
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 enforce_reasoning: bool = True,
                 skip_local_check: bool = False):
        """
        Initialize the DeepSeek prompt generator with memory optimizations.
        
        Args:
            model_name: HuggingFace model identifier or local path
            optimize_memory: If True, enables memory optimizations
            device: Specify 'cuda' or 'cpu' (if None, will auto-detect)
            max_length: Maximum token length for generation (default: 32768)
            model_dir: Directory to store/load model files (prevents re-downloading)
            temperature: Temperature for generation (default: 0.6, recommended range: 0.5-0.7)
            top_p: Top-p value for nucleus sampling (default: 0.95)
            enforce_reasoning: Whether to enforce reasoning prefix <think> (default: True)
            skip_local_check: Skip local model pre-check (for performance on network drives)
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Check if model_name is a local path or a HuggingFace model ID
        # First, check if it's an existing local path
        is_local_path = os.path.exists(model_name) and os.path.isdir(model_name)
        
        # If not a local path and local check is not skipped, check if we have a local copy in ./local_model_dir
        if not is_local_path and not skip_local_check:
            local_model_path = check_local_model_exists(model_name, './local_model_dir')
            if local_model_path:
                model_name = local_model_path
                is_local_path = True
                print(f"Using existing local model instead of downloading")
        elif skip_local_check:
            print(f"Skipping local model check (--skip-local-check enabled)")
        
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
        self.temperature = temperature
        self.top_p = top_p
        self.enforce_reasoning = enforce_reasoning
        
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
        
        # Check if bitsandbytes is available for quantization
        use_quantization = False
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes
            use_quantization = True
            print("✓ Using 4-bit quantization (bitsandbytes available)")
        except (ImportError, Exception) as e:
            print(f"⚠️  Quantization disabled (bitsandbytes not available: {e})")
            use_quantization = False
        
        # Base model configuration
        model_config = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        # Add quantization only if bitsandbytes is available
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_config["quantization_config"] = quantization_config
        else:
            # Explicitly disable any quantization from model config
            model_config["quantization_config"] = None
        
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
                # For local models, we need to handle quantization config more carefully
                if not use_quantization:
                    # Load config first and remove any quantization settings
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                    # Completely remove any quantization config from the model's config
                    if hasattr(config, 'quantization_config'):
                        delattr(config, 'quantization_config')
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        config=config,
                        **model_config,
                    )
                else:
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

    def generate_prompt(self, description: str, style: str = "cinematic", 
                       is_math_problem: bool = False) -> str:
        """
        Generate a detailed image generation prompt based on the description.
        
        Args:
            description: Brief description to expand into a detailed prompt
            style: Style template to use (cinematic, anime, etc.)
            is_math_problem: If True, formats as math problem with step-by-step reasoning
            
        Returns:
            A detailed prompt for image generation
        """
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Build the core prompt structure that is common to both cases
        prompt_core = f"""Create a detailed image generation prompt based on this description: "{description}"
Style reference: {self.style_templates.get(style, "Create a high-quality image")}
Include specific details about:
- Composition
- Lighting
- Colors
- Atmosphere
- Technical qualities
Format the response as a single, detailed prompt."""
        
        if is_math_problem:
            # Prepend the math-specific instruction
            math_instruction = "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
            user_prompt = math_instruction + prompt_core
        else:
            user_prompt = prompt_core

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
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=torch.ones_like(inputs["input_ids"])
            )
            
        # Clear memory after generation
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Enforce reasoning prefix if enabled
        if self.enforce_reasoning and not generated_text.strip().startswith("<think>"):
            # If the generated text doesn't start with <think>, add it
            generated_text = f"<think>\n{generated_text.strip()}\n</think>"
        
        return generated_text

    def generate_variations(self, prompt: str, num_variations: int = 3, 
                           style: str = "cinematic", is_math_problem: bool = False) -> List[str]:
        variations = []
        
        for _ in range(num_variations):
            # Generate variation with memory management
            try:
                with torch.inference_mode():
                    variation = self.generate_prompt(prompt, style, is_math_problem)
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
    parser.add_argument("--max-length", type=int, default=32768, help="Maximum length for generation")
    parser.add_argument("--model-dir", type=str, default="./models", 
                        help="Directory to store/load model files (prevents re-downloading)")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model to use (HuggingFace model ID or local path)")
    parser.add_argument("--temperature", type=float, default=0.6, 
                        help="Temperature for generation (recommended: 0.5-0.7)")
    parser.add_argument("--top-p", type=float, default=0.95, 
                        help="Top-p value for nucleus sampling")
    parser.add_argument("--no-reasoning", action="store_true", 
                        help="Disable reasoning prefix enforcement")
    parser.add_argument("--math-problem", action="store_true", 
                        help="Format as math problem with step-by-step reasoning")
    parser.add_argument("--skip-local-check", action="store_true", 
                        help="Skip local model pre-check (for performance on network drives)")
    
    args = parser.parse_args()
    
    try:
        print(f"DeepSeek Prompt Generator - Generating prompts for: {args.description}")
        print(f"Style: {args.style}, Variations: {args.variations}")
        
        generator = DeepSeekGenerator(
            model_name=args.model_name,
            optimize_memory=args.optimize,
            device=args.device,
            max_length=args.max_length,
            model_dir=args.model_dir,
            temperature=args.temperature,
            top_p=args.top_p,
            enforce_reasoning=not args.no_reasoning,
            skip_local_check=args.skip_local_check
        )
        
        base_prompt = generator.generate_prompt(args.description, args.style, args.math_problem)
        variations = generator.generate_variations(args.description, args.variations, 
                                                  args.style, args.math_problem)
        
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