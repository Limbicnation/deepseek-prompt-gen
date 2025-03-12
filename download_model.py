#!/usr/bin/env python3
"""
Download and save a DeepSeek model locally to avoid redownloading it each time.
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser(description="Download a DeepSeek model to a local directory")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="HuggingFace model ID to download")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the model")
    
    args = parser.parse_args()
    
    model_id = args.model
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading model {model_id} to {output_dir}...")
    
    # Download and save tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    # Configure quantization (optional - just for checking model structure)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Download and save model
    print("Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Save the model without quantization for maximum compatibility
    print("Saving model...")
    model.save_pretrained(output_dir)
    
    print(f"\nModel successfully downloaded to {output_dir}")
    print(f"You can use it with: python deepseek_generator.py \"prompt\" --model-name {output_dir}")

if __name__ == "__main__":
    main()