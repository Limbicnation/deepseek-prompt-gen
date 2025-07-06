# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based prompt generator using DeepSeek-R1-Distill-Llama-8B for creating high-quality prompts for Stable Diffusion and Flux image generation models. The project supports both standard and memory-optimized modes for different hardware configurations.

## Development Commands

### Environment Setup
```bash
# Method 1: Using conda environment
conda env create -f environment.yml
conda activate deepseek-env

# Method 2: Using pip
conda create -n deepseek python=3.10
conda activate deepseek
pip install -r requirements.txt
```

### Running the Generator
```bash
# Basic usage
python deepseek_generator.py "a mystical forest" --style fantasy

# Memory-optimized mode (for 4GB+ VRAM)
python deepseek_generator.py "a mystical forest" --style fantasy --optimize

# Advanced usage with custom settings
python deepseek_generator.py "a mystical forest" \
    --style fantasy \
    --optimize \
    --device cuda \
    --max-length 32768 \
    --variations 2 \
    --temperature 0.6 \
    --top-p 0.95 \
    --output prompts.json

# Math problem formatting
python deepseek_generator.py "solve this math problem" \
    --math-problem \
    --temperature 0.6

# Disable reasoning enforcement
python deepseek_generator.py "a mystical forest" \
    --no-reasoning
```

### Model Management
```bash
# Download model for offline use
python download_model.py --output-dir ./local_model_dir

# Use local model
python deepseek_generator.py "prompt" --model-name /path/to/local_model_dir --optimize
```

### Testing
```bash
# Run tests (using pytest from requirements.txt)
pytest

# Code formatting
black .
```

## Code Architecture

### Core Components

1. **DeepSeekGenerator Class** (`deepseek_generator.py:16-142`)
   - Main class handling model initialization and prompt generation
   - Supports both HuggingFace model IDs and local model paths
   - Implements memory optimization strategies for different VRAM configurations
   - Key methods:
     - `__init__()`: Model loading with quantization and memory optimization
     - `generate_prompt()`: Core prompt generation with style templates
     - `generate_variations()`: Creates multiple prompt variations
     - `enhance_prompt()`: Adds quality modifiers to prompts

2. **Model Loading Strategy** (`deepseek_generator.py:37-134`)
   - Automatic detection of local vs HuggingFace models
   - 4-bit quantization using BitsAndBytesConfig for memory efficiency
   - Configurable memory limits and offloading for low-VRAM systems
   - Error handling for model loading failures

3. **Style System** (`deepseek_generator.py:143-158`)
   - Template-based style system with fallback defaults
   - Supports: cinematic, anime, photorealistic, fantasy, abstract, cyberpunk, sci-fi
   - Extensible through `data/style_templates.json` (optional)

4. **Memory Management** (`deepseek_generator.py:100-110`, `171-174`, `205-209`)
   - CUDA cache clearing before/after generation
   - Memory optimization flags for 4GB+ VRAM systems
   - Gradient checkpointing for reduced memory usage
   - Configurable token limits based on memory constraints

### Key Files

- `deepseek_generator.py`: Main generator class and CLI interface
- `download_model.py`: Utility for downloading models locally
- `requirements.txt`: Python dependencies
- `environment.yml`: Conda environment specification
- `TROUBLESHOOTING.md`: Common issues and solutions

## Official Usage Recommendations

Based on the DeepSeek-R1 official documentation, the following settings are recommended:

### Generation Parameters
- **Temperature**: 0.6 (range: 0.5-0.7) - prevents endless repetitions
- **Top-p**: 0.95 - for nucleus sampling
- **Max Length**: 32,768 tokens - official maximum generation length
- **Reasoning Enforcement**: Enabled by default - ensures `<think>` prefix

### Prompt Guidelines
- Avoid system prompts - all instructions should be in the user prompt
- For math problems: use `--math-problem` flag for step-by-step reasoning
- Multiple tests and averaging recommended for evaluation
- The model may bypass thinking patterns, so reasoning enforcement is recommended

### Command Line Examples
```bash
# Standard usage with recommended settings
python deepseek_generator.py "a mystical forest" --temperature 0.6 --top-p 0.95

# Math problem with step-by-step reasoning
python deepseek_generator.py "solve 2x + 3 = 7" --math-problem
```

## Hardware Requirements

- **Standard Mode**: 8GB+ VRAM
- **Optimized Mode**: 4GB+ VRAM (use `--optimize` flag)
- **CPU Fallback**: Available but significantly slower
- CUDA 11.7+ required for GPU acceleration

## Development Notes

### Memory Optimization
- Always use `--optimize` flag when working with local models
- The generator automatically clears CUDA cache to prevent OOM errors
- Memory limits are configurable through the `max_memory` parameter
- Gradient checkpointing is enabled automatically for memory efficiency

### Model Paths
- Local models: Must contain `config.json` and tokenizer files
- HuggingFace models: Cached in `./models` directory by default
- Use absolute paths for local models to avoid loading issues

### Error Handling
- Comprehensive error messages for model loading failures
- Memory error detection and recovery during generation
- Validation of model directory structure for local models

### Style Templates
- Default styles are hardcoded as fallback
- Optional `data/style_templates.json` for custom styles
- Templates guide the prompt generation process for different artistic styles