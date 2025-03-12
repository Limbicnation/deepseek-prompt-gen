# DeepSeek Prompt Generator

Prompt generator using DeepSeek-R1-Distill-Llama-8B for Stable Diffusion and Flux.

## Installation

Choose one of these installation methods:

```bash
# Method 1: Using environment.yml
git clone https://github.com/yourusername/deepseek-prompt-gen.git
cd deepseek-prompt-gen
conda env create -f environment.yml
conda activate deepseek-env

# Method 2: Using requirements.txt
git clone https://github.com/yourusername/deepseek-prompt-gen.git
cd deepseek-prompt-gen
conda create -n deepseek python=3.10
conda activate deepseek
pip install -r requirements.txt
```

## Usage

```python
from deepseek_generator import DeepSeekGenerator

# Standard usage (8GB+ VRAM)
generator = DeepSeekGenerator()
prompt = generator.generate_prompt(
    "a mystical forest at twilight",
    style="fantasy"
)

# Memory-optimized (4GB+ VRAM)
generator = DeepSeekGenerator(optimize_memory=True)
prompt = generator.generate_prompt(
    "a mystical forest at twilight",
    style="fantasy"
)
```

## Command Line Usage

```bash
# Standard mode
python deepseek_generator.py "a mystical forest" --style fantasy

# Memory-optimized mode
python deepseek_generator.py "a mystical forest" --style fantasy --optimize

# Save output to file
python deepseek_generator.py "a mystical forest" --output prompts.json

# Advanced options
python deepseek_generator.py "a mystical forest" \
    --style fantasy \
    --optimize \
    --device cuda \
    --max-length 2048 \
    --variations 2
```

## Model Settings
- Temperature: 0.6 (optimized for coherent outputs)
- Top-p: 0.95
- Available styles: cinematic, anime, photorealistic, fantasy, abstract
- Max generation length: 2048 tokens (configurable)

## Advanced Configuration

- `--device`: Choose between 'cuda' or 'cpu'
- `--max-length`: Set maximum token length (default: 2048)
- `--variations`: Number of prompt variations to generate (default: 2)
- `--optimize`: Enable memory optimizations for lower VRAM usage

## Requirements
- Python 3.10+
- CUDA-capable GPU:
  - Standard: 8GB+ VRAM
  - Optimized: 4GB+ VRAM
- CUDA 11.7+
- Key dependencies:
    - torch>=2.0.0
    - transformers>=4.36.0
    - accelerate>=0.24.0
    - bitsandbytes>=0.41.1

## Troubleshooting

If encountering CUDA out-of-memory errors:

- Enable `--optimize` flag
- Reduce `--max-length`
- Lower number of `--variations`
 
For CPU-only systems, use `--device cpu`

## License
Apache License 2.0
