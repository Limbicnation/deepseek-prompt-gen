# DeepSeek Prompt Generator

Prompt generator using DeepSeek-R1-Distill-Llama-8B for Stable Diffusion and Flux.

## Installation

Choose one of these installation methods:

```bash
git clone https://github.com/yourusername/deepseek-prompt-gen.git
cd deepseek-prompt-gen
conda env create -f environment.yml
conda activate deepseek-env

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
```

## Model Settings
- Temperature: 0.6
- Top-p: 0.95
- Available styles: cinematic, anime, photorealistic, fantasy, abstract

## Requirements
- Python 3.10+
- CUDA-capable GPU:
  - Standard: 8GB+ VRAM
  - Optimized: 4GB+ VRAM
- CUDA 11.7+

## License
Apache License 2.0