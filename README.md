# DeepSeek Prompt Generator

Prompt generator using DeepSeek-R1-Distill-Llama-8B for Stable Diffusion and Flux.

## Setup

```bash
git clone https://github.com/yourusername/deepseek-prompt-gen.git
cd deepseek-prompt-gen
conda env create -f environment.yml
conda activate deepseek-env
```

## Usage

Standard configuration:
```python
from src.prompt_generator import PromptGenerator

generator = PromptGenerator()
prompt = generator.generate_prompt(
    "a mystical forest at twilight",
    style="fantasy",
    variations=3
)
```

8GB GPU optimization:
```python
from src.prompt_generator import PromptGenerator

generator = PromptGenerator(optimize_memory=True)  # Enables 8-bit quantization
prompt = generator.generate_prompt(
    "a mystical forest at twilight",
    style="fantasy",
    variations=2  # Reduced variations for memory efficiency
)
```

## Model Settings
- Temperature: 0.6
- Top-p: 0.95
- No system prompts

Memory Options:
- Standard: 8GB+ VRAM
- Optimized: 4GB+ VRAM with 8-bit quantization

## Development

```bash
# Testing & Formatting
python -m pytest tests/
black src/ tests/

# Environment
conda env update -f environment.yml
conda env export > environment.yml
```

## Requirements
- CUDA-capable GPU (4GB+ VRAM with optimization, 8GB+ recommended)
- 8GB+ RAM
- CUDA 11.7+

## License
Apache License 2.0

See [LICENSE](LICENSE) for the full license text.
