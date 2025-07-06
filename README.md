# DeepSeek Prompt Generator

Prompt generator using DeepSeek-R1-Distill-Llama-8B for Stable Diffusion and Flux.

## Installation

**Important:** For best results and to avoid dependency conflicts, use the conda-based installation method.

### Recommended: Conda Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deepseek-prompt-gen.git
cd deepseek-prompt-gen

# Method 1: Using environment.yml (Recommended)
conda env create -f environment.yml
conda activate deepseek-env

# Method 2: Manual conda setup
conda create -n deepseek python=3.10
conda activate deepseek
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install transformers accelerate tokenizers -c conda-forge
pip install safetensors sentencepiece protobuf huggingface-hub

# Verify installation
python verify_installation.py
```

### Alternative: pip Installation

```bash
# Only use if conda is not available
pip install -r requirements.txt
```

**Note:** The pip method may require manual CUDA library management. Use conda for the most reliable installation.

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

## Model Management

### Using HuggingFace Models (Default)
By default, the script will download the model from HuggingFace and cache it in the `./models` directory:

```bash
python deepseek_generator.py "a mystical forest"
```

### Downloading Models for Offline Use
To avoid re-downloading models, you can use the included `download_model.py` script:

```bash
python download_model.py --output-dir ./local_model_dir
```

### Using Local Models
To use a previously downloaded model:

```bash
python deepseek_generator.py "prompt" --model-name /path/to/local_model_dir --optimize
```

**Important:** Always use the `--optimize` flag when using local models to avoid VRAM issues.

## Model Settings
- Temperature: 0.6 (optimized for coherent outputs)
- Top-p: 0.95
- Available styles: cinematic, anime, photorealistic, fantasy, abstract
- Max generation length: 2048 tokens (configurable)

## Advanced Configuration

- `--model-name`: Path to a local model or HuggingFace model ID
- `--device`: Choose between 'cuda' or 'cpu'
- `--max-length`: Set maximum token length (default: 2048)
- `--variations`: Number of prompt variations to generate (default: 2)
- `--optimize`: Enable memory optimizations for lower VRAM usage
- `--model-dir`: Directory to cache downloaded models (default: ./models)

## Requirements

### System Requirements
- Python 3.10+
- CUDA-capable GPU with 4GB+ VRAM (8GB+ recommended)
- CUDA 12.1+ (12.6 tested and working)
- Ubuntu/Linux (WSL2 supported)

### Key Dependencies
- PyTorch 2.1+ with CUDA 12.1 support
- transformers 4.46+ (with tokenizers 0.20+)
- accelerate 0.25+
- bitsandbytes 0.43+ (optional, for quantization)

### Verified Working Configuration
- CUDA 12.1/12.6
- PyTorch 2.5.1+cu121
- transformers 4.46+
- tokenizers 0.20+

## Troubleshooting

### Installation Issues

**Tensor Shape Mismatch Errors:**
```bash
# Clean up corrupted installations
rm -rf ./local_model_dir ./models
conda remove pytorch torchvision transformers --yes
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**tokenizers Version Conflicts:**
```bash
# Ensure compatible versions
pip install "transformers>=4.46.0" "tokenizers>=0.20.0,<0.22.0"
```

**bitsandbytes CUDA Issues:**
```bash
# Skip quantization if problematic
python deepseek_generator.py "prompt" --output test.json
# The generator will automatically disable quantization if bitsandbytes is unavailable
```

### Runtime Issues

**CUDA Out-of-Memory Errors:**
- Enable `--optimize` flag for 4-bit quantization
- Reduce `--max-length` (try 1024 or 512)
- Lower number of `--variations`
- Use `--device cpu` as last resort (much slower)

**Model Loading Issues:**
- Use `--skip-local-check` to bypass corrupted local models
- Delete `./local_model_dir` and let the system download fresh models
- Verify CUDA installation: `nvidia-smi` and `nvcc --version`

**Performance on Network Drives:**
- Use `--skip-local-check` flag for better performance
- Download models to local storage when possible

## License
Apache License 2.0