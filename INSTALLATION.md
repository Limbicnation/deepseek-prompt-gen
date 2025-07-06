# Installation Guide

## Quick Fix for Current Issue

If you're experiencing the `torchvision::nms` error, you need to reinstall with compatible versions.

### Step 1: Uninstall Current Packages
```bash
pip uninstall torch torchvision torchaudio transformers accelerate bitsandbytes -y
```

### Step 2: Install Compatible Versions

**Option A: Using pip (Recommended for immediate fix)**
```bash
# Install PyTorch with CUDA 11.8 (compatible versions)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install compatible HuggingFace and other libraries
pip install transformers==4.45.2 accelerate==0.26.1 bitsandbytes==0.42.0
pip install safetensors==0.4.1 sentencepiece protobuf tokenizers
```

**Option B: Using conda environment (Clean slate)**
```bash
# Remove existing environment and recreate
conda env remove -n deepseek-env
conda env create -f environment.yml
conda activate deepseek-env
```

**Option C: Using requirements.txt**
```bash
pip install -r requirements.txt
```

## Complete Installation Methods

### Method 1: Conda Environment (Recommended)

```bash
# Create environment
conda env create -f environment.yml
conda activate deepseek-env

# Verify installation
python verify_installation.py
```

### Method 2: Virtual Environment + pip

```bash
# Create virtual environment
python -m venv deepseek-venv
source deepseek-venv/bin/activate  # Linux/Mac
# or
deepseek-venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

### Method 3: System-wide Installation

```bash
# Install PyTorch first
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

## Hardware Requirements

- **GPU**: CUDA-compatible GPU with 4GB+ VRAM (8GB+ recommended)
- **CUDA**: Version 11.8 (compatible with PyTorch 2.1.x)
- **RAM**: 8GB+ system RAM
- **Storage**: 10GB+ free space for model files

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # If you have CUDA 12.x, use:
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
   ```

2. **bitsandbytes Compilation Issues**
   ```bash
   # Use pre-compiled wheel
   pip install bitsandbytes==0.42.0 --force-reinstall --no-deps
   ```

3. **Memory Issues**
   ```bash
   # Use memory-optimized mode
   python deepseek_generator.py "prompt" --optimize
   ```

### Verification Commands

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Run full verification
python verify_installation.py
```

## Model Compatibility Note

Based on DeepSeek documentation:
- ✅ **DeepSeek-R1-Distill models** work with HuggingFace transformers
- ⚠️ **DeepSeek-R1 models** may have limited HuggingFace support

This project uses `DeepSeek-R1-Distill-Llama-8B` which should be fully compatible.

## Performance Optimization

### For 4GB VRAM GPUs:
```bash
python deepseek_generator.py "prompt" --optimize --max-length 16384
```

### For Network Drives:
```bash
python deepseek_generator.py "prompt" --skip-local-check
```

### For CPU-only Systems:
```bash
python deepseek_generator.py "prompt" --device cpu
```

## Next Steps

After successful installation:

1. Download the model: `python download_model.py --output-dir ./local_model_dir`
2. Test generation: `python deepseek_generator.py "a mystical forest" --output test.json`
3. Verify output: Check `test.json` for generated prompts