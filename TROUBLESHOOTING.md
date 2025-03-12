# Troubleshooting Guide

This guide helps resolve common issues with the DeepSeek Prompt Generator.

## Setup Issues

### Error: "No module named 'transformers'"

**Solution**: Install the required packages:
```bash
pip install -r requirements.txt
```

### Error: "ImportError: cannot import name 'BitsAndBytesConfig'"

**Solution**: Update transformers to the latest version:
```bash
pip install -U transformers
```

## Model Loading Issues

### Error: "Incorrect path_or_model_id"

This error occurs when the model path is incorrect or inaccessible.

**Solution**:
1. For HuggingFace models, ensure you're using the correct model ID:
   ```bash
   python deepseek_generator.py "description" --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
   ```

2. For local models, make sure the path exists and contains all model files:
   ```bash
   # Check if the directory exists
   ls -la /path/to/your/model
   
   # Make sure it contains model files (config.json, tokenizer files, etc.)
   ls -la /path/to/your/model | grep -E "config.json|tokenizer"
   ```

3. If using a relative path, make sure it's correct:
   ```bash
   # Use absolute path instead
   python deepseek_generator.py "description" --model-name $(realpath ./models/DeepSeek-Model)
   ```

### Error: "RuntimeError: CUDA out of memory"

**Solution**:
1. Enable memory optimization:
   ```bash
   python deepseek_generator.py "description" --optimize
   ```

2. Reduce the maximum length:
   ```bash
   python deepseek_generator.py "description" --max-length 1024
   ```

3. If all else fails, use CPU instead of GPU:
   ```bash
   python deepseek_generator.py "description" --device cpu
   ```

## File and Directory Issues

### Error: "FileNotFoundError: [Errno 2] No such file or directory: 'data/style_templates.json'"

**Solution**:
Run the setup script to create required directories and files:
```bash
chmod +x create_directories.sh
./create_directories.sh
```

## Class and Module Name Issues

### Error: "NameError: name 'SDPromptGenerator' is not defined"

This happens when there's a mismatch between the class name in the code.

**Solution**:
1. Make sure you're using the correct file:
   ```bash
   # Verify the file content
   grep -n "class" deepseek_generator.py
   
   # Should show: class DeepSeekGenerator
   ```

2. If the class name is wrong, use the provided file:
   ```bash
   # Remove the old file and use the new one
   rm deepseek_generator.py
   # Copy the new file from the artifacts
   ```

## Runtime Issues

### Error: "RecursionError: maximum recursion depth exceeded"

**Solution**:
This is likely due to an infinite loop in the model's generation process.

```bash
# Try a different temperature setting
python deepseek_generator.py "description" --temperature 0.7
```

### No Output or Incomplete Output

**Solution**:
Increase the token generation limit:
```bash
python deepseek_generator.py "description" --max-length 4096
```

## Still Having Issues?

If you continue to experience problems:

1. Check GPU status:
   ```bash
   nvidia-smi
   ```

2. Check CUDA availability in Python:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   ```

3. Try downloading and using a different model:
   ```bash
   python deepseek_generator.py "description" --model-name gpt2
   ```