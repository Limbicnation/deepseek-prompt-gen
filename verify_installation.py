#!/usr/bin/env python3
"""
Installation verification script for DeepSeek Prompt Generator
"""

import sys
import subprocess
import importlib.util
from typing import Dict, List, Tuple, Optional
import platform

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)"

def check_package_import(package_name: str, expected_version: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a package can be imported and optionally check version"""
    try:
        if importlib.util.find_spec(package_name) is None:
            return False, f"✗ {package_name} not found"
        
        module = importlib.import_module(package_name)
        
        if hasattr(module, '__version__'):
            version = module.__version__
            if expected_version and not version.startswith(expected_version):
                return False, f"✗ {package_name} {version} (expected {expected_version}+)"
            return True, f"✓ {package_name} {version}"
        else:
            return True, f"✓ {package_name} (version unknown)"
            
    except ImportError as e:
        return False, f"✗ {package_name} import failed: {str(e)}"
    except Exception as e:
        return False, f"✗ {package_name} error: {str(e)}"

def check_torch_cuda() -> Tuple[bool, str]:
    """Check PyTorch CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            cuda_version = torch.version.cuda
            return True, f"✓ CUDA available: {device_count} device(s), {device_name}, CUDA {cuda_version}"
        else:
            return False, "✗ CUDA not available (will use CPU)"
    except Exception as e:
        return False, f"✗ CUDA check failed: {str(e)}"

def check_model_compatibility() -> Tuple[bool, str]:
    """Check if transformers can load the model architecture"""
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        # Test with a small config to see if LlamaForCausalLM is available
        config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")  # Small test model
        
        # Check if we can import the model class we need
        from transformers import LlamaForCausalLM
        return True, "✓ Model architecture compatible"
        
    except ImportError as e:
        return False, f"✗ Model compatibility issue: {str(e)}"
    except Exception as e:
        return False, f"✗ Model check failed: {str(e)}"

def check_system_resources() -> List[Tuple[bool, str]]:
    """Check system resources"""
    results = []
    
    try:
        import psutil
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 8:
            results.append((True, f"✓ System RAM: {memory_gb:.1f}GB"))
        else:
            results.append((False, f"⚠ System RAM: {memory_gb:.1f}GB (8GB+ recommended)"))
        
        # Disk space check
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        if disk_gb >= 10:
            results.append((True, f"✓ Free disk space: {disk_gb:.1f}GB"))
        else:
            results.append((False, f"⚠ Free disk space: {disk_gb:.1f}GB (10GB+ recommended)"))
            
    except ImportError:
        results.append((False, "⚠ psutil not available (install with: pip install psutil)"))
    except Exception as e:
        results.append((False, f"⚠ System resource check failed: {str(e)}"))
    
    return results

def check_local_model() -> Tuple[bool, str]:
    """Check if local model exists"""
    try:
        from deepseek_generator import check_local_model_exists
        
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        local_path = check_local_model_exists(model_name, './local_model_dir')
        
        if local_path:
            return True, f"✓ Local model found at: {local_path}"
        else:
            return False, "✗ Local model not found (run: python download_model.py --output-dir ./local_model_dir)"
            
    except Exception as e:
        return False, f"✗ Local model check failed: {str(e)}"

def run_simple_test() -> Tuple[bool, str]:
    """Run a simple generation test"""
    try:
        # This is a basic import test - actual model loading would require the model files
        from deepseek_generator import DeepSeekGenerator
        return True, "✓ DeepSeek generator can be imported"
    except Exception as e:
        return False, f"✗ DeepSeek generator test failed: {str(e)}"

def main():
    """Main verification function"""
    print("🔍 DeepSeek Prompt Generator - Installation Verification")
    print("=" * 60)
    
    all_passed = True
    
    # Core checks
    print("\n📋 Core Requirements:")
    checks = [
        check_python_version(),
        check_package_import("torch", "2.1"),
        check_package_import("torchvision", "0.16"),
        check_package_import("transformers", "4.4"),
        check_package_import("accelerate"),
        check_package_import("bitsandbytes"),
        check_package_import("safetensors"),
        check_torch_cuda(),
        check_model_compatibility(),
    ]
    
    for passed, message in checks:
        print(f"  {message}")
        if not passed:
            all_passed = False
    
    # System resources
    print("\n💻 System Resources:")
    resource_checks = check_system_resources()
    for passed, message in resource_checks:
        print(f"  {message}")
        # Don't fail on resource warnings
    
    # Application-specific checks
    print("\n🎯 Application Checks:")
    app_checks = [
        check_local_model(),
        run_simple_test(),
    ]
    
    for passed, message in app_checks:
        print(f"  {message}")
        # Don't fail on missing local model
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ Installation verification PASSED!")
        print("\n📚 Next steps:")
        print("  1. Download model: python download_model.py --output-dir ./local_model_dir")
        print("  2. Test generation: python deepseek_generator.py \"a mystical forest\" --output test.json")
        print("  3. Check output: cat test.json")
    else:
        print("❌ Installation verification FAILED!")
        print("\n🔧 Recommended fixes:")
        print("  1. Follow INSTALLATION.md for compatible versions")
        print("  2. Uninstall and reinstall with: pip install -r requirements.txt")
        print("  3. Check CUDA compatibility with: nvidia-smi")
    
    print(f"\n🖥️  System Info: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)