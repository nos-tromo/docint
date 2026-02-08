# Migration Guide: Ollama to Llama.cpp

This guide helps you migrate from the old Ollama-based setup to the new Llama.cpp-based implementation.

## Overview

The application has been migrated from using Ollama as an external service to using llama.cpp directly for LLM inference. This change provides:

- **Self-contained deployment**: No external LLM service required
- **Better GPU control**: Granular control over GPU layer offloading
- **Wider model support**: Use any GGUF-quantized model
- **Improved memory management**: More efficient resource usage

## Breaking Changes

### 1. Model Format Change

**Before (Ollama):**
- Models: `gpt-oss:20b`, `qwen3-vl:8b`
- Storage: Managed by Ollama service
- Format: Ollama's internal format

**After (Llama.cpp):**
- Models: `model-q4_k_m.gguf`, `model-vision-q4_k_m.gguf`
- Storage: `~/.cache/llama.cpp/` (or `$LLAMA_CPP_CACHE`)
- Format: GGUF (GGML Universal Format)

### 2. Environment Variables

**Removed:**
- `OLLAMA_HOST`
- `OLLAMA_NUM_PARALLEL`
- `OLLAMA_MAX_LOADED_MODELS`
- `OLLAMA_THINKING`
- `OLLAMA_CTX_WINDOW` → Replaced with `LLAMA_CPP_CTX_WINDOW`
- `OLLAMA_TEMPERATURE` → Replaced with `LLAMA_CPP_TEMPERATURE`
- `OLLAMA_TOP_K` → Replaced with `LLAMA_CPP_TOP_K`
- `OLLAMA_TOP_P` → Replaced with `LLAMA_CPP_TOP_P`

**Added:**
- `LLAMA_CPP_CACHE`: Directory for GGUF models (default: `~/.cache/llama.cpp`)
- `LLAMA_CPP_N_GPU_LAYERS`: GPU layer offloading (-1 = all, 0 = CPU only)
- `LLAMA_CPP_CTX_WINDOW`: Context window size (default: 8192)
- `LLAMA_CPP_TEMPERATURE`: Sampling temperature (default: 0.0)
- `LLAMA_CPP_TOP_K`: Top-k sampling (default: 40)
- `LLAMA_CPP_TOP_P`: Top-p sampling (default: 0.95)
- `LLAMA_CPP_REPEAT_PENALTY`: Repetition penalty (default: 1.1)
- `LLAMA_CPP_SEED`: Random seed (default: 42)

### 3. Docker Compose Changes

**Before:**
```yaml
services:
  backend:
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      - OLLAMA_HOST=http://ollama-net:11434
  
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama-models:/root/.ollama
```

**After:**
```yaml
services:
  backend:
    # No ollama dependency
    environment:
      - LLAMA_CPP_N_GPU_LAYERS=0  # or -1 for GPU
    volumes:
      - model-cache:/root/.cache  # GGUF models in here
  
  # ollama service removed
```

### 4. Information Extraction Engine

If using LLM-based information extraction, update the engine setting:

**Before:**
```env
IE_ENGINE=ollama
```

**After:**
```env
IE_ENGINE=llama_cpp
```

Or continue using GLiNER (recommended, no LLM required):
```env
IE_ENGINE=gliner
```

## Migration Steps

### Step 1: Update Dependencies

```bash
# Pull the latest changes
git pull origin main

# Regenerate lock file and sync dependencies
uv lock
uv sync
```

### Step 2: Obtain GGUF Models

Download GGUF quantized models from Hugging Face or convert existing models:

**Option A: Download Pre-quantized Models**

Example sources:
- [TheBloke's GGUF models](https://huggingface.co/TheBloke) (many popular models)
- [NousResearch](https://huggingface.co/NousResearch)
- Search Hugging Face for "GGUF"

```bash
# Create cache directory
mkdir -p ~/.cache/llama.cpp

# Example: Download a model (replace with your preferred model)
cd ~/.cache/llama.cpp
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
mv mistral-7b-instruct-v0.2.Q4_K_M.gguf model-q4_k_m.gguf

# For vision models, download a multimodal GGUF model
# wget https://huggingface.co/<user>/<repo>/resolve/main/<model>.gguf
# mv <model>.gguf model-vision-q4_k_m.gguf
```

**Option B: Convert Models**

If you have a model in another format, convert it using llama.cpp tools:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert model (example for HF models)
python convert.py /path/to/your/model --outtype q4_k_m --outfile ~/.cache/llama.cpp/model-q4_k_m.gguf
```

### Step 3: Update Configuration

Update your `.env` file:

**Before:**
```env
LLM=gpt-oss:20b
VLM=qwen3-vl:8b
OLLAMA_NUM_PARALLEL=4
OLLAMA_MAX_LOADED_MODELS=2
```

**After:**
```env
LLM=model-q4_k_m.gguf
VLM=model-vision-q4_k_m.gguf
LLAMA_CPP_N_GPU_LAYERS=-1  # -1 for all layers on GPU, 0 for CPU only
LLAMA_CPP_CTX_WINDOW=8192
```

### Step 4: Verify Installation

```bash
# Load models and verify setup
uv run load-models

# Run tests
uv run pytest

# Start the application
uv run uvicorn docint.core.api:app --reload
```

### Step 5: Update Docker Setup (if using Docker)

```bash
# Remove old volumes (optional, saves space)
docker volume rm docint_ollama-models

# Update .env.docker
cp .env.docker.example .env.docker
# Edit .env.docker with your GGUF model names

# Place GGUF models in the mounted cache directory
mkdir -p ~/docint/.cache/llama.cpp
cp /path/to/your/models/*.gguf ~/docint/.cache/llama.cpp/

# Rebuild and restart
docker compose --profile cpu build
docker compose --profile cpu up
```

## GPU Acceleration

### CUDA (NVIDIA)

For GPU acceleration with NVIDIA GPUs:

```bash
# Install with CUDA support (if not using Docker)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Or use the GPU Docker profile
docker compose --profile gpu up
```

Set environment variable:
```env
LLAMA_CPP_N_GPU_LAYERS=-1  # Offload all layers to GPU
```

### Metal (Apple Silicon)

For GPU acceleration on macOS with Apple Silicon:

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Set environment variable:
```env
LLAMA_CPP_N_GPU_LAYERS=-1  # Enable GPU acceleration
```

### CPU Only

For CPU-only inference:

```bash
# Standard installation (OpenBLAS support)
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Set environment variable:
```env
LLAMA_CPP_N_GPU_LAYERS=0  # CPU only
```

## Performance Tuning

### Context Window

Larger context windows use more memory but allow longer conversations:

```env
LLAMA_CPP_CTX_WINDOW=8192   # Default
LLAMA_CPP_CTX_WINDOW=16384  # Larger context
LLAMA_CPP_CTX_WINDOW=32768  # Very large (requires more RAM/VRAM)
```

### GPU Layer Offloading

Fine-tune GPU offloading for optimal performance:

```env
LLAMA_CPP_N_GPU_LAYERS=-1   # All layers (fastest, most VRAM)
LLAMA_CPP_N_GPU_LAYERS=32   # Partial offload (balanced)
LLAMA_CPP_N_GPU_LAYERS=0    # CPU only (slowest, no VRAM)
```

### Sampling Parameters

Adjust for different use cases:

**Deterministic (reproducible):**
```env
LLAMA_CPP_TEMPERATURE=0.0
LLAMA_CPP_TOP_K=1
LLAMA_CPP_TOP_P=0.0
```

**Creative (varied outputs):**
```env
LLAMA_CPP_TEMPERATURE=0.8
LLAMA_CPP_TOP_K=40
LLAMA_CPP_TOP_P=0.95
```

## Troubleshooting

### Issue: Model not found

**Error:** `Model file not found: ~/.cache/llama.cpp/model-q4_k_m.gguf`

**Solution:**
1. Verify the model file exists: `ls -lh ~/.cache/llama.cpp/`
2. Check the `LLM` environment variable matches the filename
3. Ensure `LLAMA_CPP_CACHE` points to the correct directory

### Issue: Out of memory

**Error:** `CUDA out of memory` or similar

**Solution:**
1. Reduce `LLAMA_CPP_N_GPU_LAYERS` (try 32, 16, or 0)
2. Reduce `LLAMA_CPP_CTX_WINDOW`
3. Use a smaller quantization (Q4_K_M instead of Q8_0)

### Issue: Slow inference on GPU

**Symptoms:** GPU usage low, inference slow

**Solution:**
1. Verify CUDA/Metal build: Check installation logs for acceleration support
2. Ensure `LLAMA_CPP_N_GPU_LAYERS=-1`
3. Check GPU visibility: `nvidia-smi` (CUDA) or Activity Monitor (Metal)

### Issue: Build fails with CMAKE errors

**Error:** CMake or compiler errors during installation

**Solution:**
```bash
# Ensure build tools are installed
# Ubuntu/Debian:
sudo apt-get install build-essential cmake

# macOS:
xcode-select --install
brew install cmake

# Then retry installation
CMAKE_ARGS="..." pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Rollback Plan

If you need to rollback to Ollama:

```bash
# Checkout previous version
git checkout <previous-commit-sha>

# Reinstall dependencies
uv sync

# Start Ollama service
# Docker: docker compose --profile cpu up
# Local: Ensure Ollama is running separately

# Restore old .env settings
```

## Support

For issues specific to:
- **llama.cpp**: See [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- **llama-cpp-python**: See [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- **GGUF models**: Check model repository documentation on Hugging Face
- **This application**: Open an issue in the docint repository

## FAQ

**Q: Can I use the same models I used with Ollama?**

A: Not directly. You need GGUF versions. Many popular Ollama models have GGUF equivalents on Hugging Face.

**Q: Do GGUF models perform the same as Ollama models?**

A: Yes, GGUF is a well-optimized format. Performance depends on quantization level (Q4_K_M, Q8_0, etc.).

**Q: Can I still use Ollama?**

A: The application no longer supports Ollama. You would need to use an older version or revert the changes.

**Q: What quantization level should I use?**

A: 
- Q4_K_M: Good balance (recommended for most users)
- Q5_K_M: Better quality, slightly larger
- Q8_0: Best quality, much larger
- Q2_K/Q3_K: Smaller, lower quality

**Q: How do I check if GPU acceleration is working?**

A: 
- CUDA: Run `nvidia-smi` while inferencing, check GPU utilization
- Metal: Check Activity Monitor → GPU History
- Look for "offloaded X layers" in application logs

**Q: Can I use multiple GPUs?**

A: Yes, llama.cpp supports multi-GPU. The layers are automatically distributed. Ensure all GPUs are visible via `CUDA_VISIBLE_DEVICES`.

## Additional Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Hugging Face GGUF Models](https://huggingface.co/models?library=gguf)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
