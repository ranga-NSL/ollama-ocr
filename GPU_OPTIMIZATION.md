# GPU Optimization Guide for Ollama

## Problem
Your Ollama model is running on **100% CPU** instead of GPU, causing slow performance and low GPU utilization (39%).

## Diagnosis
Check current status:
```powershell
ollama ps
```

If you see `100% CPU` in the PROCESSOR column, the model is not using GPU.

## Solutions

### 1. Verify CUDA Installation
```powershell
# Check CUDA version
nvidia-smi
```

Ensure you have:
- NVIDIA GPU drivers installed
- CUDA 12.1 or compatible version
- cuDNN installed

### 2. Reinstall Ollama with GPU Support

**Option A: Download GPU-enabled Ollama**
1. Uninstall current Ollama
2. Download the latest Ollama from https://ollama.com/download
3. Ensure you download the version with CUDA support for Windows

**Option B: Use Ollama with DirectML (Windows)**
Ollama on Windows should automatically detect and use GPU if:
- NVIDIA drivers are installed
- CUDA is properly configured
- Ollama was installed with GPU support

### 3. Set Environment Variables

Create or update `OLLAMA_GPU_ENABLED`:
```powershell
# PowerShell
$env:OLLAMA_GPU_ENABLED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"  # Use first GPU

# Or set permanently
[System.Environment]::SetEnvironmentVariable('OLLAMA_GPU_ENABLED', '1', 'User')
```

### 4. Restart Ollama Service

After setting environment variables:
```powershell
# Stop Ollama
Stop-Process -Name ollama -Force

# Restart Ollama (it should auto-start, or start manually)
ollama serve
```

### 5. Verify GPU Detection

Check if Ollama can see your GPU:
```powershell
ollama ps
```

After restart, it should show GPU usage instead of CPU.

### 6. Use a Vision Model Optimized for GPU

Try using a vision model that's better optimized:
```powershell
# Pull a vision model
ollama pull llama3.2-vision:latest

# Or use a quantized version that fits better in GPU memory
ollama pull llama3.2-vision:3b
```

### 7. Check Model Size vs GPU Memory

Your RTX 4060 Ti has 8GB VRAM. The model `ministral-3:8b` might be too large:
- **8B model** typically needs 8-16GB VRAM (depending on quantization)
- If model doesn't fit, it falls back to CPU

**Solutions:**
- Use a quantized model (Q4, Q5, Q8)
- Use a smaller model (3B, 7B quantized)
- Use a vision-specific model that's optimized

### 8. Force GPU Usage via Modelfile

Create a custom modelfile that forces GPU:
```powershell
# Create modelfile
ollama create ministral-3:8b-gpu -f modelfile.txt
```

In `modelfile.txt`:
```
FROM ministral-3:8b
PARAMETER num_gpu 1
PARAMETER num_thread 4
```

### 9. Check Ollama Logs

Check if there are GPU-related errors:
```powershell
# On Windows, check Event Viewer or Ollama logs
# Look for CUDA errors or GPU initialization failures
```

### 10. Alternative: Use Cloud Models

If local GPU setup is problematic, use cloud models which are already GPU-accelerated:
```powershell
$env:OLLAMA_API_KEY = "<your-key>"
python ollama_cloud_ocr.py "images/handwriting.jpg" --model "mistral-large-3:675b-cloud"
```

## Quick Test

After applying fixes, test GPU usage:
```powershell
# Start a model
ollama run ministral-3:8b "Hello"

# In another terminal, check
ollama ps
```

You should see GPU percentage instead of "100% CPU".

## Expected Results

After fixing:
- `ollama ps` should show GPU usage (not 100% CPU)
- Task Manager should show higher GPU utilization (60-90%+)
- Processing should be significantly faster
- GPU memory should be in use (check in Task Manager)

## Troubleshooting

### Still showing CPU?
1. Check NVIDIA drivers: `nvidia-smi` should work
2. Verify CUDA: `nvcc --version`
3. Reinstall Ollama completely
4. Check Windows GPU scheduling is enabled
5. Ensure no other processes are blocking GPU access

### Model too large?
- Try: `ollama pull llama3.2-vision:3b` (smaller model)
- Or use quantized: `ollama pull ministral-3:8b-q4_0`

### Still slow?
- Increase batch size in processing
- Use smaller context windows
- Process multiple images in parallel (if memory allows)



