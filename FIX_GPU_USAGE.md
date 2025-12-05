# Fix GPU Usage - Immediate Solution

## Problem Identified
- Model `ministral-3:8b` shows **16GB SIZE** in `ollama ps`
- Your RTX 4060 Ti has **8GB VRAM**
- Model doesn't fit in GPU memory → Falls back to **100% CPU**

## Quick Fixes

### Option 1: Use llama3.2-vision (7.8GB - Should Fit)
```powershell
python ollama_local_ocr.py "images/handwriting.jpg" --model llama3.2-vision:latest
```

### Option 2: Use Quantized Model (Recommended)
Pull a quantized version that fits in 8GB:
```powershell
# Check available quantized models
ollama list ministral-3

# Try Q4 quantization (smaller)
ollama pull ministral-3:8b-q4_0
python ollama_local_ocr.py "images/handwriting.jpg" --model ministral-3:8b-q4_0
```

### Option 3: Use Cloud Models (Already GPU-Accelerated)
```powershell
$env:OLLAMA_API_KEY = "<your-key>"
python ollama_cloud_ocr.py "images/handwriting.jpg" --model "mistral-large-3:675b-cloud"
```

### Option 4: Check Ollama GPU Detection
```powershell
# Verify Ollama can see GPU
ollama run llama3.2-vision:latest "test"

# In another terminal, immediately check:
ollama ps
# Should show GPU percentage, not "100% CPU"
```

## Verify GPU Usage

After running a model, check:
```powershell
# Check Ollama process
ollama ps

# Check GPU in Task Manager or:
nvidia-smi
# Should show GPU memory usage and higher GPU utilization
```

## Expected Results

When GPU is working:
- `ollama ps` shows GPU percentage (not "100% CPU")
- `nvidia-smi` shows GPU memory usage (2-7GB)
- Task Manager shows 60-90%+ GPU utilization
- Processing is 3-5x faster

## Model Size Guide for 8GB VRAM

| Model | Size | Fits? | Recommendation |
|-------|------|-------|----------------|
| ministral-3:8b | 16GB | ❌ No | Use quantized version |
| llama3.2-vision:latest | 7.8GB | ✅ Maybe | Should work |
| ministral-3:8b-q4_0 | ~4-5GB | ✅ Yes | Best option |
| ministral-3:8b-q5_0 | ~5-6GB | ✅ Yes | Good balance |

## Next Steps

1. **Try llama3.2-vision first** (already installed, 7.8GB)
2. **If still CPU**, pull quantized model: `ollama pull ministral-3:8b-q4_0`
3. **Monitor GPU usage** with `nvidia-smi` while processing
4. **If GPU still not used**, check Ollama installation has GPU support



