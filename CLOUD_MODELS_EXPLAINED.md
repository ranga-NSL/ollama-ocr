# Cloud Vision Models Explained

This document explains the cloud vision models used in `ollama_cloud_ocr.py`.

## CLOUD_VISION_MODELS List

```python
CLOUD_VISION_MODELS = [
    'mistral-large-3:675b-cloud',  # Vision and reasoning model (OCR capable)
    'qwen3-vl:235b-cloud',  # Vision-language model (OCR capable)
    'minimax-m2:cloud',     # Cloud LLM (may be text-only)
    'gpt-oss:120b-cloud'    # Cloud LLM (may be text-only)
]
```

## Model Details

### 1. `mistral-large-3:675b-cloud` ⭐ (Primary/Default)
- **Type**: Vision + Reasoning model
- **Size**: 675 billion parameters
- **Capabilities**:
  - ✅ **Vision/OCR**: Can read and extract text from images
  - ✅ **Advanced Reasoning**: Mathematical, logical, and technical reasoning
  - ✅ **Document Analysis**: Understands document structure and context
  - ✅ **Multi-modal**: Processes both text and images simultaneously
- **Best For**:
  - Complex document analysis requiring reasoning
  - OCR tasks that need context understanding
  - Handwriting recognition with context
  - Structured data extraction (receipts, forms)
- **Status**: ✅ **Confirmed vision-capable** (detected by code: `'mistral' in model.lower()`)
- **Default Model**: Used when no model is specified (first in list)

### 2. `qwen3-vl:235b-cloud` ⭐ (Recommended Vision Model)
- **Type**: Vision-Language model
- **Size**: 235 billion parameters
- **Capabilities**:
  - ✅ **Vision/OCR**: Specialized for visual understanding and text extraction
  - ✅ **Multi-modal Understanding**: Excellent at image-text relationships
  - ✅ **Language Understanding**: Strong natural language processing
- **Best For**:
  - Pure OCR tasks
  - Image description and analysis
  - Visual question answering
  - Document text extraction
- **Status**: ✅ **Confirmed vision-capable** (detected by code: `'-vl' in model.lower()`)
- **Note**: The `-vl` suffix indicates "Vision-Language" capability

### 3. `minimax-m2:cloud` ⚠️ (May be text-only)
- **Type**: General-purpose LLM
- **Size**: Unknown (cloud model)
- **Capabilities**:
  - ❓ **Vision/OCR**: **Unconfirmed** - may not support vision
  - ✅ **Text Processing**: General language model capabilities
- **Best For**:
  - Text-only tasks
  - Language understanding
  - **NOT recommended for OCR** (may fail with images)
- **Status**: ⚠️ **Warning issued** by code (no `-vl`, `vision`, or `mistral` in name)
- **Note**: Code will warn: `"[WARNING] Model 'minimax-m2:cloud' may not support vision/OCR capabilities."`

### 4. `gpt-oss:120b-cloud` ⚠️ (May be text-only)
- **Type**: Open-source GPT-style model
- **Size**: 120 billion parameters
- **Capabilities**:
  - ❓ **Vision/OCR**: **Unconfirmed** - may not support vision
  - ✅ **Text Processing**: GPT-style language model
- **Best For**:
  - Text generation and understanding
  - **NOT recommended for OCR** (may fail with images)
- **Status**: ⚠️ **Warning issued** by code (no vision indicators in name)
- **Note**: Code will warn: `"[WARNING] Model 'gpt-oss:120b-cloud' may not support vision/OCR capabilities."`

## How the Code Selects Models

### Default Behavior
```python
self.model = model or CLOUD_VISION_MODELS[0]  # Defaults to mistral-large-3:675b-cloud
```

If no model is specified, it uses the **first model** in the list: `mistral-large-3:675b-cloud`

### Vision Detection Logic
```python
if '-vl' not in model.lower() and 'vision' not in model.lower() and 'mistral' not in model.lower():
    print(f"[WARNING] Model '{model}' may not support vision/OCR capabilities.")
```

The code checks for:
- `-vl` suffix (Vision-Language indicator)
- `vision` in the name
- `mistral` in the name (Mistral models are known to support vision)

**Models that pass the check** (no warning):
- ✅ `mistral-large-3:675b-cloud` (has "mistral")
- ✅ `qwen3-vl:235b-cloud` (has "-vl")

**Models that trigger warning**:
- ⚠️ `minimax-m2:cloud` (no vision indicators)
- ⚠️ `gpt-oss:120b-cloud` (no vision indicators)

## Recommended Usage

### For OCR Tasks (Vision Required)
```python
# Best options (vision-capable):
analyzer = CloudOCRAnalyzer(model='mistral-large-3:675b-cloud')
analyzer = CloudOCRAnalyzer(model='qwen3-vl:235b-cloud')

# Or use default (mistral-large-3:675b-cloud):
analyzer = CloudOCRAnalyzer()
```

### For Text-Only Tasks
```python
# These may work for text, but will warn for images:
analyzer = CloudOCRAnalyzer(model='minimax-m2:cloud')
analyzer = CloudOCRAnalyzer(model='gpt-oss:120b-cloud')
```

## Model Comparison Table

| Model | Vision | Reasoning | Size | Best Use Case |
|-------|--------|-----------|------|---------------|
| `mistral-large-3:675b-cloud` | ✅ Yes | ✅ Advanced | 675B | Complex OCR + reasoning |
| `qwen3-vl:235b-cloud` | ✅ Yes | ✅ Good | 235B | Pure OCR, image analysis |
| `minimax-m2:cloud` | ❓ Unknown | ✅ Good | Unknown | Text-only tasks |
| `gpt-oss:120b-cloud` | ❓ Unknown | ✅ Good | 120B | Text-only tasks |

## Usage Examples

### Example 1: Using Default (Mistral)
```python
# Uses mistral-large-3:675b-cloud automatically
analyzer = CloudOCRAnalyzer()
result = analyzer.extract_text("image.jpg")
```

### Example 2: Explicitly Using Qwen Vision Model
```python
analyzer = CloudOCRAnalyzer(model='qwen3-vl:235b-cloud')
result = analyzer.extract_text("image.jpg")
```

### Example 3: Batch Processing with All Models
```python
# The main() function iterates through ALL models
# This will try all 4 models, but minimax-m2 and gpt-oss may fail on images
python ollama_cloud_ocr.py
```

## Important Notes

1. **Cloud Models Only**: These models are **only available via Ollama Cloud API**, not locally
2. **API Key Required**: All models require `OLLAMA_API_KEY` environment variable
3. **Cost**: Cloud models may have usage costs (check Ollama Cloud pricing)
4. **Performance**: Cloud models run on Ollama's servers (GPU-accelerated), not your local machine
5. **Availability**: Model availability depends on Ollama Cloud service status

## When to Use Which Model

### Use `mistral-large-3:675b-cloud` when:
- You need both OCR and reasoning
- Processing complex documents requiring context
- Extracting structured data (receipts, forms)
- Handwriting that needs interpretation

### Use `qwen3-vl:235b-cloud` when:
- Pure OCR/text extraction
- Image description tasks
- Visual question answering
- Standard document processing

### Avoid `minimax-m2:cloud` and `gpt-oss:120b-cloud` for:
- ❌ Image processing
- ❌ OCR tasks
- ❌ Vision-related operations

### Use `minimax-m2:cloud` and `gpt-oss:120b-cloud` for:
- ✅ Text-only language tasks
- ✅ Text generation
- ✅ Language understanding (no images)

