# OCR Comparison Project

This project compares different OCR (Optical Character Recognition) technologies to evaluate their performance on various document types including printed text, handwriting, receipts, and technical drawings.

## 🎯 Project Overview

The project implements and tests three different OCR approaches:
- **LLM-based OCR** (Ollama with vision models)
- **Traditional OCR** (PyTesseract)
- **Modern OCR** (Surya OCR)

## 📁 Project Structure

```
ollama-ocr/
├── images/                          # Test images
│   ├── handwriting.jpg              # Handwritten text
│   ├── trader-joes-receipt.jpg      # Receipt with columns
│   ├── test_notes.jpg               # Technical notes
│   └── test_dwg.jpg                 # Technical drawing
├── ocr_router.py                   # Unified OCR router (Tesseract + TrOCR, images + PDFs)
├── advanced_ocr.py                  # Local LLM OCR (multi-capability demo)
├── ollama_local_ocr.py             # Local OCR (crisp, deterministic options)
├── ollama_cloud_ocr.py             # Cloud OCR (Ollama Cloud API)
├── test_ollama_search.py           # List/search/test installed Ollama models
└── requirements.txt                # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Ollama installed and running
- CUDA 12.1 (for GPU acceleration)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ollama-ocr
   ```

2. **Create virtual environment**
   ```bash
   python -m venv ollama-ocr_env
   .\ollama-ocr_env\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama models**
   ```bash
   ollama pull llama3.2-vision:latest
   ollama pull minicpm-v:latest
   ```

## 🔧 Usage

### LLM-based OCR (Local - Recommended for Handwriting)
```bash
# Process all test images
python advanced_ocr.py

# Process specific image
python advanced_ocr.py "images/handwriting.jpg"
```

### Local OCR (Crisp, deterministic)
```bash
# Batch over images directory
python ollama_local_ocr.py --temperature 0.0 --seed 1

# Single image
python ollama_local_ocr.py "images/handwriting.jpg" --temperature 0.0 --seed 1
```

### Cloud OCR (Ollama Cloud)
Requires `OLLAMA_API_KEY` and a vision-capable cloud model (e.g., `qwen3-vl:235b-cloud`).
```powershell
# Windows (PowerShell)
$env:OLLAMA_API_KEY = "<your-key>"
python ollama_cloud_ocr.py

# Single image with explicit model
python ollama_cloud_ocr.py "images/handwriting.jpg" --model "qwen3-vl:235b-cloud"
```

### Cloud OCR with Advanced Models
The `ollama_cloud_ocr.py` supports various cloud models including `mistral-large-3:675b-cloud` which provides both vision/OCR capabilities and advanced reasoning.

**Prerequisites:**
- `OLLAMA_API_KEY` environment variable set
- Access to Ollama Cloud API

**Usage Examples:**
```powershell
# Windows (PowerShell)
$env:OLLAMA_API_KEY = "<your-key>"

# Use Mistral for OCR
python ollama_cloud_ocr.py "images/handwriting.jpg" --model "mistral-large-3:675b-cloud"

# Document analysis mode
python ollama_cloud_ocr.py "images/trader-joes-receipt.jpg" --model "mistral-large-3:675b-cloud" --mode document

# Structured data extraction (receipts, forms)
python ollama_cloud_ocr.py "images/trader-joes-receipt.jpg" --model "mistral-large-3:675b-cloud" --mode structured --data-type receipt

# Process PDF files
python ollama_cloud_ocr.py "document.pdf" --model "mistral-large-3:675b-cloud"

# Process specific PDF page
python ollama_cloud_ocr.py "document.pdf" --page 1 --model "mistral-large-3:675b-cloud"
```

**Available Modes:**
- `text`: Basic text extraction
- `handwriting`: Specialized handwriting transcription
- `structured`: Extract structured data (receipts, forms, price tags)
- `document`: Comprehensive document analysis
- `multi`: Multi-step analysis (description, text extraction, object identification)

### Unified OCR Router (Recommended - Tesseract + TrOCR)
The `ocr_router.py` is a unified program that handles both images and PDFs, automatically routing to the best OCR engine (Tesseract for printed text, TrOCR for handwriting).

**Features:**
- Supports images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`) and PDFs (`.pdf`)
- Auto-detection: automatically chooses printed vs handwriting OCR
- PDF support: converts PDF pages to images and processes each page
- No LLM dependencies: uses traditional OCR engines only

```bash
# Single image - auto-detect engine
python ocr_router.py "images/handwriting.jpg"

# Single PDF - processes all pages with page separators
python ocr_router.py "document.pdf"

# Force a specific engine
python ocr_router.py "images/test_notes.jpg" --engine printed
python ocr_router.py "images/handwriting.jpg" --engine handwriting

# Batch mode - processes all images and PDFs in images/ directory
python ocr_router.py

# Specify output directory
python ocr_router.py --outdir my_outputs
```

### Batch Runner (summary CSV)
```bash
# Process all images and write outputs + summary.csv
python run_all_ocr.py
```

### Model Discovery / Sanity Checks
```powershell
# List installed models, find vision candidates, and test connectivity
python test_ollama_search.py

# Inspect a specific model and test it
python test_ollama_search.py llama3.2-vision:latest
```

### Surya OCR (Good for Printed Text)
```bash
# Process specific image
python test_SuryaOCR.py "images/trader-joes-receipt.jpg"
```

### PyTesseract (Best for Printed Text with Layout)
```bash
# Process specific image
python test_pytesseract.py "images/trader-joes-receipt.jpg"
```

## 📊 OCR Technology Comparison

| Technology | Best For | Strengths | Limitations |
|------------|----------|-----------|-------------|
| **LLM-based (Ollama)** | Handwriting, Complex layouts | Excellent handwriting recognition, Context understanding | Slower processing, Requires GPU/API key |
| **TrOCR** | Handwriting | Good handwriting recognition, GPU-accelerated | Requires GPU, Slower than Tesseract |
| **Tesseract (PyTesseract)** | Printed text, Line detection | Fast, Reliable for printed text, Good line detection | Poor handwriting recognition |
| **Surya OCR** | Multilingual, Modern documents | Good accuracy, Layout detection | Column reading issues, Complex setup |

## 🎯 Key Findings & Recommendations

### 📝 Handwriting Recognition
- **✅ LLM-based OCR (Ollama)** - **BEST CHOICE** (Cloud/Local)
  - Excellent at understanding cursive and script handwriting
  - Context-aware text interpretation
  - Handles various handwriting styles effectively
- **✅ TrOCR** - **GOOD CHOICE** (Local, no API key needed)
  - Good handwriting recognition
  - GPU-accelerated
  - Available in `ocr_router.py` with auto-detection

### 📄 Printed Text Recognition
- **✅ Tesseract (PyTesseract)** - **BEST CHOICE**
  - Superior line detection and column handling
  - Fast processing for printed documents
  - Excellent at maintaining text structure and layout
  - Reliable for receipts, forms, and structured documents
  - Available in `ocr_router.py` with auto-detection

### 🏢 Complex Documents
- **⚠️ Surya OCR** - **MIXED RESULTS**
  - Good text detection but struggles with column layouts
  - Tends to read left-to-right across columns instead of column-wise
  - Better for single-column documents

## 🔍 Detailed Analysis

### Handwriting Performance
```
LLM-based OCR: ⭐⭐⭐⭐⭐ (Excellent)
- Accurately transcribes cursive writing
- Understands context and formatting
- Handles various handwriting styles

TrOCR: ⭐⭐⭐⭐ (Very Good)
- Good handwriting recognition
- Better than Tesseract for cursive/script
- GPU-accelerated

Tesseract (PyTesseract): ⭐⭐ (Poor)
- Struggles with cursive text
- Produces garbled output for handwriting
- Low confidence scores

Surya OCR: ⭐⭐⭐ (Good)
- Better than Tesseract for handwriting
- But still struggles with complex scripts
```

### Printed Text Performance
```
Tesseract (PyTesseract): ⭐⭐⭐⭐⭐ (Excellent)
- Perfect line detection
- Maintains column structure
- Fast and reliable

LLM-based OCR: ⭐⭐⭐⭐ (Very Good)
- Good accuracy but slower
- Better for complex layouts
- Context understanding helps

Surya OCR: ⭐⭐⭐ (Good)
- Good text detection
- Issues with column reading
- Reads across columns instead of within columns
```

## 🛠️ Technical Details

### OCR Engines
- **Tesseract (PyTesseract)**: Traditional OCR for printed text, included in `ocr_router.py`
- **TrOCR**: Transformer-based OCR for handwriting, included in `ocr_router.py`
- **LLM-based (Ollama Local)**: Vision models for advanced OCR, see `ollama_local_ocr.py` and `advanced_ocr.py`
- **LLM-based (Ollama Cloud)**: Cloud-hosted vision models, see `ollama_cloud_ocr.py`

### Models Used
- Local LLM: **llama3.2-vision:latest** (primary vision model used here)
- Cloud (example): **qwen3-vl:235b-cloud** (vision-language model)
- Cloud (advanced): **mistral-large-3:675b-cloud** (vision + reasoning model)
- TrOCR: **microsoft/trocr-base-handwritten** (handwriting model)

### Performance Considerations
- **GPU Acceleration**: Recommended for LLM-based OCR and TrOCR
- **Processing Time**: Tesseract < TrOCR < LLM-based
- **Accuracy**: Depends on document type (see comparison above)
- **Determinism** (local): Use `--temperature 0.0 --seed 1` in `ollama_local_ocr.py` for stable outputs
- **Handwriting**: TrOCR provides higher accuracy than Tesseract for cursive/script handwriting ([source](https://www.handwritingocr.com/handwriting-to-text/how-to-convert-handwriting-to-text-using-python))
- **PDF Support**: Both `ocr_router.py` and `ollama_cloud_ocr.py` support PDF files (requires `pdf2image`)

## 📈 Output Examples

### Receipt Processing
**PyTesseract Output:**
```
SOUR CREAM & ONION CORN: $2.49
SLICED WHOLE WHEAT BREAD: $2.49
RICE CAKES KOREAN TTEOK: $3.99
```

**Surya OCR Output:**
```
SOUR CREAM & ONION CORN<br>SLICED WHOLE WHEAT BREAD<br>RICE CAKES KOREAN TTEOK
$2.49<br>$2.49<br>$3.99
```

## 🔧 Troubleshooting

### Common Issues
1. **Ollama not running**: Start Ollama service before running LLM-based OCR
2. **Model not found**: Pull required models with `ollama pull <model-name>`
3. **CUDA errors**: Ensure CUDA 12.1 is properly installed
4. **Import errors**: Activate virtual environment and install dependencies
5. **Cloud 500/400 errors**: Ensure you pass a valid `OLLAMA_API_KEY` and use a vision-capable cloud model. The program encodes images appropriately; verify the image path exists.
6. **Tesseract not found**: Install the Tesseract binary and ensure it’s on PATH; on Windows, use the official installer. You can also set `pytesseract.pytesseract.tesseract_cmd` to the full path.

### Performance Optimization
- Use GPU acceleration for LLM-based OCR
- Preprocess images for better PyTesseract results
- Adjust confidence thresholds based on document quality

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Summary**: 
- **For handwriting**: Use LLM-based OCR (cloud/local) or TrOCR (local, in `ocr_router.py`)
- **For printed text with columns**: Use Tesseract (in `ocr_router.py` with auto-detection)
- **For unified solution**: Use `ocr_router.py` - automatically routes to best engine, supports images and PDFs
- **Surya OCR**: Good for general text but has column reading limitations
