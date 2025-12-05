# OCR Comparison Project

This project compares different OCR (Optical Character Recognition) technologies to evaluate their performance on various document types including printed text, handwriting, receipts, and technical drawings.

## 🎯 Project Overview

The project provides three OCR solutions:
- **Traditional OCR** (`ocr_router.py`): Tesseract for printed text + TrOCR for handwriting
- **Cloud OCR** (`ollama_cloud_ocr.py`): Cloud-hosted vision models via Ollama Cloud API
- **Local OCR** (`ollama_local_ocr.py`): Local Ollama vision models (requires Ollama running locally)

## 📁 Project Structure

```
ollama-ocr/
├── images/                    # Test images and PDFs
├── outputs/                   # Output directory (created automatically)
├── ocr_router.py             # Traditional OCR (Tesseract + TrOCR, images + PDFs)
├── ollama_cloud_ocr.py       # Cloud OCR (Ollama Cloud API, images + PDFs)
├── ollama_local_ocr.py       # Local LLM OCR (images only, requires local Ollama)
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Tesseract OCR binary (for `ocr_router.py`)
- `OLLAMA_API_KEY` environment variable (for `ollama_cloud_ocr.py`)
- Ollama installed and running locally (for `ollama_local_ocr.py`)

### Setup
1. **Create virtual environment**
   ```bash
   python -m venv ollama-ocr_env
   .\ollama-ocr_env\Scripts\activate  # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract** (Windows)
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Ensure it's on PATH or set `pytesseract.pytesseract.tesseract_cmd`

## 🔧 Usage

### Traditional OCR (`ocr_router.py`)

Unified program that automatically routes to the best OCR engine:
- **Tesseract** for printed text (receipts, forms, documents)
- **TrOCR** for handwriting

**Features:**
- Supports images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`) and PDFs (`.pdf`)
- Auto-detection: automatically chooses printed vs handwriting OCR
- No API keys required
- Output files: `OCR_{filename}.txt`

```bash
# Single image - auto-detect engine
python ocr_router.py "images/handwriting.jpg"

# Single PDF - processes all pages
python ocr_router.py "document.pdf"

# Force a specific engine
python ocr_router.py "images/test_notes.jpg" --engine printed
python ocr_router.py "images/handwriting.jpg" --engine handwriting

# Batch mode - processes all files in images/ directory
python ocr_router.py

# Batch mode with CSV summary and custom output directory
python ocr_router.py --outdir my_outputs --csv
```

**Arguments:**
- `file_path` (optional): Single file to process; if omitted, processes all files in `images/` directory
- `--engine`: `auto` (default), `printed`, or `handwriting`
- `--outdir`: Output directory (default: `outputs`, batch mode only)
- `--csv`: Generate CSV summary file (batch mode only)

### Cloud OCR (`ollama_cloud_ocr.py`)

Cloud-hosted vision models with multiple analysis modes.

**Features:**
- Multiple OCR modes: text, handwriting, structured, document, multi-step
- PDF support with per-page processing
- Advanced models: `mistral-large-3:675b-cloud`, `qwen3-vl:235b-cloud`
- Output files: `LLM_{filename}.txt`
- Requires `OLLAMA_API_KEY` environment variable

```powershell
# Windows (PowerShell)
$env:OLLAMA_API_KEY = "<your-key>"

# Batch mode - processes all files in images/ directory
python ollama_cloud_ocr.py

# Batch mode with CSV summary
python ollama_cloud_ocr.py --csv

# Single image
python ollama_cloud_ocr.py "images/handwriting.jpg"

# Specific mode and model
python ollama_cloud_ocr.py "images/receipt.jpg" --mode structured --data-type receipt --model "mistral-large-3:675b-cloud"

# PDF processing (specific page)
python ollama_cloud_ocr.py "document.pdf" --page 1

# Custom output directory (batch mode only)
python ollama_cloud_ocr.py --outdir my_outputs --csv
```

**Arguments:**
- `file_path` (optional): Single file to process; if omitted, processes all files in `images/` directory
- `--model`: Cloud model to use (default: `mistral-large-3:675b-cloud`)
- `--mode`: `text` (default), `handwriting`, `structured`, `document`, or `multi`
- `--data-type`: `receipt`, `form`, or `price_tag` (for structured mode)
- `--page`: Page number for PDFs (1-indexed, processes all pages if omitted)
- `--stream`: Stream the response
- `--show-timing`: Display performance timing metrics
- `--outdir`: Output directory (default: `outputs`, batch mode only)
- `--csv`: Generate CSV summary file (batch mode only)

### Local OCR (`ollama_local_ocr.py`)

Local Ollama vision models (requires Ollama running locally).

**Features:**
- Images only (no PDF support)
- GPU usage monitoring
- Deterministic outputs with temperature/seed control
- No API keys required

```bash
# Single image
python ollama_local_ocr.py "images/handwriting.jpg"

# Batch mode - processes all images in images/ directory
python ollama_local_ocr.py

# Deterministic output
python ollama_local_ocr.py "images/handwriting.jpg" --temperature 0.0 --seed 1

# Custom model and context window
python ollama_local_ocr.py "images/handwriting.jpg" --model "llama3.2-vision:latest" --num-ctx 2048

# Performance profiling
python ollama_local_ocr.py "images/handwriting.jpg" --profile --show-timing
```

**Arguments:**
- `image_path` (optional): Single image to process; if omitted, processes all images in `images/` directory
- `--model`: Ollama model to use (default: `llama3.2-vision:latest`)
- `--temperature`: Sampling temperature (default: 0.1, lower = more deterministic)
- `--seed`: Random seed for determinism (default: 1)
- `--num-ctx`: Context window size (smaller = less GPU memory)
- `--no-check-gpu`: Disable GPU usage check
- `--show-timing`: Show detailed timing information
- `--profile`: Enable detailed performance profiling

## 📁 Output Files

### File Naming
- **Traditional OCR**: `OCR_{filename}.txt` (e.g., `OCR_handwriting.txt`, `OCR_playbook_1.txt`)
- **Cloud OCR**: `LLM_{filename}.txt` (e.g., `LLM_handwriting.txt`, `LLM_playbook_1.txt`)
- **Local OCR**: No file output (console only)

### Output Directory
```
outputs/
├── OCR_handwriting.txt          # Traditional OCR output
├── OCR_playbook_1.txt           # Traditional OCR output (PDF)
├── LLM_handwriting.txt          # Cloud OCR output
├── LLM_playbook_1.txt           # Cloud OCR output (PDF)
└── summary.csv                   # CSV summary (if --csv flag used in batch mode)
```

### CSV Summary
When using `--csv` flag in batch mode:
- **Traditional OCR**: `file`, `engine`, `chars`, `pages`, `type`
- **Cloud OCR**: `file`, `mode`, `model`, `chars`, `pages`, `type`

## 📊 OCR Technology Comparison

| Technology | Best For | Strengths | Limitations |
|------------|----------|-----------|-------------|
| **Tesseract** | Printed text | Fast, reliable, good line detection | Poor handwriting recognition |
| **TrOCR** | Handwriting | Good handwriting recognition, GPU-accelerated | Requires GPU, slower than Tesseract |
| **LLM-based (Cloud)** | Handwriting, Complex layouts | Excellent handwriting, context understanding | Requires API key, slower processing |
| **LLM-based (Local)** | Handwriting | Good handwriting, no API key needed | Requires local Ollama, slower processing |

## 🎯 Recommendations

- **For handwriting**: Use `ocr_router.py` (TrOCR) or `ollama_cloud_ocr.py` (LLM)
- **For printed text**: Use `ocr_router.py` (Tesseract) - auto-detected
- **For unified solution**: Use `ocr_router.py` - automatically routes to best engine, supports images and PDFs
- **For advanced analysis**: Use `ollama_cloud_ocr.py` with structured/document modes
- **For local processing**: Use `ollama_local_ocr.py` - no API key, but requires local Ollama

## 🛠️ Technical Details

### OCR Engines
- **Tesseract (PyTesseract)**: Traditional OCR for printed text
- **TrOCR**: Transformer-based OCR for handwriting (`microsoft/trocr-base-handwritten`)
- **LLM-based (Ollama Cloud)**: Cloud-hosted vision models
- **LLM-based (Ollama Local)**: Local vision models (requires Ollama running)

### Models
- **TrOCR**: `microsoft/trocr-base-handwritten` (handwriting model)
- **Cloud**: `mistral-large-3:675b-cloud` (default, vision + reasoning), `qwen3-vl:235b-cloud` (vision-language)
- **Local**: `llama3.2-vision:latest` (default)

### Performance
- **Processing Time**: Tesseract < TrOCR < LLM-based
- **GPU Acceleration**: Recommended for TrOCR and local LLM OCR
- **PDF Support**: `ocr_router.py` and `ollama_cloud_ocr.py` support PDFs (requires `pdf2image`)

## 🔧 Troubleshooting

1. **Tesseract not found**: Install Tesseract binary and ensure it's on PATH
2. **Cloud API errors**: Verify `OLLAMA_API_KEY` is set and model is vision-capable
3. **PDF errors**: Install `pdf2image` and poppler (Windows: download separately)
4. **CUDA errors**: Ensure CUDA 12.1 is properly installed (for GPU acceleration)
5. **Ollama not running**: Start Ollama service before running local OCR
6. **Import errors**: Activate virtual environment and install dependencies

## 📝 License

This project is for educational and research purposes.
