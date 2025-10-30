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
├── advanced_ocr.py                  # Local LLM OCR (multi-capability demo)
├── ollama_local_ocr.py             # Local OCR (crisp, deterministic options)
├── ollama_cloud_ocr.py             # Cloud OCR (Ollama Cloud API)
├── test_ollama_search.py           # List/search/test installed Ollama models
├── ollama_cloud_model.py           # Minimal Cloud API connectivity test
├── requirements.txt                # Python dependencies (default)
└── requirements_313_GPU.txt        # Optional pinned deps for Py 3.13 + GPU
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

### Printed OCR (Recommended for typeset documents)
```bash
# Single image (preprocess + Tesseract)
python ocr_printed.py "images/trader-joes-receipt.jpg"

# Batch over images/ and write outputs/*.printed.txt
python ocr_printed.py
```

### Handwriting OCR (TrOCR)
```bash
# Single image with Hugging Face TrOCR
python ocr_handwriting_trocr.py "images/handwriting.jpg"

# Batch over images/ and write outputs/*.handwriting.txt
python ocr_handwriting_trocr.py
```

### Auto Router (printed vs handwriting)
```bash
# Route automatically and write outputs/*.printed.txt or *.handwriting.txt
python ocr_router.py

# Force a specific engine
python ocr_router.py "images/test_notes.jpg" --engine handwriting
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

# Verify Cloud connectivity (requires OLLAMA_API_KEY)
python ollama_cloud_model.py
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
| **LLM-based (Ollama)** | Handwriting, Complex layouts | Excellent handwriting recognition, Context understanding | Slower processing, Requires GPU |
| **Surya OCR** | Multilingual, Modern documents | Good accuracy, Layout detection | Column reading issues, Complex setup |
| **PyTesseract** | Printed text, Line detection | Fast, Reliable for printed text, Good line detection | Poor handwriting recognition |

## 🎯 Key Findings & Recommendations

### 📝 Handwriting Recognition
- **✅ LLM-based OCR (Ollama)** - **BEST CHOICE**
  - Excellent at understanding cursive and script handwriting
  - Context-aware text interpretation
  - Handles various handwriting styles effectively

### 📄 Printed Text Recognition
- **✅ PyTesseract** - **BEST CHOICE**
  - Superior line detection and column handling
  - Fast processing for printed documents
  - Excellent at maintaining text structure and layout
  - Reliable for receipts, forms, and structured documents

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

PyTesseract: ⭐⭐ (Poor)
- Struggles with cursive text
- Produces garbled output for handwriting
- Low confidence scores

Surya OCR: ⭐⭐⭐ (Good)
- Better than PyTesseract for handwriting
- But still struggles with complex scripts
```

### Printed Text Performance
```
PyTesseract: ⭐⭐⭐⭐⭐ (Excellent)
- Perfect line detection
- Maintains column structure
- Fast and reliable

Surya OCR: ⭐⭐⭐ (Good)
- Good text detection
- Issues with column reading
- Reads across columns instead of within columns

LLM-based OCR: ⭐⭐⭐⭐ (Very Good)
- Good accuracy but slower
- Better for complex layouts
- Context understanding helps
```

## 🛠️ Technical Details

### Models Used
- Local: **llama3.2-vision:latest** (primary vision model used here)
- Cloud (example): **qwen3-vl:235b-cloud** (vision-language model)

### Performance Considerations
- **GPU Acceleration**: Recommended for LLM-based OCR
- **Processing Time**: PyTesseract < Surya < LLM-based
- **Accuracy**: Depends on document type (see comparison above)
- **Determinism** (local): Use `--temperature 0.0 --seed 1` in `ollama_local_ocr.py` for stable outputs
- **Handwriting**: TrOCR provides higher accuracy than Tesseract for cursive/script handwriting ([source](https://www.handwritingocr.com/handwriting-to-text/how-to-convert-handwriting-to-text-using-python))

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

**Summary**: For handwriting, use LLM-based OCR. For printed text with columns, use PyTesseract. Surya OCR is good for general text but has column reading limitations.
