# Mistral Cloud Model Testing

This document describes the `test_mistral_cloud.py` script and the test images used for evaluating the `mistral-large-3:675b-cloud` model's capabilities.

## Overview

The `mistral-large-3:675b-cloud` model is a cloud-hosted model available through Ollama Cloud API that provides both:
- **Text Reasoning**: Advanced logical, mathematical, and technical reasoning capabilities
- **Vision/OCR**: Image understanding, text extraction, and document analysis

## Prerequisites

1. **Python 3.12+** installed
2. **Ollama Python package** installed:
   ```powershell
   pip install ollama
   ```
3. **OLLAMA_API_KEY** environment variable set with your Ollama Cloud API key:
   ```powershell
   # Windows PowerShell
   $env:OLLAMA_API_KEY = "<your-api-key>"
   
   # Windows CMD
   set OLLAMA_API_KEY=<your-api-key>
   ```

## Test Script: test_mistral_cloud.py

### Features

The script tests two main capabilities of the Mistral model:

1. **Text Reasoning Tests** - Evaluates the model's ability to:
   - Solve mathematical problems
   - Explain technical concepts
   - Perform logical reasoning

2. **Vision/OCR Tests** - Evaluates the model's ability to:
   - Extract text from images (OCR)
   - Describe image content
   - Analyze document structure and type

### Usage

#### Run All Tests
```powershell
python test_mistral_cloud.py
```

This will run both text reasoning and vision/OCR tests on all images in the `images/` directory.

#### Run Only Text Reasoning Tests
```powershell
python test_mistral_cloud.py --reasoning-only
```

#### Run Only Vision/OCR Tests
```powershell
python test_mistral_cloud.py --vision-only
```

#### Specify Custom Images Directory
```powershell
python test_mistral_cloud.py --images-dir "path/to/images"
```

### Test Structure

#### Text Reasoning Tests

The script includes three reasoning questions:

1. **Mathematical Reasoning**: Train speed and distance problem
2. **Technical Explanation**: Recursion in programming
3. **Logical Reasoning**: Syllogistic reasoning with roses and flowers

#### Vision/OCR Tests

For each image, the script performs three types of analysis:

1. **Text Extraction (OCR)**: Extracts all visible text while preserving formatting
2. **Image Description**: Provides detailed description of image content, text, objects, and layout
3. **Document Analysis**: Identifies document type and extracts key structured information

## Test Images

The script automatically processes all images in the `images/` subdirectory. The following test images are included:

### handwriting.jpg
- **Type**: Handwritten text sample
- **Purpose**: Tests OCR capabilities on cursive and script handwriting
- **Expected Analysis**: Text transcription, handwriting style recognition

### trader-joes-receipt.jpg
- **Type**: Receipt with structured data
- **Purpose**: Tests structured data extraction and column layout understanding
- **Expected Analysis**: Store information, itemized list, prices, totals, date

### test_notes.jpg
- **Type**: Technical notes/document
- **Purpose**: Tests text extraction from technical documents with mixed formatting
- **Expected Analysis**: Technical content extraction, layout preservation

### test_dwg.jpg
- **Type**: Technical drawing
- **Purpose**: Tests vision capabilities on technical drawings and diagrams
- **Expected Analysis**: Drawing elements, annotations, technical symbols

## Output Format

The script provides structured output:

```
============================================================
Mistral Cloud Model Test
Model: mistral-large-3:675b-cloud
============================================================
✓ API Key present (length: 57)

============================================================
TEST 1: Text Reasoning
============================================================

--- Question 1: Mathematical reasoning problem ---
Prompt: [question text]
Response:
------------------------------------------------------------
[model response]
✓ Completed successfully

============================================================
TEST 2: Vision/OCR Capabilities
============================================================

Found 4 image(s) to test:
  - handwriting.jpg
  - test_dwg.jpg
  - test_notes.jpg
  - trader-joes-receipt.jpg

============================================================
Processing: handwriting.jpg
============================================================

--- Test 1: Text Extraction (OCR) ---
------------------------------------------------------------
[extracted text]
✓ Completed successfully
```

## Error Handling

The script includes error handling for:
- Missing API key
- Network connectivity issues
- Invalid image files
- API response errors

All errors are displayed with clear messages and stack traces for debugging.

## Integration with Cloud OCR

The Mistral model can also be used with the main `ollama_cloud_ocr.py` script:

```powershell
# Text extraction
python ollama_cloud_ocr.py "images/handwriting.jpg" --model "mistral-large-3:675b-cloud"

# Document analysis
python ollama_cloud_ocr.py "images/trader-joes-receipt.jpg" --model "mistral-large-3:675b-cloud" --mode document

# Structured data extraction
python ollama_cloud_ocr.py "images/trader-joes-receipt.jpg" --model "mistral-large-3:675b-cloud" --mode structured --data-type receipt
```

## Model Capabilities Summary

| Capability | Description | Use Case |
|------------|-------------|----------|
| **Text Reasoning** | Mathematical problems, logical reasoning, technical explanations | Problem solving, analysis, explanations |
| **OCR** | Text extraction from images | Document digitization, data entry |
| **Image Description** | Detailed visual analysis | Content understanding, accessibility |
| **Document Analysis** | Structure and type identification | Document classification, information extraction |
| **Structured Data** | Extract formatted data (receipts, forms) | Data processing, automation |

## Troubleshooting

### API Key Issues
- Ensure `OLLAMA_API_KEY` is set in your environment
- Verify the key is valid and has access to Ollama Cloud
- Check key length (should be around 50-60 characters)

### No Content Received
- Check network connectivity
- Verify the model name is correct: `mistral-large-3:675b-cloud`
- Ensure images are in supported formats (jpg, png, bmp, tiff, webp)

### Image Processing Errors
- Verify image files exist and are readable
- Check file permissions
- Ensure images are not corrupted

## Notes

- The script uses streaming responses for real-time output
- All images are automatically encoded to base64 for API transmission
- The script processes images in alphabetical order
- Windows console encoding is automatically configured for proper display

