import os
import sys
import tempfile
from pathlib import Path
from typing import List, Literal, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# PDF support with graceful fallback
try:
	from pdf2image import convert_from_path
	PDF2IMAGE_AVAILABLE = True
except ImportError:
	PDF2IMAGE_AVAILABLE = False
	convert_from_path = None


Engine = Literal['auto', 'printed', 'handwriting']
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.pdf'}


# ============================================================================
# TrOCR Engine Class (for handwriting OCR)
# ============================================================================

class TrOCREngine:
	"""Handwriting OCR engine using Hugging Face TrOCR"""
	def __init__(self, model_name: str = 'microsoft/trocr-base-handwritten', device: str | None = None):
		self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
		self.processor = TrOCRProcessor.from_pretrained(model_name)
		self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

	@torch.inference_mode()
	def ocr(self, image_path: str) -> str:
		if not os.path.exists(image_path):
			raise FileNotFoundError(f"Image file '{image_path}' not found")
		image = Image.open(image_path).convert('RGB')
		pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
		generated_ids = self.model.generate(pixel_values)
		text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		return text.strip()


# ============================================================================
# Printed Text OCR Functions (Tesseract)
# ============================================================================

def preprocess_image(img):
	"""Preprocess image for Tesseract OCR"""
	if img is None or img.size == 0:
		raise ValueError("Invalid image: image is None or empty")
	
	# Check image dimensions
	if len(img.shape) < 2:
		raise ValueError(f"Invalid image shape: {img.shape}")
	
	# Grayscale
	if len(img.shape) == 3:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img
	
	# Otsu threshold (binary)
	_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# Optional slight dilation to connect characters
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
	proc = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
	return proc


def ocr_printed(image_path: str) -> str:
	"""Perform OCR on printed text using Tesseract"""
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Image file '{image_path}' not found")
	img = cv2.imread(image_path)
	if img is None:
		raise ValueError(f"Failed to read image: {image_path}")
	proc = preprocess_image(img)
	config = '--psm 6'  # Assume a block of text
	text = pytesseract.image_to_string(proc, config=config)
	return text.strip()


# ============================================================================
# Utility Functions
# ============================================================================

def collect_images(images_dir: Path) -> List[str]:
	"""Collect image and PDF files from directory"""
	if not images_dir.exists():
		return []
	return [str(p) for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]


def is_pdf(file_path: str) -> bool:
	"""Check if file is a PDF"""
	return Path(file_path).suffix.lower() == '.pdf'


def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
	"""Convert PDF to list of PIL Images, one per page"""
	if not PDF2IMAGE_AVAILABLE:
		raise ImportError("pdf2image is not installed. Install it with: pip install pdf2image")
	
	if not os.path.exists(pdf_path):
		raise FileNotFoundError(f"PDF file '{pdf_path}' not found")
	
	try:
		images = convert_from_path(pdf_path)
		if not images:
			raise ValueError(f"Could not read any pages from PDF: {pdf_path}")
		return images
	except Exception as e:
		raise RuntimeError(f"PDF conversion failed: {str(e)}")


def simple_heuristic(image_path: str | Image.Image) -> Engine:
	"""
	Very lightweight heuristic: if filename hints at receipt/form or many straight lines -> printed
	Accepts either file path (str) or PIL Image
	"""
	if isinstance(image_path, Image.Image):
		# Convert PIL Image to numpy array for cv2
		try:
			img_array = np.array(image_path.convert('RGB'))
			if img_array.size == 0 or len(img_array.shape) < 2:
				return 'handwriting'  # Default to handwriting if image is invalid (safer for handwritten content)
			img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
		except Exception as e:
			# If conversion fails, try to use filename hint if available
			return 'handwriting'  # Default to handwriting on conversion error
		name = ''  # No filename for PIL Image
	else:
		# File path
		name = Path(image_path).name.lower()
		# Check for explicit hints in filename
		if any(k in name for k in ['receipt', 'invoice', 'form', 'document', 'pdf']):
			return 'printed'
		if any(k in name for k in ['handwriting', 'handwritten', 'note', 'notes', 'draft']):
			return 'handwriting'
		
		# Try to read image
		img = cv2.imread(image_path)
		if img is None or img.size == 0:
			# If image can't be read, use filename hint or default to handwriting
			return 'handwriting'
	
	# Validate image before processing
	if img is None or img.size == 0 or len(img.shape) < 2:
		return 'handwriting'
	
	try:
		# Check image dimensions - very small images might be problematic
		height, width = img.shape[:2]
		if height < 10 or width < 10:
			return 'handwriting'
		
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
		edges = cv2.Canny(gray, 50, 150)
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=60, maxLineGap=5)
		
		# If many straight lines detected, likely printed document
		if lines is not None and len(lines) > 15:
			return 'printed'
		# Otherwise, assume handwriting
		return 'handwriting'
	except Exception as e:
		# If processing fails, default to handwriting (safer for handwritten content)
		return 'handwriting'


def ocr_pil_image(pil_image: Image.Image, engine: Engine, trocr: TrOCREngine | None = None) -> str:
	"""
	Perform OCR on a PIL Image (for PDF pages)
	"""
	if engine == 'printed':
		# Convert PIL Image to numpy array for cv2
		try:
			img_array = np.array(pil_image.convert('RGB'))
			if img_array.size == 0 or len(img_array.shape) < 2:
				raise ValueError("Invalid PIL image dimensions")
			img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
			
			# Use same preprocessing as ocr_printed
			proc = preprocess_image(img)
			
			# Use pytesseract directly
			config = '--psm 6'
			text = pytesseract.image_to_string(proc, config=config)
			return text.strip()
		except Exception as e:
			raise RuntimeError(f"Failed to process image for printed OCR: {e}")
	else:
		# Handwriting - TrOCR needs a file path, so save temporarily
		if trocr is None:
			trocr = TrOCREngine()
		
		# Save PIL Image to temporary file
		with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
			pil_image.save(tmp_file.name, 'PNG')
			tmp_path = tmp_file.name
		
		try:
			text = trocr.ocr(tmp_path)
		finally:
			# Clean up temporary file
			try:
				os.unlink(tmp_path)
			except:
				pass
		
		return text


def route_ocr(file_path: str, engine: Engine = 'auto', trocr: TrOCREngine | None = None) -> Tuple[str, str, int]:
	"""
	Route OCR to appropriate engine for image or PDF file.
	
	Args:
		file_path: Path to image or PDF file
		engine: Engine selection ('auto', 'printed', 'handwriting')
		trocr: Optional TrOCR engine instance (for reuse)
	
	Returns:
		Tuple of (engine_name, text, page_count) where page_count=1 for images
	"""
	if is_pdf(file_path):
		return process_pdf(file_path, engine, trocr)
	else:
		# Process single image
		chosen: Engine = engine
		if engine == 'auto':
			chosen = simple_heuristic(file_path)
		
		if chosen == 'printed':
			text = ocr_printed(file_path)
			return 'printed', text, 1
		else:
			if trocr is None:
				trocr = TrOCREngine()
			text = trocr.ocr(file_path)
			return 'handwriting', text, 1


def process_pdf(pdf_path: str, engine: Engine, trocr: TrOCREngine | None = None) -> Tuple[str, str, int]:
	"""
	Process PDF: convert pages, apply heuristic per page, combine results.
	
	Args:
		pdf_path: Path to PDF file
		engine: Engine selection ('auto', 'printed', 'handwriting')
		trocr: Optional TrOCR engine instance (for reuse)
	
	Returns:
		Tuple of (engine_name, combined_text, page_count)
	"""
	if not PDF2IMAGE_AVAILABLE:
		raise ImportError("pdf2image is not installed. Install it with: pip install pdf2image")
	
	# Convert PDF to images
	pages = convert_pdf_to_images(pdf_path)
	page_count = len(pages)
	
	if page_count == 0:
		raise ValueError(f"No pages found in PDF: {pdf_path}")
	
	# Process each page
	page_texts = []
	engine_used = None
	
	if trocr is None and engine != 'printed':
		trocr = TrOCREngine()
	
	for page_num, page_image in enumerate(pages, 1):
		try:
			# Determine engine for this page
			page_engine = engine
			if engine == 'auto':
				page_engine = simple_heuristic(page_image)
			
			# Track which engine is being used (use first page's engine as primary)
			if engine_used is None:
				engine_used = page_engine
			
			# Perform OCR on this page
			page_text = ocr_pil_image(page_image, page_engine, trocr)
			
			# Add page separator
			page_texts.append(f"\n\n--- Page {page_num} ---\n\n{page_text}")
			
		except Exception as e:
			# Continue processing other pages if one fails
			page_texts.append(f"\n\n--- Page {page_num} ---\n\n[Error processing page: {e}]")
	
	# Combine all pages
	combined_text = ''.join(page_texts).strip()
	
	# Use the determined engine (or specified engine if not auto)
	# If auto mode and no engine was determined (all pages failed), default to 'printed'
	if engine == 'auto':
		final_engine = engine_used if engine_used is not None else 'printed'
	else:
		final_engine = engine
	
	return final_engine, combined_text, page_count


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(
		description='Route OCR to printed (Tesseract) or handwriting (TrOCR). Supports images and PDFs.'
	)
	parser.add_argument(
		'file_path', 
		nargs='?', 
		help='Path to a single image or PDF file; if omitted, process images/ directory'
	)
	parser.add_argument(
		'--engine', 
		choices=['auto', 'printed', 'handwriting'], 
		default='auto', 
		help='Engine selection (default: auto)'
	)
	parser.add_argument(
		'--outdir', 
		default='outputs', 
		help='Directory to write outputs for batch mode'
	)
	parser.add_argument(
		'--csv',
		action='store_true',
		help='Generate CSV summary file (batch mode only)'
	)
	args = parser.parse_args()

	# Check PDF support
	if not PDF2IMAGE_AVAILABLE:
		print("Warning: pdf2image is not installed. PDF support will be limited.")
		print("Install it with: pip install pdf2image")

	# Single file mode
	if args.file_path:
		file_path = args.file_path
		if not os.path.exists(file_path):
			print(f"Error: File not found: {file_path}")
			return
		
		if is_pdf(file_path) and not PDF2IMAGE_AVAILABLE:
			print("Error: PDF support requires pdf2image. Install it with: pip install pdf2image")
			return
		
		try:
			engine_name, text, page_count = route_ocr(file_path, engine=args.engine)
			file_type = "PDF" if is_pdf(file_path) else "Image"
			print(f"[{engine_name}] {file_path} ({file_type}, {page_count} page{'s' if page_count > 1 else ''})")
			print('=' * 60)
			print(text)
		except Exception as e:
			print(f"Error processing {file_path}: {e}")
		return

	# Batch mode - process images/ directory
	images_dir = Path(__file__).resolve().parent / 'images'
	files = collect_images(images_dir)
	if not files:
		print("No images or PDFs found in the 'images' directory.")
		return
	
	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	# CSV summary file (if requested)
	csv_file = None
	csv_f = None
	csv_writer = None
	if args.csv:
		import csv
		csv_file = outdir / 'summary.csv'
		csv_f = csv_file.open('w', newline='', encoding='utf-8')
		csv_writer = csv.writer(csv_f)
		csv_writer.writerow(['file', 'engine', 'chars', 'pages', 'type'])

	# Reuse TrOCR engine for efficiency
	trocr = TrOCREngine()
	
	for file_path in files:
		try:
			if is_pdf(file_path) and not PDF2IMAGE_AVAILABLE:
				print(f"Skipping {file_path}: PDF support requires pdf2image")
				continue
			
			engine_name, text, page_count = route_ocr(file_path, engine=args.engine, trocr=trocr)
			
			# Generate output filename
			file_stem = Path(file_path).stem
			out_file = outdir / f"OCR_{file_stem}.txt"
			
			out_file.write_text(text, encoding='utf-8')
			
			# Write to CSV if requested
			if csv_writer:
				file_type = "PDF" if is_pdf(file_path) else "Image"
				csv_writer.writerow([file_path, engine_name, len(text), page_count, file_type])
			
			file_type = "PDF" if is_pdf(file_path) else "Image"
			print(f"[{engine_name}] {file_path} ({file_type}, {page_count} page{'s' if page_count > 1 else ''})")
		except Exception as e:
			print(f"Error processing {file_path}: {e}")
			if csv_writer:
				csv_writer.writerow([file_path, 'error', 0, 0, 'error'])
	
	# Close CSV file if opened
	if csv_file and csv_f:
		csv_f.close()
		print(f"\nCSV summary saved to: {csv_file}")


if __name__ == '__main__':
	# Windows console encoding
	if sys.platform == 'win32':
		sys.stdout.reconfigure(encoding='utf-8')
	main()


