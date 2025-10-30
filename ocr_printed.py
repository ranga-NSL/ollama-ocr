import os
import sys
from pathlib import Path
from typing import List

import cv2
import pytesseract


VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def collect_images(images_dir: Path) -> List[str]:
	if not images_dir.exists():
		return []
	return [str(p) for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]


def preprocess(img):
	# Grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Otsu threshold (binary)
	_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# Optional slight dilation to connect characters
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
	proc = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
	return proc


def ocr_image(image_path: str) -> str:
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Image file '{image_path}' not found")
	img = cv2.imread(image_path)
	if img is None:
		raise ValueError(f"Failed to read image: {image_path}")
	proc = preprocess(img)
	config = '--psm 6'  # Assume a block of text
	text = pytesseract.image_to_string(proc, config=config)
	return text.strip()


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(description='Printed-text OCR using PyTesseract + OpenCV preprocessing')
	parser.add_argument('image_path', nargs='?', help='Path to a single image; if omitted, process images/ directory')
	parser.add_argument('--outdir', default='outputs', help='Directory to write outputs for batch mode')
	args = parser.parse_args()

	if args.image_path:
		print(f"Processing (printed): {args.image_path}")
		print('=' * 60)
		print(ocr_image(args.image_path))
		return

	images_dir = Path(__file__).resolve().parent / 'images'
	images = collect_images(images_dir)
	if not images:
		print("No images found in the 'images' directory.")
		return

	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	for img in images:
		print(f"\n-- {img}")
		try:
			text = ocr_image(img)
			out_file = outdir / (Path(img).stem + '.printed.txt')
			out_file.write_text(text, encoding='utf-8')
			print(text if len(text) <= 1000 else text[:1000] + '...')
		except Exception as e:
			print(f"Error: {e}")


if __name__ == '__main__':
	# Windows console encoding
	if sys.platform == 'win32':
		sys.stdout.reconfigure(encoding='utf-8')
	main()


