import os
import sys
from pathlib import Path
from typing import List, Literal

import cv2

from ocr_printed import ocr_image as ocr_printed
from ocr_handwriting_trocr import TrOCREngine


Engine = Literal['auto', 'printed', 'handwriting']
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def collect_images(images_dir: Path) -> List[str]:
	if not images_dir.exists():
		return []
	return [str(p) for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]


def simple_heuristic(image_path: str) -> Engine:
	# Very lightweight heuristic: if filename hints at receipt/form or many straight lines -> printed
	name = Path(image_path).name.lower()
	if any(k in name for k in ['receipt', 'invoice', 'form']):
		return 'printed'
	img = cv2.imread(image_path)
	if img is None:
		return 'printed'
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 150)
	lines = cv2.HoughLinesP(edges, 1, cv2.THRESH_BINARY, threshold=80, minLineLength=60, maxLineGap=5)
	if lines is not None and len(lines) > 15:
		return 'printed'
	return 'handwriting'


def route_ocr(image_path: str, engine: Engine = 'auto', trocr: TrOCREngine | None = None) -> tuple[str, str]:
	chosen: Engine = engine
	if engine == 'auto':
		chosen = simple_heuristic(image_path)
	if chosen == 'printed':
		return 'printed', ocr_printed(image_path)
	else:
		if trocr is None:
			trocr = TrOCREngine()
		return 'handwriting', trocr.ocr(image_path)


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(description='Route OCR to printed (Tesseract) or handwriting (TrOCR)')
	parser.add_argument('image_path', nargs='?', help='Path to a single image; if omitted, process images/ directory')
	parser.add_argument('--engine', choices=['auto', 'printed', 'handwriting'], default='auto', help='Engine selection')
	parser.add_argument('--outdir', default='outputs', help='Directory to write outputs for batch mode')
	args = parser.parse_args()

	if args.image_path:
		engine_name, text = route_ocr(args.image_path, engine=args.engine)
		print(f"[{engine_name}] {args.image_path}")
		print('=' * 60)
		print(text)
		return

	images_dir = Path(__file__).resolve().parent / 'images'
	images = collect_images(images_dir)
	if not images:
		print("No images found in the 'images' directory.")
		return
	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	trocr = TrOCREngine()
	for img in images:
		try:
			engine_name, text = route_ocr(img, engine=args.engine, trocr=trocr)
			out_file = outdir / (Path(img).stem + f'.{engine_name}.txt')
			out_file.write_text(text, encoding='utf-8')
			print(f"[{engine_name}] {img}")
		except Exception as e:
			print(f"Error {img}: {e}")


if __name__ == '__main__':
	# Windows console encoding
	if sys.platform == 'win32':
		sys.stdout.reconfigure(encoding='utf-8')
	main()


