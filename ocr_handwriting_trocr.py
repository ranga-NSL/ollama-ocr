import os
import sys
from pathlib import Path
from typing import List

from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def collect_images(images_dir: Path) -> List[str]:
	if not images_dir.exists():
		return []
	return [str(p) for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]


class TrOCREngine:
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


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(description='Handwriting OCR using Hugging Face TrOCR')
	parser.add_argument('image_path', nargs='?', help='Path to a single image; if omitted, process images/ directory')
	parser.add_argument('--model', default='microsoft/trocr-base-handwritten', help='TrOCR model name')
	parser.add_argument('--outdir', default='outputs', help='Directory to write outputs for batch mode')
	args = parser.parse_args()

	engine = TrOCREngine(model_name=args.model)

	if args.image_path:
		print(f"Processing (handwriting): {args.image_path}")
		print('=' * 60)
		print(engine.ocr(args.image_path))
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
			text = engine.ocr(img)
			out_file = outdir / (Path(img).stem + '.handwriting.txt')
			out_file.write_text(text, encoding='utf-8')
			print(text if len(text) <= 1000 else text[:1000] + '...')
		except Exception as e:
			print(f"Error: {e}")


if __name__ == '__main__':
	# Windows console encoding
	if sys.platform == 'win32':
		sys.stdout.reconfigure(encoding='utf-8')
	main()


