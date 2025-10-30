import os
import sys
from typing import List, Optional
from pathlib import Path
import ollama

# Windows console encoding
if sys.platform == 'win32':
	sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_MODEL = 'llama3.2-vision:latest'
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def collect_images(images_dir: Path) -> List[str]:
	if not images_dir.exists():
		return []
	return [str(p) for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]


def ensure_image(path: str) -> None:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Image file '{path}' not found")
	if Path(path).suffix.lower() not in VALID_EXTS:
		raise ValueError(f"Unsupported file format: {Path(path).suffix}")


def extract_text(
    image_path: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.001,
    seed: Optional[int] = 1,
) -> str:
	ensure_image(image_path)
	resp = ollama.chat(
		model=model,
		messages=[
			{'role': 'system', 'content': 'You are an OCR assistant. Extract all visible text accurately.'},
			{'role': 'user', 'content': 'Extract all text from this image', 'images': [image_path]},
		],
		options={
			'temperature': temperature,
			'seed': seed,
		},
	)
	return getattr(resp.message, 'content', str(resp))


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(description='Local OCR using llama3.2-vision (Ollama)')
	parser.add_argument('image_path', nargs='?', help='Path to a single image to process')
	parser.add_argument('--model', default=DEFAULT_MODEL, help='Ollama model to use')
	parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature (lower = more deterministic)')
	parser.add_argument('--seed', type=int, default=1, help='Random seed for determinism')
	args = parser.parse_args()

	model = args.model
	if args.image_path:
		# Single image mode
		print(f"Processing: {args.image_path}")
		print(f"Model: {model}")
		print('=' * 60)
		try:
			text = extract_text(args.image_path, model=model, temperature=args.temperature, seed=args.seed)
			print(text)
		except Exception as e:
			print(f"Error: {e}")
			sys.exit(1)
		return

	# Batch mode over images directory
	images_dir = Path(__file__).resolve().parent / 'images'
	images = collect_images(images_dir)
	if not images:
		print("No images found in the 'images' directory.")
		return

	print(f"Model: {model}")
	print('=' * 60)
	for img in images:
		print(f"\n-- {img}")
		try:
			text = extract_text(img, model=model, temperature=args.temperature, seed=args.seed)
			print(text if len(text) <= 1200 else text[:1200] + '...')
		except Exception as e:
			print(f"Error: {e}")


if __name__ == '__main__':
	main()


