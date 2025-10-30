import csv
from pathlib import Path
import sys

from ocr_router import collect_images, route_ocr


def main() -> None:
	images_dir = Path(__file__).resolve().parent / 'images'
	outputs = Path(__file__).resolve().parent / 'outputs'
	outputs.mkdir(parents=True, exist_ok=True)

	images = collect_images(images_dir)
	if not images:
		print("No images found in the 'images' directory.")
		return

	summary_csv = outputs / 'summary.csv'
	with summary_csv.open('w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(['image', 'engine', 'chars'])
		for img in images:
			try:
				engine, text = route_ocr(img, engine='auto')
				(out_text := outputs / (Path(img).stem + f'.{engine}.txt')).write_text(text, encoding='utf-8')
				writer.writerow([img, engine, len(text)])
				print(f"[{engine}] -> {out_text}")
			except Exception as e:
				print(f"Error {img}: {e}")


if __name__ == '__main__':
	# Windows console encoding
	if sys.platform == 'win32':
		sys.stdout.reconfigure(encoding='utf-8')
	main()


