from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from surya.common.surya.schema import TaskNames
from surya.input.load import load_image
from surya.input.processing import convert_if_not_rgb
import os
from PIL import Image

def main(input_file_path):
    # Check if input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' not found!")
        return
    
    print(f"Starting OCR processing for: {input_file_path}")
    
    # Load models
    print("Loading foundation model...")
    foundation_predictor = FoundationPredictor()
    print("Loading detection model...")
    det_predictor = DetectionPredictor()
    print("Loading recognition model...")
    rec_predictor = RecognitionPredictor(foundation_predictor)
    
    # Check file extension to determine processing method
    file_ext = os.path.splitext(input_file_path)[1].lower()
    images = []
    
    if file_ext == '.pdf':
        print("Processing PDF file...")
        # If you have a PDF, you need to convert it to images first
        # You can use pypdfium2 or pdf2image for this
        try:
            import pypdfium2 as pdfium
            import gc
            
            pdf = pdfium.PdfDocument(input_file_path)
            pages = []
            
            try:
                # Render all pages first
                for i in range(len(pdf)):
                    page = pdf[i]
                    pil_image = page.render(scale=2.0).to_pil()
                    images.append(pil_image)
                    pages.append(page)
                
                # Close all pages
                for page in pages:
                    page.close()
                pages.clear()
                
            finally:
                # Always close the PDF document
                pdf.close()
                # Force garbage collection
                gc.collect()
            
            print(f"PDF converted to {len(images)} page(s)")
        except ImportError:
            print("pypdfium2 not installed. Please install it with: pip install pypdfium2")
            return
    
    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        print("Processing image file...")
        # Load single image file
        try:
            image = Image.open(input_file_path)
            images = [image]
            print(f"Image loaded: {input_file_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return
    
    else:
        print(f"Unsupported file format: {file_ext}")
        print("Supported formats: PDF, JPG, JPEG, PNG, BMP, TIFF")
        return
    
    # Convert all images to RGB format
    images = convert_if_not_rgb(images)
    
    # Set task name for OCR
    task_names = [TaskNames.ocr_with_boxes] * len(images)
    
    # Run OCR with enhanced settings for better column detection
    print("Running OCR...")
    predictions = rec_predictor(
        images,
        task_names=task_names,
        det_predictor=det_predictor,
        math_mode=True
    )
    print("OCR completed")
    
    # Print extracted text per page
    print("\n" + "="*50)
    print("EXTRACTED TEXT RESULTS")
    print("="*50)
    
    # Also save to file with input filename
    input_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_file = f"{input_name}_extract.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("EXTRACTED TEXT RESULTS\n")
        f.write("="*50 + "\n\n")
        
        for page_idx, prediction in enumerate(predictions):
            page_header = f"--- Page {page_idx + 1} ---"
            print(f"\n{page_header}")
            f.write(f"{page_header}\n")
            
            if not prediction.text_lines:
                no_text_msg = "No text detected on this page"
                print(no_text_msg)
                f.write(f"{no_text_msg}\n")
                continue
                
            # Process text lines and try to improve column reading
            all_lines = []
            for line in prediction.text_lines:
                # Filter low-confidence lines (like "DELETED" artifacts)
                if line.confidence > 0.75:
                    all_lines.append((line.text, line.confidence))
            
            # Try to pair items with prices by looking for price patterns
            processed_lines = []
            i = 0
            while i < len(all_lines):
                text, conf = all_lines[i]
                
                # Check if this line contains a price (starts with $)
                if text.strip().startswith('$'):
                    # This is a price line, try to pair with previous line
                    if processed_lines and not processed_lines[-1].startswith('$'):
                        # Combine previous line with this price
                        prev_text = processed_lines[-1].split('[conf:')[0].strip()
                        combined = f"{prev_text}: {text}"
                        processed_lines[-1] = f"[conf: {conf:.2f}] {combined}"
                    else:
                        processed_lines.append(f"[conf: {conf:.2f}] {text}")
                else:
                    processed_lines.append(f"[conf: {conf:.2f}] {text}")
                i += 1
            
            # Output processed lines
            for line in processed_lines:
                print(line)
                f.write(f"{line}\n")
    
    print(f"\nOCR processing completed successfully!")
    print(f"Results also saved to: {output_file}")
    
    # Final cleanup
    import gc
    gc.collect()

if __name__ == "__main__":
    import sys
    # Require command line argument for file path
    if len(sys.argv) < 2:
        print("Usage: python test_SuryaOCR.py <file_path>")
        print("Example: python test_SuryaOCR.py images/handwriting.jpg")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    main(input_file_path)