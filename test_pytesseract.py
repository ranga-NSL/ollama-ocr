import pytesseract
import os
import sys
from PIL import Image
import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess image for better OCR results
    """
    # Load image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.medianBlur(binary, 3)
    
    return denoised

def main(input_file_path):
    # Check if input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' not found!")
        return
    
    print(f"Starting PyTesseract OCR processing for: {input_file_path}")
    
    # Check file extension to determine processing method
    file_ext = os.path.splitext(input_file_path)[1].lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        print("Processing image file...")
        
        # Load image with PIL
        try:
            image = Image.open(input_file_path)
            print(f"Image loaded: {input_file_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return
        
        # Preprocess image for better OCR
        print("Preprocessing image...")
        processed_image = preprocess_image(input_file_path)
        
        # Convert back to PIL Image
        processed_pil = Image.fromarray(processed_image)
        
        # Configure Tesseract
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?$%&()[]{}:;@#-+=/ '
        
        print("Running OCR...")
        try:
            # Extract text with confidence scores
            data = pytesseract.image_to_data(processed_pil, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract text
            text = pytesseract.image_to_string(processed_pil, config=custom_config)
            
            print("OCR completed")
            
        except Exception as e:
            print(f"OCR processing failed: {e}")
            return
    
    elif file_ext == '.pdf':
        print("PDF processing not implemented in this version.")
        print("Please convert PDF to images first or use a PDF-specific OCR tool.")
        return
    
    else:
        print(f"Unsupported file format: {file_ext}")
        print("Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        return
    
    # Process and display results
    print("\n" + "="*50)
    print("EXTRACTED TEXT RESULTS")
    print("="*50)
    
    # Create output filename
    input_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_file = f"{input_name}_pytesseract_extract.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("EXTRACTED TEXT RESULTS (PyTesseract)\n")
        f.write("="*50 + "\n\n")
        
        # Write full text
        f.write("FULL TEXT:\n")
        f.write("-" * 30 + "\n")
        f.write(text)
        f.write("\n\n")
        
        # Write detailed results with confidence scores
        f.write("DETAILED RESULTS WITH CONFIDENCE:\n")
        f.write("-" * 40 + "\n")
        
        # Process data with confidence scores
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # Filter low confidence
                text_line = data['text'][i].strip()
                if text_line:
                    conf = data['conf'][i]
                    f.write(f"[conf: {conf}] {text_line}\n")
                    print(f"[conf: {conf}] {text_line}")
    
    print(f"\nOCR processing completed successfully!")
    print(f"Results saved to: {output_file}")
    
    # Also print the full text
    print("\n" + "="*50)
    print("FULL EXTRACTED TEXT:")
    print("="*50)
    print(text)

if __name__ == "__main__":
    # Require command line argument for file path
    if len(sys.argv) < 2:
        print("Usage: python test_pytesseract.py <file_path>")
        print("Example: python test_pytesseract.py images/handwriting.jpg")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    main(input_file_path)
