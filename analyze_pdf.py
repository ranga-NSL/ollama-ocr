import ollama
from pdf2image import convert_from_path
import io
from pathlib import Path
import sys

# Set Windows console encoding for proper output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def analyze_pdf_page(pdf_path, page_number=1, prompt="Describe this page"):
    """
    Analyze a PDF page using Ollama's vision model.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to analyze (1-indexed)
        prompt: Prompt to send to the vision model
    
    Returns:
        str: Response from the vision model
    """
    # 1. Convert PDF page to image (in memory, no temp files needed)
    # distinct from saving to disk, we convert to bytes directly
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    
    if not images:
        return "Error: Could not read page."
    
    img = images[0]
    
    # 2. Convert PIL image to bytes for Ollama
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    # 3. Send to Llama 3.2 Vision via Ollama
    response = ollama.chat(
        model='llama3.2-vision:latest',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [img_bytes]
        }]
    )
    
    return response['message']['content']


if __name__ == "__main__":
    # Usage example with playbook_1.pdf
    pdf_file = Path(__file__).parent / "playbook_1.pdf"
    
    if not pdf_file.exists():
        print(f"Error: {pdf_file} not found")
    else:
        result = analyze_pdf_page(
            str(pdf_file), 
            page_number=1, 
            prompt="Summarize the technical specs table."
        )
        print(result)

