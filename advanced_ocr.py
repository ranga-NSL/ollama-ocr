import ollama
import os
import json
import sys
from typing import Dict, List, Optional

class AdvancedImageAnalyzer:
    """
    Advanced image analyzer using llama3.2-vision with multiple analysis capabilities
    Based on the comprehensive guide recommendations
    """
    #llama3.2-vision
    # minicpm-v:latest  # Not that good
    # granite3.2-vision:2b #not that good
    def __init__(self, model: str = 'llama3.2-vision:latest'):
        self.model = model
        self.valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def _validate_image(self, image_path: str) -> None:
        """Validate image file exists and has supported format"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found!")
        
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.valid_extensions:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def extract_text(self, image_path: str, output_format: str = 'text') -> Dict:
        """
        Extract text from image with optional JSON output
        """
        self._validate_image(image_path)
        
        if output_format == 'json':
            system_prompt = 'You are an OCR system. Extract all text and return as JSON with "text" field.'
            user_prompt = 'Extract all visible text from this image. Return as JSON: {"text": "extracted_text"}'
        else:
            system_prompt = 'You are a specialized OCR assistant. Extract all visible text accurately.'
            user_prompt = 'Extract all text from this image'
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt, 'images': [image_path]}
                ]
            )
            return response
        except Exception as e:
            raise RuntimeError(f"OCR processing failed: {str(e)}")
    
    def transcribe_handwriting(self, image_path: str) -> Dict:
        """
        Specialized handwriting transcription
        """
        self._validate_image(image_path)
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Transcribe all handwritten text in this image. Maintain original formatting and structure.',
                    'images': [image_path]
                }]
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Handwriting transcription failed: {str(e)}")
    
    def extract_structured_data(self, image_path: str, data_type: str = 'receipt') -> Dict:
        """
        Extract structured data from images (receipts, forms, etc.)
        """
        self._validate_image(image_path)
        
        prompts = {
            'receipt': {
                'system': 'Extract receipt data and return as JSON with fields: store_name, total_amount, date, items (array of {name, price, quantity})',
                'user': 'Extract all receipt information and return as structured JSON'
            },
            'form': {
                'system': 'Extract form field data and return as JSON with field names and values',
                'user': 'Extract all form field data and return as structured JSON'
            },
            'price_tag': {
                'system': 'Extract product information and return as JSON with fields: product_name, price, sku',
                'user': 'Extract product name, price, and SKU from this price tag'
            }
        }
        
        if data_type not in prompts:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': prompts[data_type]['system']},
                    {'role': 'user', 'content': prompts[data_type]['user'], 'images': [image_path]}
                ]
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Structured data extraction failed: {str(e)}")
    
    def analyze_document(self, image_path: str) -> Dict:
        """
        Comprehensive document analysis
        """
        self._validate_image(image_path)
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Analyze this document comprehensively. Include: document type, key information, text content, layout structure, and any notable features.',
                    'images': [image_path]
                }]
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Document analysis failed: {str(e)}")
    
    def multi_step_analysis(self, image_path: str) -> Dict[str, Dict]:
        """
        Perform multiple analysis steps on the same image
        """
        self._validate_image(image_path)
        
        results = {}
        
        # Step 1: Basic description
        try:
            results['description'] = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Provide a basic description of this image',
                    'images': [image_path]
                }]
            )
        except Exception as e:
            results['description'] = {'error': str(e)}
        
        # Step 2: Text extraction
        try:
            results['text_extraction'] = self.extract_text(image_path)
        except Exception as e:
            results['text_extraction'] = {'error': str(e)}
        
        # Step 3: Object identification
        try:
            results['objects'] = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'List all objects and items visible in this image',
                    'images': [image_path]
                }]
            )
        except Exception as e:
            results['objects'] = {'error': str(e)}
        
        return results

def main():
    """Example usage of the AdvancedImageAnalyzer"""
    analyzer = AdvancedImageAnalyzer()
    
    # Test with different image types
    test_images = [
        ('images/handwriting.jpg', 'handwriting'),
        ('images/trader-joes-receipt.jpg', 'receipt'),
        ('images/test_notes.jpg', 'notes'),
        ('images/test_dwg.jpg', 'drawing')
    ]
    
    for image_path, image_type in test_images:
        if not os.path.exists(image_path):
            print(f"Skipping {image_path} - file not found")
            continue
            
        print(f"\n{'='*50}")
        print(f"Analyzing: {image_path} ({image_type})")
        print(f"{'='*50}")
        
        try:
            # Basic text extraction
            print("\n1. Text Extraction:")
            text_result = analyzer.extract_text(image_path)
            print(text_result.message.content[:200] + "..." if len(text_result.message.content) > 200 else text_result.message.content)
            
            # Document analysis
            print("\n2. Document Analysis:")
            doc_result = analyzer.analyze_document(image_path)
            print(doc_result.message.content[:200] + "..." if len(doc_result.message.content) > 200 else doc_result.message.content)
            
            # Structured data extraction for receipts
            if image_type == 'receipt':
                print("\n3. Structured Data Extraction:")
                try:
                    struct_result = analyzer.extract_structured_data(image_path, 'receipt')
                    print(struct_result.message.content)
                except Exception as e:
                    print(f"Structured extraction failed: {e}")
            
        except Exception as e:
            print(f"Analysis failed: {e}")

if __name__ == "__main__":
    import sys
    # Allow command line argument for specific image
    if len(sys.argv) > 1:
        # Process single image
        image_path = sys.argv[1]
        analyzer = AdvancedImageAnalyzer()
        
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found!")
            sys.exit(1)
        
        print(f"Processing: {image_path}")
        try:
            # Basic text extraction
            print("\n1. Text Extraction:")
            text_result = analyzer.extract_text(image_path)
            print(text_result.message.content)
            
            # Document analysis
            print("\n2. Document Analysis:")
            doc_result = analyzer.analyze_document(image_path)
            print(doc_result.message.content)
            
        except Exception as e:
            print(f"Analysis failed: {e}")
    else:
        # Process all test images
        main() 