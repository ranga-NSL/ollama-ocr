import os
import sys
import base64
from ollama import Client
from typing import Dict, Optional, List
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Recommended vision/OCR-capable cloud models
CLOUD_VISION_MODELS = [
    'qwen3-vl:235b-cloud',  # Vision-language model (OCR capable)
    'minimax-m2:cloud',     # Cloud LLM (may be text-only)
    'gpt-oss:120b-cloud'    # Cloud LLM (may be text-only)
]

class CloudOCRAnalyzer:
    """
    Cloud-based OCR analyzer using Ollama Cloud API
    Provides OCR capabilities for images using cloud-hosted vision models
    """
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Cloud OCR Analyzer
        
        Args:
            model: Optional cloud model to use; defaults to the first in CLOUD_VISION_MODELS.
            api_key: Ollama API key (defaults to OLLAMA_API_KEY environment variable)
        """
        self.api_key = api_key or os.environ.get('OLLAMA_API_KEY')
        if not self.api_key:
            raise ValueError("OLLAMA_API_KEY environment variable not set or provided")
        
        # Use provided model or default to the first entry in CLOUD_VISION_MODELS
        self.model = model or CLOUD_VISION_MODELS[0]
        
        # Warn if using a non-vision model
        if '-vl' not in model.lower() and 'vision' not in model.lower():
            print(f"[WARNING] Model '{model}' may not support vision/OCR capabilities.")
            print(f"         Recommended vision models: {', '.join(CLOUD_VISION_MODELS[:2])}")
        
        self.valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Create client for Ollama Cloud
        self.client = Client(
            host='https://ollama.com',
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
    
    def _validate_image(self, image_path: str) -> None:
        """Validate image file exists and has supported format"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found!")
        
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.valid_extensions:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string for Cloud API
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string (raw, without data URI prefix)
        """
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return base64_image
    
    def _get_response_content(self, response, stream: bool = False) -> str:
        """Extract content from response, handling both streaming and non-streaming"""
        if stream:
            content = ""
            for chunk in response:
                if isinstance(chunk, dict) and 'message' in chunk:
                    content += chunk['message'].get('content', '')
                elif hasattr(chunk, 'message'):
                    content += getattr(chunk.message, 'content', '')
            return content
        else:
            if isinstance(response, dict):
                return response.get('message', {}).get('content', '')
            elif hasattr(response, 'message'):
                return response.message.content if hasattr(response.message, 'content') else str(response.message)
            return str(response)

    def _collect_project_images(self, images_dir: Optional[str] = None) -> List[str]:
        """
        Collect image file paths from the project's images directory
        """
        base_dir = Path(__file__).resolve().parent
        dir_path = Path(images_dir) if images_dir else base_dir / 'images'
        if not dir_path.exists():
            return []
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return [str(p) for p in sorted(dir_path.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    
    def extract_text(self, image_path: str, output_format: str = 'text', stream: bool = False) -> Dict:
        """
        Extract text from image with optional JSON output
        
        Args:
            image_path: Path to image file
            output_format: 'text' or 'json'
            stream: Whether to stream the response
            
        Returns:
            Dictionary with response data
        """
        self._validate_image(image_path)
        
        if output_format == 'json':
            system_prompt = 'You are an OCR system. Extract all text and return as JSON with "text" field.'
            user_prompt = 'Extract all visible text from this image. Return as JSON: {"text": "extracted_text"}'
        else:
            system_prompt = 'You are a specialized OCR assistant. Extract all visible text accurately.'
            user_prompt = 'Extract all text from this image'
        
        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt, 'images': [base64_image]}
                ],
                stream=stream
            )
            
            if stream:
                content = self._get_response_content(response, stream=True)
                return {'content': content, 'model': self.model}
            else:
                content = self._get_response_content(response, stream=False)
                return {'content': content, 'model': self.model}
                
        except Exception as e:
            raise RuntimeError(f"OCR processing failed: {str(e)}")
    
    def transcribe_handwriting(self, image_path: str, stream: bool = False) -> Dict:
        """
        Specialized handwriting transcription
        
        Args:
            image_path: Path to image file
            stream: Whether to stream the response
            
        Returns:
            Dictionary with transcribed text
        """
        self._validate_image(image_path)
        
        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Transcribe all handwritten text in this image. Maintain original formatting and structure.',
                    'images': [base64_image]
                }],
                stream=stream
            )
            
            if stream:
                content = self._get_response_content(response, stream=True)
                print(content, end='', flush=True)
                print()  # Newline after streaming
                return {'content': content, 'model': self.model}
            else:
                content = self._get_response_content(response, stream=False)
                return {'content': content, 'model': self.model}
                
        except Exception as e:
            raise RuntimeError(f"Handwriting transcription failed: {str(e)}")
    
    def extract_structured_data(self, image_path: str, data_type: str = 'receipt', stream: bool = False) -> Dict:
        """
        Extract structured data from images (receipts, forms, etc.)
        
        Args:
            image_path: Path to image file
            data_type: Type of structured data ('receipt', 'form', 'price_tag')
            stream: Whether to stream the response
            
        Returns:
            Dictionary with extracted structured data
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
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': prompts[data_type]['system']},
                    {'role': 'user', 'content': prompts[data_type]['user'], 'images': [base64_image]}
                ],
                stream=stream
            )
            
            if stream:
                content = self._get_response_content(response, stream=True)
                print(content, end='', flush=True)
                print()  # Newline after streaming
                return {'content': content, 'model': self.model}
            else:
                content = self._get_response_content(response, stream=False)
                return {'content': content, 'model': self.model}
                
        except Exception as e:
            raise RuntimeError(f"Structured data extraction failed: {str(e)}")
    
    def analyze_document(self, image_path: str, stream: bool = False) -> Dict:
        """
        Comprehensive document analysis
        
        Args:
            image_path: Path to image file
            stream: Whether to stream the response
            
        Returns:
            Dictionary with document analysis
        """
        self._validate_image(image_path)
        
        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Analyze this document comprehensively. Include: document type, key information, text content, layout structure, and any notable features.',
                    'images': [base64_image]
                }],
                stream=stream
            )
            
            if stream:
                content = self._get_response_content(response, stream=True)
                print(content, end='', flush=True)
                print()  # Newline after streaming
                return {'content': content, 'model': self.model}
            else:
                content = self._get_response_content(response, stream=False)
                return {'content': content, 'model': self.model}
                
        except Exception as e:
            raise RuntimeError(f"Document analysis failed: {str(e)}")
    
    def multi_step_analysis(self, image_path: str, stream: bool = False) -> Dict[str, Dict]:
        """
        Perform multiple analysis steps on the same image
        
        Args:
            image_path: Path to image file
            stream: Whether to stream the responses
            
        Returns:
            Dictionary with results from all analysis steps
        """
        self._validate_image(image_path)
        
        # Encode image once for all steps
        base64_image = self._encode_image_to_base64(image_path)
        
        results = {}
        
        # Step 1: Basic description
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Provide a basic description of this image',
                    'images': [base64_image]
                }],
                stream=stream
            )
            results['description'] = {'content': self._get_response_content(response, stream=stream)}
        except Exception as e:
            results['description'] = {'error': str(e)}
        
        # Step 2: Text extraction
        try:
            results['text_extraction'] = self.extract_text(image_path, stream=stream)
        except Exception as e:
            results['text_extraction'] = {'error': str(e)}
        
        # Step 3: Object identification
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'List all objects and items visible in this image',
                    'images': [base64_image]
                }],
                stream=stream
            )
            results['objects'] = {'content': self._get_response_content(response, stream=stream)}
        except Exception as e:
            results['objects'] = {'error': str(e)}
        
        return results

def main():
    """Run OCR over project images using the configured cloud models"""
    try:
        images = CloudOCRAnalyzer(model=CLOUD_VISION_MODELS[0])._collect_project_images()
        if not images:
            print("No images found in the project's 'images' directory.")
            return

        for model in CLOUD_VISION_MODELS:
            print(f"\n{'='*60}")
            print(f"Model: {model}")
            print(f"{'='*60}")
            analyzer = CloudOCRAnalyzer(model=model)

            for image_path in images:
                print(f"\n{'-'*60}")
                print(f"Analyzing: {image_path}")
                print(f"{'-'*60}")
                try:
                    # Text extraction
                    print(f"1) Text Extraction [{analyzer.model}]:")
                    text_result = analyzer.extract_text(image_path, stream=False)
                    content = text_result.get('content', '')
                    print(content[:400] + "..." if len(content) > 400 else content)

                    # Document analysis
                    print(f"\n2) Document Analysis [{analyzer.model}]:")
                    doc_result = analyzer.analyze_document(image_path, stream=False)
                    content = doc_result.get('content', '')
                    print(content[:400] + "..." if len(content) > 400 else content)

                    # Structured data heuristic: if 'receipt' in filename
                    if 'receipt' in Path(image_path).name.lower():
                        print(f"\n3) Structured Data (receipt) [{analyzer.model}]:")
                        try:
                            struct_result = analyzer.extract_structured_data(image_path, 'receipt', stream=False)
                            print(struct_result.get('content', ''))
                        except Exception as e:
                            print(f"Structured extraction failed: {e}")
                except Exception as e:
                    print(f"Analysis failed: {e}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set OLLAMA_API_KEY environment variable or provide it to the constructor")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cloud-based OCR using Ollama Cloud API',
        epilog=f"Models: {', '.join(CLOUD_VISION_MODELS)}"
    )
    parser.add_argument('image_path', nargs='?', help='Path to image file to process')
    parser.add_argument('--model', default=None, 
                       help='Cloud model to use for OCR; if omitted, iterate through predefined models')
    parser.add_argument('--stream', action='store_true', help='Stream the response')
    parser.add_argument('--mode', choices=['text', 'handwriting', 'structured', 'document', 'multi'],
                       default='text', help='OCR mode to use')
    parser.add_argument('--data-type', choices=['receipt', 'form', 'price_tag'],
                       default='receipt', help='Data type for structured extraction')
    
    args = parser.parse_args()
    
    try:
        analyzer = CloudOCRAnalyzer(model=args.model or CLOUD_VISION_MODELS[0])
        
        if args.image_path:
            # Process single image
            if not os.path.exists(args.image_path):
                print(f"Error: Image file '{args.image_path}' not found!")
                sys.exit(1)
            
            print(f"Processing: {args.image_path}")
            print(f"Model: {args.model}")
            print(f"{'='*60}\n")
            
            try:
                if args.mode == 'text':
                    result = analyzer.extract_text(args.image_path, stream=args.stream)
                    if not args.stream:
                        print(result.get('content', ''))
                        
                elif args.mode == 'handwriting':
                    result = analyzer.transcribe_handwriting(args.image_path, stream=args.stream)
                    if not args.stream:
                        print(result.get('content', ''))
                        
                elif args.mode == 'structured':
                    result = analyzer.extract_structured_data(args.image_path, args.data_type, stream=args.stream)
                    if not args.stream:
                        print(result.get('content', ''))
                        
                elif args.mode == 'document':
                    result = analyzer.analyze_document(args.image_path, stream=args.stream)
                    if not args.stream:
                        print(result.get('content', ''))
                        
                elif args.mode == 'multi':
                    results = analyzer.multi_step_analysis(args.image_path, stream=args.stream)
                    for step, result in results.items():
                        print(f"\n{step.upper()}:")
                        print('='*60)
                        if 'error' in result:
                            print(f"Error: {result['error']}")
                        else:
                            content = result.get('content', '')
                            print(content[:500] + "..." if len(content) > 500 else content)
                
            except Exception as e:
                print(f"Processing failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Process all images in the project images dir using predefined cloud models
            main()
            
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set OLLAMA_API_KEY environment variable")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)