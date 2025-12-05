import os
import sys
import base64
import io
import time
from dataclasses import dataclass
from ollama import Client
from typing import Dict, Optional, List, Tuple
from pathlib import Path

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Recommended vision/OCR-capable cloud models
CLOUD_VISION_MODELS = [
    'mistral-large-3:675b-cloud',  # Vision and reasoning model (OCR capable)
    'qwen3-vl:235b-cloud',  # Vision-language model (OCR capable)
    'minimax-m2:cloud',     # Cloud LLM (may be text-only)
    'gpt-oss:120b-cloud'    # Cloud LLM (may be text-only)
]

# Performance tracking dataclass
@dataclass
class PerformanceMetrics:
    """Track performance metrics for OCR processing"""
    pdf_conversion_time: float = 0.0
    image_encoding_time: float = 0.0
    api_call_time: float = 0.0
    response_processing_time: float = 0.0
    total_time: float = 0.0
    file_type: str = 'image'  # 'image' or 'pdf'
    page_number: Optional[int] = None  # For PDFs
    
    def summary(self) -> str:
        """Return formatted summary of timing breakdown"""
        lines = ["Performance Breakdown:"]
        
        if self.file_type == 'pdf' and self.pdf_conversion_time > 0:
            lines.append(f"  PDF conversion: {self.pdf_conversion_time:.3f}s")
        elif self.image_encoding_time > 0:
            lines.append(f"  Image encoding: {self.image_encoding_time:.3f}s")
        
        lines.extend([
            f"  API call: {self.api_call_time:.3f}s",
            f"  Response processing: {self.response_processing_time:.3f}s",
            f"  Total: {self.total_time:.3f}s",
        ])
        
        if self.page_number:
            lines.append(f"  Page: {self.page_number}")
        
        return "\n".join(lines)

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
        model_lower = self.model.lower()
        if '-vl' not in model_lower and 'vision' not in model_lower and 'mistral' not in model_lower:
            print(f"[WARNING] Model '{self.model}' may not support vision/OCR capabilities.")
            print(f"         Recommended vision models: {', '.join(CLOUD_VISION_MODELS[:2])}")
        
        self.valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.pdf']
        
        # Create client for Ollama Cloud
        self.client = Client(
            host='https://ollama.com',
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
    
    def _validate_file(self, file_path: str) -> None:
        """Validate file exists and has supported format (image or PDF)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found!")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.valid_extensions:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _validate_image(self, image_path: str) -> None:
        """Validate image file exists and has supported format (backward compatibility)"""
        self._validate_file(image_path)
    
    def _is_pdf(self, file_path: str) -> bool:
        """Check if file is a PDF"""
        return os.path.splitext(file_path)[1].lower() == '.pdf'
    
    def _get_pdf_page_count(self, pdf_path: str) -> int:
        """Get total number of pages in PDF"""
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image is not installed. Install it with: pip install pdf2image")
        
        try:
            # Convert all pages to get count (simple approach)
            images = convert_from_path(pdf_path)
            return len(images) if images else 0
        except Exception as e:
            raise RuntimeError(f"Failed to get PDF page count: {str(e)}")
    
    def _convert_pdf_page_to_base64(self, pdf_path: str, page_number: int = 1) -> Tuple[str, float]:
        """
        Convert PDF page to base64 string with timing
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number to convert (1-indexed)
            
        Returns:
            Tuple of (base64_string, conversion_time)
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image is not installed. Install it with: pip install pdf2image")
        
        start_time = time.time()
        
        try:
            # Convert PDF page to image
            images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
            
            if not images:
                raise ValueError(f"Could not read page {page_number} from PDF")
            
            img = images[0]
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # Convert bytes to base64
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            
            conversion_time = time.time() - start_time
            
            return base64_image, conversion_time
            
        except Exception as e:
            raise RuntimeError(f"PDF conversion failed: {str(e)}")
    
    def _encode_image_to_base64(self, image_path: str) -> Tuple[str, float]:
        """
        Encode image to base64 string for Cloud API with timing
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (base64_string, encoding_time)
        """
        start_time = time.time()
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        encoding_time = time.time() - start_time
        return base64_image, encoding_time
    
    def _encode_file_to_base64(self, file_path: str, page_number: int = 1) -> Tuple[str, float, Dict]:
        """
        Encode file (image or PDF) to base64 string with timing
        
        Args:
            file_path: Path to image or PDF file
            page_number: Page number for PDFs (1-indexed, ignored for images)
            
        Returns:
            Tuple of (base64_string, encoding_time, metadata_dict)
            metadata_dict contains: file_type, page_number (if PDF)
        """
        metadata = {}
        
        if self._is_pdf(file_path):
            base64_str, encoding_time = self._convert_pdf_page_to_base64(file_path, page_number)
            metadata['file_type'] = 'pdf'
            metadata['page_number'] = page_number
        else:
            base64_str, encoding_time = self._encode_image_to_base64(file_path)
            metadata['file_type'] = 'image'
        
        return base64_str, encoding_time, metadata
    
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
        Collect image and PDF file paths from the project's images directory
        """
        base_dir = Path(__file__).resolve().parent
        dir_path = Path(images_dir) if images_dir else base_dir / 'images'
        if not dir_path.exists():
            return []
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.pdf'}
        return [str(p) for p in sorted(dir_path.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    
    def extract_text(self, file_path: str, output_format: str = 'text', stream: bool = False, 
                     page_number: int = 1, track_timing: bool = False) -> Dict:
        """
        Extract text from image or PDF with optional JSON output
        
        Args:
            file_path: Path to image or PDF file
            output_format: 'text' or 'json'
            stream: Whether to stream the response
            page_number: Page number for PDFs (1-indexed, ignored for images)
            track_timing: Whether to track and return timing metrics
            
        Returns:
            Dictionary with response data and optional timing metrics
        """
        self._validate_file(file_path)
        
        metrics = PerformanceMetrics() if track_timing else None
        total_start = time.time() if track_timing else None
        
        if output_format == 'json':
            system_prompt = 'You are an OCR system. Extract all text and return as JSON with "text" field.'
            user_prompt = 'Extract all visible text from this image. Return as JSON: {"text": "extracted_text"}'
        else:
            system_prompt = 'You are a specialized OCR assistant. Extract all visible text accurately.'
            user_prompt = 'Extract all text from this image'
        
        try:
            # Encode file to base64 with timing
            base64_image, encoding_time, metadata = self._encode_file_to_base64(file_path, page_number)
            
            if metrics:
                metrics.file_type = metadata['file_type']
                metrics.page_number = metadata.get('page_number')
                if metadata['file_type'] == 'pdf':
                    metrics.pdf_conversion_time = encoding_time
                    metrics.image_encoding_time = 0.0  # PDF conversion includes encoding
                else:
                    metrics.pdf_conversion_time = 0.0
                    metrics.image_encoding_time = encoding_time
            
            # API call
            api_start = time.time() if track_timing else None
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt, 'images': [base64_image]}
                ],
                stream=stream
            )
            
            if metrics and api_start:
                metrics.api_call_time = time.time() - api_start
            
            # Process response
            resp_start = time.time() if track_timing else None
            if stream:
                content = self._get_response_content(response, stream=True)
            else:
                content = self._get_response_content(response, stream=False)
            
            if metrics and resp_start:
                metrics.response_processing_time = time.time() - resp_start
                metrics.total_time = time.time() - total_start
            
            result = {'content': content, 'model': self.model}
            if metrics:
                result['metrics'] = metrics
                result['timing'] = metrics.summary()
            
            return result
                
        except Exception as e:
            raise RuntimeError(f"OCR processing failed: {str(e)}")
    
    def transcribe_handwriting(self, file_path: str, stream: bool = False, 
                               page_number: int = 1, track_timing: bool = False) -> Dict:
        """
        Specialized handwriting transcription
        
        Args:
            file_path: Path to image or PDF file
            stream: Whether to stream the response
            page_number: Page number for PDFs (1-indexed, ignored for images)
            track_timing: Whether to track and return timing metrics
            
        Returns:
            Dictionary with transcribed text and optional timing metrics
        """
        self._validate_file(file_path)
        
        metrics = PerformanceMetrics() if track_timing else None
        total_start = time.time() if track_timing else None
        
        try:
            # Encode file to base64 with timing
            base64_image, encoding_time, metadata = self._encode_file_to_base64(file_path, page_number)
            
            if metrics:
                metrics.file_type = metadata['file_type']
                metrics.page_number = metadata.get('page_number')
                if metadata['file_type'] == 'pdf':
                    metrics.pdf_conversion_time = encoding_time
                    metrics.image_encoding_time = 0.0  # PDF conversion includes encoding
                else:
                    metrics.pdf_conversion_time = 0.0
                    metrics.image_encoding_time = encoding_time
            
            # API call
            api_start = time.time() if track_timing else None
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Transcribe all handwritten text in this image. Maintain original formatting and structure.',
                    'images': [base64_image]
                }],
                stream=stream
            )
            
            if metrics and api_start:
                metrics.api_call_time = time.time() - api_start
            
            # Process response
            resp_start = time.time() if track_timing else None
            if stream:
                content = self._get_response_content(response, stream=True)
                print(content, end='', flush=True)
                print()  # Newline after streaming
            else:
                content = self._get_response_content(response, stream=False)
            
            if metrics and resp_start:
                metrics.response_processing_time = time.time() - resp_start
                metrics.total_time = time.time() - total_start
            
            result = {'content': content, 'model': self.model}
            if metrics:
                result['metrics'] = metrics
                result['timing'] = metrics.summary()
            
            return result
                
        except Exception as e:
            raise RuntimeError(f"Handwriting transcription failed: {str(e)}")
    
    def extract_structured_data(self, file_path: str, data_type: str = 'receipt', stream: bool = False,
                               page_number: int = 1, track_timing: bool = False) -> Dict:
        """
        Extract structured data from images or PDFs (receipts, forms, etc.)
        
        Args:
            file_path: Path to image or PDF file
            data_type: Type of structured data ('receipt', 'form', 'price_tag')
            stream: Whether to stream the response
            page_number: Page number for PDFs (1-indexed, ignored for images)
            track_timing: Whether to track and return timing metrics
            
        Returns:
            Dictionary with extracted structured data and optional timing metrics
        """
        self._validate_file(file_path)
        
        metrics = PerformanceMetrics() if track_timing else None
        total_start = time.time() if track_timing else None
        
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
            # Encode file to base64 with timing
            base64_image, encoding_time, metadata = self._encode_file_to_base64(file_path, page_number)
            
            if metrics:
                metrics.file_type = metadata['file_type']
                metrics.page_number = metadata.get('page_number')
                if metadata['file_type'] == 'pdf':
                    metrics.pdf_conversion_time = encoding_time
                    metrics.image_encoding_time = 0.0  # PDF conversion includes encoding
                else:
                    metrics.pdf_conversion_time = 0.0
                    metrics.image_encoding_time = encoding_time
            
            # API call
            api_start = time.time() if track_timing else None
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': prompts[data_type]['system']},
                    {'role': 'user', 'content': prompts[data_type]['user'], 'images': [base64_image]}
                ],
                stream=stream
            )
            
            if metrics and api_start:
                metrics.api_call_time = time.time() - api_start
            
            # Process response
            resp_start = time.time() if track_timing else None
            if stream:
                content = self._get_response_content(response, stream=True)
                print(content, end='', flush=True)
                print()  # Newline after streaming
            else:
                content = self._get_response_content(response, stream=False)
            
            if metrics and resp_start:
                metrics.response_processing_time = time.time() - resp_start
                metrics.total_time = time.time() - total_start
            
            result = {'content': content, 'model': self.model}
            if metrics:
                result['metrics'] = metrics
                result['timing'] = metrics.summary()
            
            return result
                
        except Exception as e:
            raise RuntimeError(f"Structured data extraction failed: {str(e)}")
    
    def analyze_document(self, file_path: str, stream: bool = False, 
                        page_number: int = 1, track_timing: bool = False) -> Dict:
        """
        Comprehensive document analysis
        
        Args:
            file_path: Path to image or PDF file
            stream: Whether to stream the response
            page_number: Page number for PDFs (1-indexed, ignored for images)
            track_timing: Whether to track and return timing metrics
            
        Returns:
            Dictionary with document analysis and optional timing metrics
        """
        self._validate_file(file_path)
        
        metrics = PerformanceMetrics() if track_timing else None
        total_start = time.time() if track_timing else None
        
        try:
            # Encode file to base64 with timing
            base64_image, encoding_time, metadata = self._encode_file_to_base64(file_path, page_number)
            
            if metrics:
                metrics.file_type = metadata['file_type']
                metrics.page_number = metadata.get('page_number')
                if metadata['file_type'] == 'pdf':
                    metrics.pdf_conversion_time = encoding_time
                    metrics.image_encoding_time = 0.0  # PDF conversion includes encoding
                else:
                    metrics.pdf_conversion_time = 0.0
                    metrics.image_encoding_time = encoding_time
            
            # API call
            api_start = time.time() if track_timing else None
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': 'Analyze this document comprehensively. Include: document type, key information, text content, layout structure, and any notable features.',
                    'images': [base64_image]
                }],
                stream=stream
            )
            
            if metrics and api_start:
                metrics.api_call_time = time.time() - api_start
            
            # Process response
            resp_start = time.time() if track_timing else None
            if stream:
                content = self._get_response_content(response, stream=True)
                print(content, end='', flush=True)
                print()  # Newline after streaming
            else:
                content = self._get_response_content(response, stream=False)
            
            if metrics and resp_start:
                metrics.response_processing_time = time.time() - resp_start
                metrics.total_time = time.time() - total_start
            
            result = {'content': content, 'model': self.model}
            if metrics:
                result['metrics'] = metrics
                result['timing'] = metrics.summary()
            
            return result
                
        except Exception as e:
            raise RuntimeError(f"Document analysis failed: {str(e)}")
    
    def multi_step_analysis(self, file_path: str, stream: bool = False, 
                           page_number: int = 1, track_timing: bool = False) -> Dict[str, Dict]:
        """
        Perform multiple analysis steps on the same image or PDF page
        
        Args:
            file_path: Path to image or PDF file
            stream: Whether to stream the responses
            page_number: Page number for PDFs (1-indexed, ignored for images)
            track_timing: Whether to track and return timing metrics
            
        Returns:
            Dictionary with results from all analysis steps
        """
        self._validate_file(file_path)
        
        # Encode file once for all steps
        base64_image, encoding_time, metadata = self._encode_file_to_base64(file_path, page_number)
        
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
            results['text_extraction'] = self.extract_text(file_path, stream=stream, 
                                                         page_number=page_number, track_timing=track_timing)
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

def main(model: Optional[str] = None, show_timing: bool = False):
    """Run OCR over project images and PDFs using the configured cloud model"""
    try:
        analyzer = CloudOCRAnalyzer(model=model or CLOUD_VISION_MODELS[0])
        files = analyzer._collect_project_images()
        
        if not files:
            print("No images or PDFs found in the project's 'images' directory.")
            return

        print(f"\n{'='*60}")
        print(f"Model: {analyzer.model}")
        print(f"{'='*60}")

        for file_path in files:
            file_name = Path(file_path).name
            is_pdf = analyzer._is_pdf(file_path)
            
            print(f"\n{'-'*60}")
            print(f"Analyzing: {file_name}")
            if is_pdf:
                try:
                    page_count = analyzer._get_pdf_page_count(file_path)
                    print(f"PDF with {page_count} page(s)")
                except Exception as e:
                    print(f"Warning: Could not determine page count: {e}")
                    page_count = 1
            print(f"{'-'*60}")
            
            try:
                # Determine pages to process
                if is_pdf:
                    pages_to_process = range(1, page_count + 1)
                else:
                    pages_to_process = [1]  # Dummy page number for images
                
                for page_num in pages_to_process:
                    if is_pdf:
                        print(f"\n[Page {page_num}/{page_count}]")
                    
                    # Text extraction
                    print(f"1) Text Extraction [{analyzer.model}]:")
                    text_result = analyzer.extract_text(file_path, stream=False, 
                                                       page_number=page_num, track_timing=show_timing)
                    content = text_result.get('content', '')
                    print(content[:400] + "..." if len(content) > 400 else content)
                    if show_timing and 'timing' in text_result:
                        print(f"\n{text_result['timing']}")

                    # Document analysis
                    print(f"\n2) Document Analysis [{analyzer.model}]:")
                    doc_result = analyzer.analyze_document(file_path, stream=False,
                                                          page_number=page_num, track_timing=show_timing)
                    content = doc_result.get('content', '')
                    print(content[:400] + "..." if len(content) > 400 else content)
                    if show_timing and 'timing' in doc_result:
                        print(f"\n{doc_result['timing']}")

                    # Structured data heuristic: if 'receipt' in filename
                    if 'receipt' in file_name.lower():
                        print(f"\n3) Structured Data (receipt) [{analyzer.model}]:")
                        try:
                            struct_result = analyzer.extract_structured_data(file_path, 'receipt', stream=False,
                                                                            page_number=page_num, track_timing=show_timing)
                            print(struct_result.get('content', ''))
                            if show_timing and 'timing' in struct_result:
                                print(f"\n{struct_result['timing']}")
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
        description='Cloud-based OCR using Ollama Cloud API (supports images and PDFs)',
        epilog=f"Models: {', '.join(CLOUD_VISION_MODELS)}"
    )
    parser.add_argument('file_path', nargs='?', help='Path to image or PDF file to process (optional: if omitted, processes all files in images directory)')
    parser.add_argument('--model', default=None, 
                       help=f'Cloud model to use for OCR; if omitted, uses default: {CLOUD_VISION_MODELS[0]}')
    parser.add_argument('--page', type=int, default=None,
                       help='Page number for PDF files (1-indexed). If omitted, processes all pages')
    parser.add_argument('--stream', action='store_true', help='Stream the response')
    parser.add_argument('--mode', choices=['text', 'handwriting', 'structured', 'document', 'multi'],
                       default='text', help='OCR mode to use')
    parser.add_argument('--data-type', choices=['receipt', 'form', 'price_tag'],
                       default='receipt', help='Data type for structured extraction')
    parser.add_argument('--show-timing', action='store_true', help='Display performance timing metrics')
    parser.add_argument('--outdir', default='outputs', help='Directory to write output files (batch mode only)')
    parser.add_argument('--csv', action='store_true', help='Generate CSV summary file (batch mode only)')
    
    args = parser.parse_args()
    
    try:
        # Use provided model or default
        selected_model = args.model or CLOUD_VISION_MODELS[0]
        analyzer = CloudOCRAnalyzer(model=selected_model)
        
        if args.file_path:
            # Process single file (image or PDF)
            if not os.path.exists(args.file_path):
                print(f"Error: File '{args.file_path}' not found!")
                sys.exit(1)
            
            file_name = Path(args.file_path).name
            is_pdf = analyzer._is_pdf(args.file_path)
            
            print(f"Processing: {file_name}")
            print(f"Model: {selected_model}")
            if is_pdf:
                try:
                    page_count = analyzer._get_pdf_page_count(args.file_path)
                    print(f"PDF with {page_count} page(s)")
                    if args.page:
                        if args.page < 1 or args.page > page_count:
                            print(f"Error: Page {args.page} is out of range (1-{page_count})")
                            sys.exit(1)
                        pages_to_process = [args.page]
                    else:
                        pages_to_process = range(1, page_count + 1)
                except Exception as e:
                    print(f"Warning: Could not determine page count: {e}")
                    pages_to_process = [1]
            else:
                pages_to_process = [1]  # Dummy page number for images
            
            print(f"{'='*60}\n")
            
            try:
                for page_num in pages_to_process:
                    if is_pdf and len(pages_to_process) > 1:
                        print(f"\n[Page {page_num}/{page_count}]")
                    elif is_pdf:
                        print(f"[Page {page_num}]")
                    
                    if args.mode == 'text':
                        result = analyzer.extract_text(args.file_path, stream=args.stream,
                                                      page_number=page_num, track_timing=args.show_timing)
                        if not args.stream:
                            print(result.get('content', ''))
                        if args.show_timing and 'timing' in result:
                            print(f"\n{result['timing']}")
                            
                    elif args.mode == 'handwriting':
                        result = analyzer.transcribe_handwriting(args.file_path, stream=args.stream,
                                                                page_number=page_num, track_timing=args.show_timing)
                        if not args.stream:
                            print(result.get('content', ''))
                        if args.show_timing and 'timing' in result:
                            print(f"\n{result['timing']}")
                            
                    elif args.mode == 'structured':
                        result = analyzer.extract_structured_data(args.file_path, args.data_type, stream=args.stream,
                                                                  page_number=page_num, track_timing=args.show_timing)
                        if not args.stream:
                            print(result.get('content', ''))
                        if args.show_timing and 'timing' in result:
                            print(f"\n{result['timing']}")
                            
                    elif args.mode == 'document':
                        result = analyzer.analyze_document(args.file_path, stream=args.stream,
                                                         page_number=page_num, track_timing=args.show_timing)
                        if not args.stream:
                            print(result.get('content', ''))
                        if args.show_timing and 'timing' in result:
                            print(f"\n{result['timing']}")
                            
                    elif args.mode == 'multi':
                        results = analyzer.multi_step_analysis(args.file_path, stream=args.stream,
                                                              page_number=page_num, track_timing=args.show_timing)
                        for step, result in results.items():
                            print(f"\n{step.upper()}:")
                            print('='*60)
                            if 'error' in result:
                                print(f"Error: {result['error']}")
                            else:
                                content = result.get('content', '')
                                print(content[:500] + "..." if len(content) > 500 else content)
                                if args.show_timing and 'timing' in result:
                                    print(f"\n{result['timing']}")
                
            except Exception as e:
                print(f"Processing failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Process all files in the project images dir using selected model
            # Create output directory
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            
            analyzer = CloudOCRAnalyzer(model=selected_model)
            files = analyzer._collect_project_images()
            
            if not files:
                print("No images or PDFs found in the project's 'images' directory.")
                sys.exit(0)
            
            # CSV summary file (if requested) - open after confirming files exist
            csv_file = None
            csv_f = None
            csv_writer = None
            if args.csv:
                import csv
                csv_file = outdir / 'summary.csv'
                csv_f = csv_file.open('w', newline='', encoding='utf-8')
                csv_writer = csv.writer(csv_f)
                csv_writer.writerow(['file', 'mode', 'model', 'chars', 'pages', 'type'])
            
            print(f"\n{'='*60}")
            print(f"Model: {selected_model}")
            print(f"Output directory: {outdir}")
            print(f"{'='*60}\n")
            
            for file_path in files:
                file_name = Path(file_path).name
                file_stem = Path(file_path).stem
                is_pdf = analyzer._is_pdf(file_path)
                
                try:
                    if is_pdf:
                        try:
                            page_count = analyzer._get_pdf_page_count(file_path)
                        except Exception as e:
                            print(f"Warning: Could not determine page count for {file_name}: {e}")
                            page_count = 1
                        pages_to_process = range(1, page_count + 1) if args.page is None else [args.page]
                    else:
                        pages_to_process = [1]
                        page_count = 1
                    
                    # Process each page
                    all_page_texts = []
                    for page_num in pages_to_process:
                        try:
                            if args.mode == 'text':
                                result = analyzer.extract_text(file_path, stream=False,
                                                              page_number=page_num, track_timing=args.show_timing)
                            elif args.mode == 'handwriting':
                                result = analyzer.transcribe_handwriting(file_path, stream=False,
                                                                        page_number=page_num, track_timing=args.show_timing)
                            elif args.mode == 'structured':
                                result = analyzer.extract_structured_data(file_path, args.data_type, stream=False,
                                                                          page_number=page_num, track_timing=args.show_timing)
                            elif args.mode == 'document':
                                result = analyzer.analyze_document(file_path, stream=False,
                                                                   page_number=page_num, track_timing=args.show_timing)
                            elif args.mode == 'multi':
                                # For multi mode, combine all results
                                results = analyzer.multi_step_analysis(file_path, stream=False,
                                                                       page_number=page_num, track_timing=args.show_timing)
                                combined = []
                                for step, step_result in results.items():
                                    if 'error' not in step_result:
                                        combined.append(f"{step.upper()}:\n{step_result.get('content', '')}")
                                result = {'content': '\n\n'.join(combined)}
                            else:
                                result = analyzer.extract_text(file_path, stream=False,
                                                              page_number=page_num, track_timing=args.show_timing)
                            
                            content = result.get('content', '')
                            
                            if is_pdf and page_count > 1:
                                all_page_texts.append(f"\n\n--- Page {page_num} ---\n\n{content}")
                            else:
                                all_page_texts.append(content)
                            
                        except Exception as e:
                            error_msg = f"[Error processing page {page_num}: {e}]"
                            if is_pdf and page_count > 1:
                                all_page_texts.append(f"\n\n--- Page {page_num} ---\n\n{error_msg}")
                            else:
                                all_page_texts.append(error_msg)
                    
                    # Combine all pages
                    combined_text = ''.join(all_page_texts).strip()
                    
                    # Write output file
                    out_file = outdir / f"LLM_{file_stem}.txt"
                    
                    out_file.write_text(combined_text, encoding='utf-8')
                    
                    # Write to CSV if requested
                    if csv_writer:
                        file_type = "PDF" if is_pdf else "Image"
                        csv_writer.writerow([file_path, args.mode, selected_model, len(combined_text), page_count, file_type])
                    
                    file_type = "PDF" if is_pdf else "Image"
                    print(f"[{args.mode}] {file_name} -> {out_file.name} ({file_type}, {page_count} page{'s' if page_count > 1 else ''})")
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    if csv_writer:
                        csv_writer.writerow([file_path, args.mode, selected_model, 0, 0, 'error'])
            
            # Close CSV file if opened
            if csv_file and csv_f:
                csv_f.close()
                print(f"\nCSV summary saved to: {csv_file}")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set OLLAMA_API_KEY environment variable")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)