import os
import sys
import base64
from pathlib import Path
from ollama import Client

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Mistral Cloud Model Configuration
MISTRAL_MODEL = "ministral-3:8b"   #'mistral-large-3:675b-cloud'

def get_client():
    """Create and return Ollama Cloud client"""
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        raise ValueError("OLLAMA_API_KEY environment variable not set")
    
    return Client(
        host='https://ollama.com',
        headers={'Authorization': f'Bearer {api_key}'}
    )

def test_text_reasoning(client):
    """Test Mistral's text reasoning capabilities"""
    print(f"\n{'='*60}")
    print("TEST 1: Text Reasoning")
    print(f"{'='*60}")
    
    reasoning_questions = [
        {
            'prompt': 'A train leaves Station A at 9:00 AM traveling 60 mph. Another train leaves Station B at 10:00 AM traveling 80 mph. If the stations are 200 miles apart, when will they meet?',
            'description': 'Mathematical reasoning problem'
        },
        {
            'prompt': 'Explain the concept of recursion in programming with a simple example.',
            'description': 'Technical explanation'
        },
        {
            'prompt': 'If all roses are flowers, and some flowers are red, can we conclude that some roses are red? Explain your reasoning.',
            'description': 'Logical reasoning'
        }
    ]
    
    for i, test in enumerate(reasoning_questions, 1):
        print(f"\n--- Question {i}: {test['description']} ---")
        print(f"Prompt: {test['prompt']}")
        print("\nResponse:")
        print('-' * 60)
        
        try:
            response = client.chat(
                model=MISTRAL_MODEL,
                messages=[
                    {'role': 'user', 'content': test['prompt']}
                ],
                stream=True
            )
            
            for chunk in response:
                print(chunk['message']['content'], end='', flush=True)
            print()  # Final newline
            print("✓ Completed successfully")
            
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
            break
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for Cloud API"""
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return base64_image

def test_vision_ocr(client, images_dir='images'):
    """Test Mistral's vision/OCR capabilities with images from the project"""
    print(f"\n{'='*60}")
    print("TEST 2: Vision/OCR Capabilities")
    print(f"{'='*60}")
    
    # Find images directory
    base_dir = Path(__file__).resolve().parent
    images_path = base_dir / images_dir
    
    if not images_path.exists():
        print(f"⚠ Images directory '{images_dir}' not found. Skipping vision tests.")
        return
    
    # Collect image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in sorted(images_path.iterdir()) 
                   if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"⚠ No image files found in '{images_dir}' directory. Skipping vision tests.")
        return
    
    print(f"\nFound {len(image_files)} image(s) to test:")
    for img in image_files:
        print(f"  - {img.name}")
    
    vision_tests = [
        {
            'prompt': 'Extract all visible text from this image. Be accurate and preserve formatting.',
            'description': 'Text Extraction (OCR)'
        },
        {
            'prompt': 'Describe what you see in this image in detail, including any text, objects, and layout.',
            'description': 'Image Description'
        },
        {
            'prompt': 'Analyze this image and identify the document type. Extract key information and structure.',
            'description': 'Document Analysis'
        }
    ]
    
    for image_file in image_files:
        print(f"\n{'='*60}")
        print(f"Processing: {image_file.name}")
        print(f"{'='*60}")
        
        try:
            # Encode image
            base64_image = encode_image_to_base64(str(image_file))
            
            # Run vision tests
            for i, test in enumerate(vision_tests, 1):
                print(f"\n--- Test {i}: {test['description']} ---")
                print('-' * 60)
                
                try:
                    response = client.chat(
                        model=MISTRAL_MODEL,
                        messages=[
                            {
                                'role': 'user',
                                'content': test['prompt'],
                                'images': [base64_image]
                            }
                        ],
                        stream=True
                    )
                    
                    for chunk in response:
                        print(chunk['message']['content'], end='', flush=True)
                    print()  # Final newline
                    print("✓ Completed successfully")
                    
                except KeyboardInterrupt:
                    print("\n⚠ Interrupted by user")
                    break
                except Exception as e:
                    print(f"✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"✗ Failed to process {image_file.name}: {e}")

def main():
    """Main test function"""
    print(f"{'='*60}")
    print("Mistral Cloud Model Test")
    print(f"Model: {MISTRAL_MODEL}")
    print(f"{'='*60}")
    
    # Check API key
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        print("✗ ERROR: OLLAMA_API_KEY environment variable not set")
        print("\nPlease set it using:")
        print("  PowerShell: $env:OLLAMA_API_KEY = \"<your-key>\"")
        print("  CMD: set OLLAMA_API_KEY=<your-key>")
        sys.exit(1)
    
    print(f"✓ API Key present (length: {len(api_key)})")
    
    try:
        client = get_client()
        
        # Test 1: Text Reasoning
        test_text_reasoning(client)
        
        # Test 2: Vision/OCR
        test_vision_ocr(client)
        
        print(f"\n{'='*60}")
        print("All tests completed!")
        print(f"{'='*60}")
        
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test Mistral Cloud Model (mistral-large-3:675b-cloud) with reasoning and vision capabilities',
        epilog='Requires OLLAMA_API_KEY environment variable'
    )
    parser.add_argument('--images-dir', default='images',
                       help='Directory containing test images (default: images)')
    parser.add_argument('--reasoning-only', action='store_true',
                       help='Run only text reasoning tests')
    parser.add_argument('--vision-only', action='store_true',
                       help='Run only vision/OCR tests')
    
    args = parser.parse_args()
    
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        print("✗ ERROR: OLLAMA_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        client = get_client()
        
        if args.reasoning_only:
            test_text_reasoning(client)
        elif args.vision_only:
            test_vision_ocr(client, args.images_dir)
        else:
            main()
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
