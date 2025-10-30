import ollama
import os
import sys
import json
import traceback

def get_model_list():
    """Helper function to extract model list from ollama.list() response"""
    try:
        response = ollama.list()
        
        # ollama.list() returns an object with 'models' attribute containing a list
        # Each model in the list has a 'model' field (the model name)
        if hasattr(response, 'models'):
            return response.models
        elif isinstance(response, dict):
            return response.get('models', [])
        elif isinstance(response, list):
            return response
        else:
            # Last resort: try to extract from __dict__
            if hasattr(response, '__dict__'):
                return response.__dict__.get('models', [])
            return []
        
    except Exception as e:
        print(f"Error getting model list: {e}")
        traceback.print_exc()
        return []

def list_available_models():
    """List all available Ollama models"""
    print("Available Ollama Models:")
    print("=" * 50)
    try:
        model_list = get_model_list()
        
        if model_list:
            for model in model_list:
                # Model name is in 'model' field (ollama Python library standard)
                model_name = None
                if isinstance(model, dict):
                    model_name = model.get('model') or model.get('name')
                else:
                    model_name = getattr(model, 'model', None) or getattr(model, 'name', None)
                
                if model_name:
                    print(f"  - {model_name}")
                    # Handle size - can be dict key or attribute
                    size = model.get('size') if isinstance(model, dict) else getattr(model, 'size', None)
                    if size:
                        size_gb = size / (1024**3)
                        print(f"    Size: {size_gb:.2f} GB")
        else:
            print("No models found. Pull a model first:")
            print("  ollama pull llama3.2-vision")
        return model_list
    except Exception as e:
        print(f"Error listing models: {e}")
        traceback.print_exc()
        return None

def test_model_exists(model_name: str) -> bool:
    """Check if a specific model exists locally"""
    try:
        model_list = get_model_list()
        
        if model_list:
            model_names = []
            for model in model_list:
                if isinstance(model, dict):
                    name = model.get('model') or model.get('name')
                else:
                    name = getattr(model, 'model', None) or getattr(model, 'name', None)
                if name:
                    model_names.append(name)
            return model_name in model_names
        return False
    except Exception as e:
        print(f"Error checking models: {e}")
        traceback.print_exc()
        return False

def search_vision_models():
    """Search for vision models available in Ollama"""
    print("\nVision Models for OCR:")
    print("=" * 50)
    
    vision_models = [
        'llama3.2-vision:latest',
        'gpt-oss:20b',
            ]
    
    available_models = []
    unavailable_models = []
    
    try:
        model_list = get_model_list()
        
        installed_names = []
        for model in model_list:
            if isinstance(model, dict):
                name = model.get('model') or model.get('name')
            else:
                name = getattr(model, 'model', None) or getattr(model, 'name', None)
            if name:
                installed_names.append(name)
        
        for model in vision_models:
            # Check exact match or partial match
            found = False
            for installed in installed_names:
                if model in installed or installed.startswith(model.split(':')[0]):
                    found = True
                    available_models.append(f"[OK] {model} (installed)")
                    break
            
            if not found:
                unavailable_models.append(f"[X] {model} (not installed)")
        
        print("Available/Vision Models:")
        for model in available_models:
            print(f"  {model}")
        
        if unavailable_models:
            print("\nNot Installed (can be pulled):")
            for model in unavailable_models[:5]:  # Show first 5
                print(f"  {model}")
            
            print("\nTo install a model:")
            print("  ollama pull llama3.2-vision")
            print("  ollama pull minicpm-v")
            print("  ollama pull yasserrmd/nanonets-ocr-s")
    
    except Exception as e:
        print(f"Error searching models: {e}")
        traceback.print_exc()

def test_model_connection(model_name: str) -> bool:
    """Test if a model can be accessed"""
    print(f"\nTesting connection to model: {model_name}")
    print("-" * 50)
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': 'Hello, can you respond? This is a test.'
            }]
        )
        print(f"[OK] Model {model_name} is accessible!")
        print(f"Response: {response.message.content[:100]}...")
        return True
    except Exception as e:
        print(f"[X] Error accessing model {model_name}: {e}")
        return False

def get_model_info(model_name: str):
    """Get information about a specific model"""
    print(f"\nModel Information: {model_name}")
    print("=" * 50)
    try:
        model_list = get_model_list()
        
        if model_list:
            for model in model_list:
                if isinstance(model, dict):
                    name = model.get('model') or model.get('name')
                else:
                    name = getattr(model, 'model', None) or getattr(model, 'name', None)
                if name and model_name in name:
                    print(f"Name: {name}")
                    size = model.get('size') if isinstance(model, dict) else getattr(model, 'size', None)
                    if size:
                        size_gb = size / (1024**3)
                        print(f"Size: {size_gb:.2f} GB")
                    modified = model.get('modified_at') if isinstance(model, dict) else getattr(model, 'modified_at', None)
                    if modified:
                        print(f"Modified: {modified}")
                    return model
        print(f"Model {model_name} not found locally")
        return None
    except Exception as e:
        print(f"Error getting model info: {e}")
        traceback.print_exc()
        return None

def main():
    """Main function to test Ollama models"""
    print("Ollama Model Search and Test")
    print("=" * 50)
    
    # List all models
    list_available_models()
    
    # Search for vision models
    search_vision_models()
    
    # Test specific model if provided
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"\n{'='*50}")
        get_model_info(model_name)
        test_model_connection(model_name)
    else:
        # Test default vision model
        default_model = "llama3.2-vision:latest"
        print(f"\n{'='*50}")
        print("Testing default vision model...")
        if test_model_exists(default_model):
            test_model_connection(default_model)
        else:
            print(f"\n[WARNING] {default_model} not found!")
            print("Install it with: ollama pull llama3.2-vision")

if __name__ == "__main__":
    main()


