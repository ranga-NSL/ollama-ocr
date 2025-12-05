import os
import sys
import subprocess
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import argparse
import ollama
from dataclasses import dataclass

# --- Configuration & Setup ---

DEFAULT_MODEL = 'llama3.2-vision:latest'  # Vision model, 6.1 GB - should fit in 8GB VRAM
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Set Windows console encoding for proper output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Performance Tracking ---

@dataclass
class PerformanceMetrics:
    """Track performance metrics for OCR processing"""
    image_validation_time: float = 0.0
    image_encoding_time: float = 0.0
    model_load_time: float = 0.0
    api_call_time: float = 0.0
    response_processing_time: float = 0.0
    total_time: float = 0.0
    gpu_check_before: Optional[bool] = None
    gpu_check_after: Optional[bool] = None
    
    def summary(self) -> str:
        """Return formatted summary of timing breakdown"""
        lines = [
            "Performance Breakdown:",
            f"  Image validation: {self.image_validation_time:.3f}s",
            f"  Image encoding: {self.image_encoding_time:.3f}s",
            f"  Model load/API call: {self.api_call_time:.3f}s",
            f"  Response processing: {self.response_processing_time:.3f}s",
            f"  Total: {self.total_time:.3f}s",
        ]
        if self.gpu_check_before is not None:
            lines.append(f"  GPU before: {'✅' if self.gpu_check_before else '❌'}")
        if self.gpu_check_after is not None:
            lines.append(f"  GPU after: {'✅' if self.gpu_check_after else '❌'}")
        return "\n".join(lines)

# --- Utility Functions ---

def _get_processor_type(line: str) -> str:
    """Determine if a loaded model line suggests CPU or GPU usage."""
    line_upper = line.upper()
    if 'CPU' in line_upper or '100% CPU' in line_upper:
        return 'CPU'
    if '%' in line and 'CPU' not in line_upper:
        return 'GPU'
    return 'UNKNOWN'

def check_gpu_usage_via_nvidia_smi() -> Dict[str, Any]:
    """Check actual GPU usage via nvidia-smi (more accurate than ollama ps)."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 2:
                memory_mb = int(parts[0].strip())
                gpu_util = int(parts[1].strip())
                # If GPU memory > 500MB or utilization > 5%, likely GPU is being used
                if memory_mb > 500 or gpu_util > 5:
                    return {
                        'status': 'gpu',
                        'message': f'GPU is active (Memory: {memory_mb}MB, Utilization: {gpu_util}%)',
                        'using_gpu': True,
                        'memory_mb': memory_mb,
                        'utilization': gpu_util
                    }
                else:
                    return {
                        'status': 'idle',
                        'message': f'GPU appears idle (Memory: {memory_mb}MB, Utilization: {gpu_util}%)',
                        'using_gpu': False
                    }
    except FileNotFoundError:
        pass  # nvidia-smi not available, fall back to ollama ps
    except Exception:
        pass  # Ignore errors, fall back to ollama ps
    return None

def check_gpu_usage(model: str = DEFAULT_MODEL, wait_for_model: bool = False) -> Dict[str, Any]:
    """Check if Ollama is using GPU for the specified model."""
    # First try nvidia-smi for actual GPU usage
    nvidia_result = check_gpu_usage_via_nvidia_smi()
    if nvidia_result:
        return nvidia_result
    
    # Fall back to ollama ps
    try:
        result = subprocess.run(
            ['ollama', 'ps'],
            capture_output=True,
            text=True,
            timeout=5,
            check=True
        )
    except FileNotFoundError:
        return {'status': 'error', 'message': 'ollama not found. Is it installed?', 'using_gpu': False}
    except subprocess.CalledProcessError as e:
        return {'status': 'error', 'message': f'Failed to run ollama ps: {e.stderr.strip()}', 'using_gpu': False}
    except subprocess.TimeoutExpired:
        return {'status': 'error', 'message': 'Timeout checking Ollama status', 'using_gpu': False}
    except Exception as e:
        return {'status': 'error', 'message': f'General error: {str(e)}', 'using_gpu': False}
        
    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        if wait_for_model:
            # Wait a bit and check again
            time.sleep(2)
            return check_gpu_usage(model, wait_for_model=False)
        return {'status': 'no_model', 'message': 'Model not loaded yet (will check during processing)', 'using_gpu': None}
    
    # Check loaded models (skip header line)
    for line in lines[1:]:
        if model in line or any(part.strip() in line for part in model.split(':')):
            processor = _get_processor_type(line)
            if processor == 'GPU':
                return {'status': 'gpu', 'message': f'Model {model} appears to be using GPU', 'using_gpu': True}
            if processor == 'CPU':
                return {'status': 'cpu', 'message': f'Model {model} is using CPU (not GPU)', 'using_gpu': False}

    return {'status': 'unknown', 'message': f'Could not determine GPU usage for {model}', 'using_gpu': None}

def extract_text(
    image_path: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.001,
    seed: Optional[int] = 1,
    num_ctx: Optional[int] = None,
    check_gpu_after: bool = False,
    track_performance: bool = False,
) -> tuple[str, float, Optional[PerformanceMetrics]]:
    """
    Extract text from image using Ollama. 
    Returns (extracted_text, execution_time_seconds, performance_metrics).
    """
    metrics = PerformanceMetrics() if track_performance else None
    total_start = time.time()
    
    # Step 1: Image validation
    print("  [0/3] Validating image file...", flush=True)
    step_start = time.time()
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file '{image_path}' not found")
    if path.suffix.lower() not in VALID_EXTS:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    if metrics:
        metrics.image_validation_time = time.time() - step_start
    print(f"  [0/3] Image validated: {path.name} ({path.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
    
    # Step 2: Check GPU before (baseline)
    if check_gpu_after and metrics:
        step_start = time.time()
        gpu_info_before = check_gpu_usage_via_nvidia_smi()
        if gpu_info_before:
            metrics.gpu_check_before = gpu_info_before.get('using_gpu', False)
    
    # Step 3: Prepare options
    step_start = time.time()
    options = {
        'temperature': temperature,
        'seed': seed,
    }
    if num_ctx is not None:
        options['num_ctx'] = num_ctx
    
    # Step 4: Image encoding (if needed - Ollama handles this, but we can track)
    # Note: Ollama handles image encoding internally, so this is part of API call
    
    # Step 5: API call (this is where most time is spent)
    api_start = time.time()
    print(f"  [1/3] Starting OCR API call to Ollama...", flush=True)
    
    # Check GPU right before API call
    print(f"  [1a] Checking GPU status...", flush=True)
    gpu_before = check_gpu_usage_via_nvidia_smi()
    if gpu_before:
        print(f"  [1a] GPU status: Memory={gpu_before.get('memory_mb', 0)}MB, Util={gpu_before.get('utilization', 0)}%", flush=True)
        if metrics:
            metrics.gpu_check_before = gpu_before.get('using_gpu', False)
    else:
        print(f"  [1a] GPU check unavailable", flush=True)
    
    # Show progress during API call (this can take a while)
    print(f"  [2/3] Processing image with model '{model}' (this may take 30-60 seconds)...", flush=True)
    print(f"        Waiting for Ollama response...", end='', flush=True)
    
    # Start progress and GPU monitoring threads
    import threading
    progress_stop = threading.Event()
    gpu_samples = []
    
    def show_progress():
        dots = 0
        elapsed = 0
        while not progress_stop.is_set():
            time.sleep(2)
            if not progress_stop.is_set():
                dots = (dots + 1) % 4
                elapsed += 2
                # Check GPU every 4 seconds
                if elapsed % 4 == 0:
                    gpu_info = check_gpu_usage_via_nvidia_smi()
                    if gpu_info:
                        mem = gpu_info.get('memory_mb', 0)
                        util = gpu_info.get('utilization', 0)
                        gpu_samples.append((elapsed, mem, util))
                        print(f"\r        [{elapsed}s] GPU: {mem}MB, Util: {util}%{'.' * dots}   ", end='', flush=True)
                    else:
                        print(f"\r        Waiting for Ollama response{'.' * dots}   ", end='', flush=True)
                else:
                    print(f"\r        Waiting for Ollama response{'.' * dots}   ", end='', flush=True)
    
    progress_thread = threading.Thread(target=show_progress, daemon=True)
    progress_thread.start()
    
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are an OCR assistant. Extract all visible text accurately.'},
                {'role': 'user', 'content': 'Extract all text from this image', 'images': [image_path]},
            ],
            options=options,
        )
    finally:
        progress_stop.set()
        progress_thread.join(timeout=0.5)
        print()  # Newline after progress indicator
    
    api_elapsed = time.time() - api_start
    if metrics:
        metrics.api_call_time = api_elapsed
    
    print(f"\n  [2/3] API call completed in {api_elapsed:.2f}s", flush=True)
    
    # Analyze GPU samples collected during processing
    if gpu_samples:
        print(f"  [2b] GPU utilization during processing:", flush=True)
        max_util = max((s[2] for s in gpu_samples), default=0)
        avg_util = sum(s[2] for s in gpu_samples) / len(gpu_samples) if gpu_samples else 0
        max_mem = max((s[1] for s in gpu_samples), default=0)
        print(f"        Peak utilization: {max_util}%", flush=True)
        print(f"        Average utilization: {avg_util:.1f}%", flush=True)
        print(f"        Peak memory: {max_mem}MB", flush=True)
        
        if max_util < 10:
            print(f"  ⚠️  WARNING: Very low GPU utilization ({max_util}%)!", flush=True)
            print(f"        This suggests the model may be using CPU or GPU inefficiently.", flush=True)
            print(f"        Possible causes:", flush=True)
            print(f"        - Model too large, offloading to CPU", flush=True)
            print(f"        - Context window too large ({num_ctx or 'default'})", flush=True)
            print(f"        - Model not fully loaded on GPU", flush=True)
    
    # Check GPU right after API call
    print(f"  [2a] Checking GPU status after processing...", flush=True)
    time.sleep(0.3)  # Brief delay for GPU to update
    gpu_after = check_gpu_usage_via_nvidia_smi()
    if gpu_after:
        mem_after = gpu_after.get('memory_mb', 0)
        util_after = gpu_after.get('utilization', 0)
        print(f"  [2a] GPU status: Memory={mem_after}MB, Util={util_after}%", flush=True)
        if metrics:
            metrics.gpu_check_after = gpu_after.get('using_gpu', False)
        
        # Warn if GPU dropped significantly
        if gpu_before and gpu_before.get('memory_mb', 0) > 500 and mem_after < 200:
            print(f"  ⚠️  WARNING: GPU memory dropped significantly - may have switched to CPU!", flush=True)
    else:
        print(f"  [2a] GPU check unavailable", flush=True)
    
    # Step 6: Response processing
    print(f"  [3/3] Processing response...", flush=True)
    step_start = time.time()
    text = getattr(resp.message, 'content', str(resp))
    if metrics:
        metrics.response_processing_time = time.time() - step_start
    print(f"  [3/3] Response processed ({len(text)} characters extracted)", flush=True)
    
    # Step 7: Check GPU after processing
    if check_gpu_after:
        time.sleep(0.3)  # Brief delay for model to register
        gpu_info_after = check_gpu_usage_via_nvidia_smi()
        if gpu_info_after:
            using_gpu = gpu_info_after.get('using_gpu', False)
            if metrics:
                metrics.gpu_check_after = using_gpu
            if using_gpu:
                mem = gpu_info_after.get('memory_mb', 0)
                util = gpu_info_after.get('utilization', 0)
                print(f"  ✅ GPU in use (Memory: {mem}MB, Util: {util}%)")
            else:
                print(f"  ⚠️  GPU not detected - may be using CPU")
    
    total_elapsed = time.time() - total_start
    if metrics:
        metrics.total_time = total_elapsed
    
    return text, total_elapsed, metrics

def collect_images(images_dir: Path) -> List[str]:
    """Collect paths of valid images in a directory."""
    if not images_dir.exists():
        return []
    return [str(p) for p in sorted(images_dir.glob('*')) if p.is_file() and p.suffix.lower() in VALID_EXTS]


# --- Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(description='Local OCR using Ollama.')
    parser.add_argument('image_path', nargs='?', help='Path to a single image to process. If omitted, runs batch mode.')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Ollama model to use.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature (lower = more deterministic).')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for determinism.')
    parser.add_argument('--num-ctx', type=int, default=None, help='Context window size (smaller = less GPU memory, default: auto).')
    parser.add_argument('--no-check-gpu', action='store_true', help='Disable GPU usage check.')
    parser.add_argument('--show-timing', action='store_true', help='Show detailed timing information.')
    parser.add_argument('--profile', action='store_true', help='Enable detailed performance profiling (tracks each step).')
    args = parser.parse_args()
    
    model = args.model

    if not args.no_check_gpu:
        print("Checking GPU usage...")
        # First check actual GPU via nvidia-smi
        gpu_info = check_gpu_usage(model)
        print(f"Status: {gpu_info['message']}")
        if gpu_info.get('using_gpu') is True:
            print("✅ GPU is being used. Good performance expected.")
            if 'memory_mb' in gpu_info:
                print(f"   GPU Memory: {gpu_info['memory_mb']}MB, Utilization: {gpu_info.get('utilization', 'N/A')}%")
        elif gpu_info.get('using_gpu') is False:
            print("⚠️ WARNING: GPU not detected. May be using CPU.")
        elif gpu_info.get('status') == 'no_model':
            print("ℹ️ Model not loaded yet - will verify during processing.")
        print("-" * 60)

    try:
        if args.image_path:
            # Single image mode
            print(f"**Single Image Mode**")
            print(f"Processing: {args.image_path}")
            print(f"Model: {model}")
            print('=' * 60)

            print("Starting OCR extraction...")
            sys.stdout.flush()
            
            # Always track performance to show intermediate steps
            text, elapsed_time, metrics = extract_text(
                args.image_path, model=model, temperature=args.temperature, 
                seed=args.seed, num_ctx=args.num_ctx, check_gpu_after=not args.no_check_gpu,
                track_performance=True  # Always show intermediate steps
            )
            
            print("\n" + "=" * 60)
            print("EXTRACTED TEXT:")
            print("=" * 60)
            print(text)
            print('\n' + '=' * 60)
            print(f"Execution Time: {elapsed_time:.2f}s")
            
            if metrics and (args.profile or args.show_timing):
                print('\n' + metrics.summary())
            
        else:
            # Batch mode
            images_dir = Path(__file__).resolve().parent / 'images'
            images = collect_images(images_dir)
            if not images:
                print(f"No images found in the '{images_dir}' directory.")
                return

            print(f"**Batch Mode**")
            print(f"Directory: {images_dir}")
            print(f"Model: {model}")
            print('=' * 60)
            
            batch_start = time.time()
            total_images = len(images)
            total_ocr_time = 0.0
            
            for idx, img in enumerate(images, 1):
                img_path = Path(img).name
                print(f"\n[{idx}/{total_images}] Processing: {img_path}")
                
                ocr_start_time = time.time()
                text, elapsed_time, metrics = extract_text(
                    img, model=model, temperature=args.temperature, 
                    seed=args.seed, num_ctx=args.num_ctx, check_gpu_after=False,
                    track_performance=args.profile
                )
                ocr_total_time = time.time() - ocr_start_time
                total_ocr_time += ocr_total_time
                
                # Show extracted text (first 1200 chars) and timing
                print(text if len(text) <= 1200 else text[:1200] + '...')
                print(f"  [OCR Time: {ocr_total_time:.2f}s]")
                
                if metrics and args.profile:
                    print(f"  {metrics.summary().replace(chr(10), chr(10) + '  ')}")
            
            batch_elapsed = time.time() - batch_start
            print('\n' + '=' * 60)
            print("Batch Processing Summary:")
            print(f"  Total images: {total_images}")
            print(f"  Total OCR Time: {total_ocr_time:.2f}s")
            print(f"  Average per image: {total_ocr_time/total_images:.2f}s")
            print(f"  Total Execution Time: {batch_elapsed:.2f}s (including overhead)")
            print('=' * 60)
            
    except Exception as e:
        print(f"\nFATAL Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


