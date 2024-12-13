# https://colab.research.google.com/drive/1IvWXy79IrXxcP4NZb9Er8xOPx46S61M7

"""
Moondream Model Performance Benchmarking Script
--------------------------------------------
Simple benchmarking script for Moondream models
"""

from google.colab import drive
drive.mount('/content/drive')

import moondream as md
from PIL import Image
import requests
from io import BytesIO
from time import perf_counter
import psutil
import os
import gc
import torch
from datetime import datetime
import json

def get_memory_usage():
    """Return current memory usage in MB for both CPU and GPU"""
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    return {'cpu': cpu_mem, 'gpu': gpu_mem}

def setup_model(model_name):
    """Downloads and sets up the specified model if needed"""
    models = {
        '0.5B INT8': {
            'path': '/content/drive/MyDrive/md-models/moondream-0_5b-int8.mf.gz',
            'url': 'https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz'
        },
        '0.5B INT4': {
            'path': '/content/drive/MyDrive/md-models/moondream-0_5b-int4.mf.gz',
            'url': 'https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int4.mf.gz'
        },
        '2B INT8': {
            'path': '/content/drive/MyDrive/md-models/moondream-2b-int8.mf.gz',
            'url': 'https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz'
        },
        '2B INT4': {
            'path': '/content/drive/MyDrive/md-models/moondream-2b-int4.mf.gz',
            'url': 'https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int4.mf.gz'
        }
    }
    
    model_path = models[model_name]['path']
    model_url = models[model_name]['url']
    
    os.makedirs('/content/drive/MyDrive/md-models', exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"\n📥 Downloading {model_name}...")
        os.system(f"wget -O {model_path} {model_url}")
        print("✅ Done!")
    else:
        print(f"\n✅ Model exists in Drive")
    
    return model_path

def save_results(model_name, all_runs, image_load_time):
    """Save benchmark results to a timestamped JSON file"""
    benchmark_dir = '/content/drive/MyDrive/md-benchmarks'
    os.makedirs(benchmark_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'benchmark_results_{timestamp}.json'
    filepath = os.path.join(benchmark_dir, filename)
    
    # Prepare system info
    system_info = {
        'platform': 'Google Colab',
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024 * 1024 * 1024),
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    # Prepare results
    save_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'system_info': system_info
        },
        'results': {
            model_name: {
                'runs': all_runs,
                'image_load_time': image_load_time
            }
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {filepath}")
    return filepath

def run_benchmark(model_name, num_runs=3, image_url=None):
    """Run benchmark for a single model multiple times"""
    if image_url is None:
        image_url = "https://cdn.pixabay.com/photo/2023/01/30/11/04/cat-7755394_1280.jpg"
    
    print(f"\n🚀 Benchmarking {model_name}...")
    all_runs = []
    model_load_times = []
    
    # Setup model path
    model_path = setup_model(model_name)
    
    # Download image once
    print("\n📥 Loading image...")
    image_start = perf_counter()
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_end = perf_counter()
    image_load_time = image_end - image_start
    
    try:
        # Load model once
        model_start = perf_counter()
        model = md.vl(model=model_path)
        model_end = perf_counter()
        initial_load_time = model_end - model_start
        
        for run in range(num_runs):
            print(f"\n📊 Run {run + 1}/{num_runs}")
            try:
                results = {}
                
                # Track initial memory
                initial_memory = get_memory_usage()
                results['initial_memory'] = initial_memory
                
                # Record model load time from first load
                results['model_load_time'] = initial_load_time
                model_load_times.append(initial_load_time)
                
                # Memory after model load
                post_model_memory = get_memory_usage()
                results['post_model_memory'] = post_model_memory
                
                # Encoding benchmarks
                print("\n🖼️ Running encoding tests...")
                
                # Cold encoding
                encode_start = perf_counter()
                encoded_image = model.encode_image(image)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                encode_end = perf_counter()
                results['cold_encode_time'] = encode_end - encode_start
                
                # Track memory after encoding
                post_encode_memory = get_memory_usage()
                results['post_encode_memory'] = post_encode_memory
                
                # Warm encodings
                warm_times = []
                for _ in range(3):
                    encode_start = perf_counter()
                    encoded_image = model.encode_image(image)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    encode_end = perf_counter()
                    warm_times.append(encode_end - encode_start)
                
                results['avg_warm_encode_time'] = sum(warm_times) / len(warm_times)
                
                # Generation tests
                print("\n🎯 Testing generation...")
                
                # TTFT
                ttft_start = perf_counter()
                first_token = next(iter(model.caption(encoded_image, stream=True)["caption"]))
                ttft_end = perf_counter()
                results['ttft'] = ttft_end - ttft_start
                results['first_token'] = first_token
                
                # Full caption
                caption_start = perf_counter()
                tokens = []
                for token in model.caption(encoded_image, stream=True)["caption"]:
                    tokens.append(token)
                caption_end = perf_counter()
                
                results['caption'] = "".join(tokens)
                results['caption_time'] = caption_end - caption_start
                results['num_tokens'] = len(tokens)
                results['tokens_per_second'] = len(tokens) / results['caption_time']
                
                # Track memory after generation
                post_generation_memory = get_memory_usage()
                results['post_generation_memory'] = post_generation_memory
                
                # Calculate peak memory increases
                results['peak_cpu_increase'] = max(
                    post_model_memory['cpu'] - initial_memory['cpu'],
                    post_encode_memory['cpu'] - initial_memory['cpu'],
                    post_generation_memory['cpu'] - initial_memory['cpu']
                )
                
                if torch.cuda.is_available():
                    results['peak_gpu_increase'] = max(
                        post_model_memory['gpu'] - initial_memory['gpu'],
                        post_encode_memory['gpu'] - initial_memory['gpu'],
                        post_generation_memory['gpu'] - initial_memory['gpu']
                    )

                # Add run results
                all_runs.append(results)
                
            except Exception as e:
                print(f"❌ Run {run + 1} failed: {str(e)}")
                continue
        
        # Cleanup only once at the end
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not all_runs:
            raise Exception("All runs failed")
        
        # Print results
        print("\n🔍 Results Summary:")
        print("=" * 60)
        
        metrics = ['model_load_time', 'cold_encode_time', 'avg_warm_encode_time', 
                  'ttft', 'caption_time', 'tokens_per_second']
        
        for metric in metrics:
            values = [run[metric] for run in all_runs]
            mean = sum(values) / len(values)
            std = (sum((x - mean)**2 for x in values) / len(values))**0.5
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean: {mean:.2f}")
            print(f"  Std:  {std:.2f}")
        
        print(f"\nCaption: {all_runs[0]['caption']}")
        print(f"\nMemory Usage:")
        print(f"  Peak CPU Increase: {all_runs[0]['peak_cpu_increase']:.1f} MB")
        if torch.cuda.is_available():
            print(f"  Peak GPU Increase: {all_runs[0]['peak_gpu_increase']:.1f} MB")
        
        # Save results
        save_results(model_name, all_runs, image_load_time)
            
    except Exception as e:
        print(f"❌ Model initialization failed: {str(e)}")
        return
        
    return all_runs

if __name__ == "__main__":
    print("\n🤖 Available Models:")
    print("1. 0.5B INT8")
    print("2. 0.5B INT4")
    print("3. 2B INT8")
    print("4. 2B INT4")
    
    while True:
        choice = input("\nSelect model (1-4): ")
        try:
            model_idx = int(choice) - 1
            if 0 <= model_idx <= 3:
                models = ['0.5B INT8', '0.5B INT4', '2B INT8', '2B INT4']
                run_benchmark(models[model_idx])
                break
            print("❌ Invalid selection. Please try again.")
        except:
            print("❌ Invalid input. Please enter a number 1-4.")