# ===== STEP 1: Clean and reinstall dependencies =====
# Install other required packages
!pip install -q moondream
!pip install -q Pillow
!pip install -q requests

# ===== STEP 2: Mount Google Drive =====
from google.colab import drive
drive.mount('/content/drive')

# ===== STEP 3: Import Required Libraries =====
import moondream as md
from PIL import Image
import requests
from io import BytesIO
from time import perf_counter
import os

# ===== Model Configuration =====
MODEL_PATH = '/content/drive/MyDrive/md-models/moondream-0_5b-int8.mf.gz'
MODEL_URL = 'https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz'

# Available Model URLs and Paths:
# 2B INT8: 
#   URL: https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz
#   Path: /content/drive/MyDrive/md-models/moondream-2b-int8.mf.gz
# 2B INT4:
#   URL: https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int4.mf.gz
#   Path: /content/drive/MyDrive/md-models/moondream-2b-int4.mf.gz
# 0.5B INT8:
#   URL: https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz
#   Path: /content/drive/MyDrive/md-models/moondream-0_5b-int8.mf.gz
# 0.5B INT4:
#   URL: https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int4.mf.gz
#   Path: /content/drive/MyDrive/md-models/moondream-0_5b-int4.mf.gz
# Note: Update MODEL_PATH and MODEL_URL above to switch models

def setup_model():
    """
    Downloads and sets up the Moondream model if not already present.
    Returns the initialized model.
    """
    # Create models directory if it doesn't exist
    os.makedirs('/content/drive/MyDrive/md-models', exist_ok=True)

    print("🔄 Setting up model...")
    model_start = perf_counter()
    
    # Download model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model... (this may take a few minutes)")
        os.system(f"wget -O {MODEL_PATH} {MODEL_URL}")
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists in Drive!")

    # Initialize the model
    model = md.vl(model=MODEL_PATH)
    model_end = perf_counter()
    print(f"⏱️ Model load time: {model_end - model_start:.2f}s")
    
    return model

def measure_ttft(model, image_url=None):
    """
    Measures the time to first token for image captioning.
    Args:
        model: Initialized Moondream model
        image_url: Optional URL for custom image testing
    """
    if image_url is None:
        image_url = "https://cdn.pixabay.com/photo/2023/01/30/11/04/cat-7755394_1280.jpg"
    
    # Download and load image
    print("\n📥 Loading image...")
    image_start = perf_counter()
    image = Image.open(BytesIO(requests.get(image_url).content))
    image_end = perf_counter()
    print(f"⏱️ Image load time: {image_end - image_start:.2f}s")

    # Encode image
    print("\n🔄 Encoding image...")
    encode_start = perf_counter()
    encoded_image = model.encode_image(image)
    encode_end = perf_counter()
    print(f"⏱️ Image encode time: {encode_end - encode_start:.2f}s")

    # Measure time to first token
    print("\n⏱️ Measuring time to first token...")
    ttft_start = perf_counter()
    first_token = next(iter(model.caption(encoded_image, stream=True)["caption"]))
    ttft_end = perf_counter()

    # Print results
    print(f"\n📊 Results:")
    print(f"First token: '{first_token}'")
    print(f"Time to first token: {ttft_end - ttft_start:.2f}s")
    print(f"Total pipeline time: {ttft_end - image_start:.2f}s")

def main():
    """Main function to run the TTFT benchmark"""
    print("🚀 Starting Moondream TTFT Benchmark...")
    
    # Setup model
    model = setup_model()
    
    # Run TTFT measurement
    measure_ttft(model)

if __name__ == "__main__":
    main()
