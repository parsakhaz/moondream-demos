# https://colab.research.google.com/drive/1gHxqUg7BZ1GYf1rmwrODezW7QHqZPUyE

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
import os

# ===== STEP 4: Download Model, Default is 0.5B INT8 =====
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
    # Create models directory if it doesn't exist
    os.makedirs('/content/drive/MyDrive/md-models', exist_ok=True)

    # Download model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model... (this may take a few minutes)")
        !wget -O $MODEL_PATH $MODEL_URL
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists in Drive!")

    return md.vl(model=MODEL_PATH)

def load_image():
    while True:
        choice = input("\n🖼️ Choose image source:\n1. Sample cat image\n2. Custom URL\n3. From Google Drive\nEnter choice (1-3): ")
        
        try:
            if choice == "1":
                url = "https://cdn.pixabay.com/photo/2023/01/30/11/04/cat-7755394_1280.jpg"
                print("\n📥 Loading sample image...")
                response = requests.get(url)
                return Image.open(BytesIO(response.content))
            
            elif choice == "2":
                url = input("\nEnter image URL: ")
                print("\n📥 Loading image from URL...")
                response = requests.get(url)
                return Image.open(BytesIO(response.content))
            
            elif choice == "3":
                path = input("\nEnter image path in Google Drive (e.g., /content/drive/MyDrive/images/photo.jpg): ")
                print("\n📥 Loading image from Drive...")
                return Image.open(path)
            
            else:
                print("\n❌ Invalid choice. Please try again.")
                
        except Exception as e:
            print(f"\n❌ Error loading image: {str(e)}")
            print("Please try again.")

def interactive_session(model, image):
    print("\n🔄 Encoding image... (this may take 20-30~ seconds on free tier)...")
    encoded_image = model.encode_image(image)
    print("✅ Image encoded successfully!")

    while True:
        print("\n🤖 Choose an action:")
        print("1. Generate image caption")
        print("2. Ask a question about the image")
        print("3. Detect objects in the image")
        print("4. Load different image")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            length = input("\nCaption length (short/normal): ").lower()
            if length not in ["short", "normal"]:
                length = "normal"
            caption = model.caption(encoded_image, length=length)["caption"]
            print(f"\nCaption: {caption}")

        elif choice == "2":
            question = input("\nEnter your question about the image: ")
            print("\nThinking...", end=" ", flush=True)
            for chunk in model.query(encoded_image, question, stream=True)["answer"]:
                print(chunk, end="", flush=True)
            print()

        elif choice == "3":
            object_type = input("\nWhat type of object would you like to detect? ")
            detect_result = model.detect(encoded_image, object_type)
            print(f"\nDetected {object_type}:", detect_result["objects"])

        elif choice == "4":
            return "reload"

        elif choice == "5":
            print("\n👋 Goodbye!")
            return "exit"

        else:
            print("\n❌ Invalid choice. Please try again.")

def main():
    print("🚀 Starting Moondream Interactive Session...")
    model = setup_model()
    
    while True:
        image = load_image()
        print("✅ Image loaded successfully!")
        
        result = interactive_session(model, image)
        if result == "exit":
            break
        elif result == "reload":
            continue

if __name__ == "__main__":
    main()