# STEP 1: Install dependencies
!pip install -q moondream
!pip install -q Pillow
!pip install -q requests
!pip install -q opencv-python
!pip install -q tqdm
!pip install -q matplotlib


# Step 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

"""
Object Detection with Moondream
-----------------------------
This script uses Moondream to detect objects in an image.
"""
import cv2
from PIL import Image, ImageDraw
import numpy as np
from tqdm.notebook import tqdm
import os
import time
import datetime
import moondream as md
import requests
import gzip
import shutil

def download_model(model_path, model_url):
    """
    Download and extract the model file if it doesn't exist.
    """
    try:
        if not os.path.exists(model_path.replace('.gz', '')):
            print(f"\nDownloading model from {model_url}...")
            response = requests.get(model_url)
            
            # Save compressed file
            with open(model_path, 'wb') as f:
                f.write(response.content)
            
            # Extract .gz file
            with gzip.open(model_path, 'rb') as f_in:
                with open(model_path.replace('.gz', ''), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove .gz file
            os.remove(model_path)
            print("✓ Model downloaded and extracted successfully")
        else:
            print("✓ Model file already exists")
            
    except Exception as e:
        print(f"\nError downloading model:")
        print(f"• Error type: {type(e).__name__}")
        print(f"• Error message: {str(e)}")
        raise

def get_moondream_model(model_dir="/content/drive/MyDrive/md-models"):
    """
    Initialize Moondream 2B INT8 model.
    """
    try:
        print("\nInitializing Moondream model...")
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Model details
        model_path = os.path.join(model_dir, "moondream-2b-int8.mf.gz")
        model_url = "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz"
        
        # Download model if needed
        download_model(model_path, model_url)
        
        # Initialize model correctly using md.vl()
        model = md.vl(model=model_path.replace('.gz', ''))
        print("✓ Model initialized successfully")
        return model

    except Exception as e:
        print(f"\nError initializing Moondream model:")
        print(f"• Error type: {type(e).__name__}")
        print(f"• Error message: {str(e)}")
        raise

def get_detection_prompt():
    """
    Get detection prompt from user input.
    """
    print("\nWhat would you like to detect?")
    print("Examples:")
    print("1. car")
    print("2. red truck")
    print("3. person on bicycle")
    print("4. custom prompt")

    while True:
        choice = input("\nEnter choice (1-4): ")
        if choice == '1':
            return "car"
        elif choice == '2':
            return "red truck"
        elif choice == '3':
            return "person on bicycle"
        elif choice == '4':
            custom = input("\nEnter your detection prompt: ")
            if custom.strip():
                return custom.strip()
        print("Invalid choice. Please enter 1-4.")

def get_processing_mode():
    """
    Get the processing mode from user input.
    """
    print("\nSelect processing mode:")
    print("1. Full video processing")
    print("2. Test mode - first 30 frames")
    print("3. Test mode - first 3 seconds")

    while True:
        mode_choice = input("\nEnter choice (1-3): ")
        if mode_choice in ['1', '2', '3']:
            mode_map = {
                '1': ('full', None),
                '2': ('test', '30frames'),
                '3': ('test', '3seconds')
            }
            return mode_map[mode_choice]
        print("Invalid choice. Please enter 1-3.")

def detect_objects_in_video(video_path, output_path, model, prompt, test_mode=None):
    """
    Detect objects in video using Moondream model.
    """
    try:
        # Print header
        print("\n" + "="*50)
        print("Moondream Video Detection")
        print("="*50)

        # Verify inputs
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        if not os.path.exists(os.path.dirname(output_path)):
            raise FileNotFoundError(f"Output directory does not exist: {os.path.dirname(output_path)}")

        # Open video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        print("\nInput Video:")
        print(f"• Path: {video_path}")
        print(f"• Resolution: {width}x{height}")
        print(f"• FPS: {fps}")
        print(f"• Total Frames: {total_frames}")
        print(f"• Duration: {total_frames/fps:.1f} seconds")

        print("\nDetection Settings:")
        print(f"• Looking for: \"{prompt}\"")
        print(f"• Processing: {'Full video' if test_mode is None else test_mode}")

        # Create timestamped output path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.replace('.mp4', f'_{prompt.replace(" ", "-")}_{timestamp}.mp4')
        print(f"• Output: {output_path}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Determine frames to process
        if test_mode == '30frames':
            frames_to_process = min(30, total_frames)
        elif test_mode == '3seconds':
            frames_to_process = min(fps * 3, total_frames)
        else:
            frames_to_process = total_frames

        frame_count = 0
        start_time = time.time()
        total_detections = 0

        # Create progress bar
        pbar = tqdm(total=frames_to_process, desc='Processing frames')

        try:
            while video.isOpened() and frame_count < frames_to_process:
                success, frame = video.read()
                if not success:
                    break

                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # First encode the image (more efficient)
                encoded_image = model.encode_image(pil_image)
                
                # Run detection with encoded image
                detections = model.detect(encoded_image, prompt)["objects"]

                if detections:
                    total_detections += len(detections)
                    # Draw detections
                    draw = ImageDraw.Draw(pil_image)
                    for det in detections:
                        x_min = int(det["x_min"] * width)
                        y_min = int(det["y_min"] * height)
                        x_max = int(det["x_max"] * width)
                        y_max = int(det["y_max"] * height)

                        # Draw box in red
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

                # Convert back to OpenCV format and write
                frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

                frame_count += 1
                pbar.update(1)

            end_time = time.time()
            processing_time = end_time - start_time

            print("\nDetection Summary:")
            print("-"*30)
            print(f"• Time taken: {processing_time:.2f} seconds")
            print(f"• Frames processed: {frame_count}")
            print(f"• Processing speed: {frame_count/processing_time:.2f} FPS")
            print(f"• Total detections: {total_detections}")
            print(f"• Detections per frame: {total_detections/frame_count:.2f}")
            print(f"\nOutput saved to: {output_path}")

            return True

        finally:
            pbar.close()
            video.release()
            out.release()

    except Exception as e:
        print(f"\nError in detect_objects_in_video:")
        print(f"• Error type: {type(e).__name__}")
        print(f"• Error message: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Setup Google Drive paths
        DRIVE_INPUT_PATH = "/content/drive/MyDrive/object_detection/input.mp4"
        DRIVE_OUTPUT_PATH = "/content/drive/MyDrive/object_detection/output.mp4"

        print("\n" + "="*50)
        print("Moondream Video Object Detection")
        print("="*50)

        # Initialize model
        model = get_moondream_model()

        # Get user inputs
        prompt = get_detection_prompt()
        mode, test_mode = get_processing_mode()

        # Process video
        success = detect_objects_in_video(
            DRIVE_INPUT_PATH,
            DRIVE_OUTPUT_PATH,
            model,
            prompt,
            test_mode=test_mode
        )

        if not success:
            print("\nProcessing failed. Please check the error messages above.")

    except Exception as e:
        print(f"\nUnexpected error:")
        print(f"• Error type: {type(e).__name__}")
        print(f"• Error message: {str(e)}")