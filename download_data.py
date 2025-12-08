import os
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Configuration
BASE_PATH = './data'
INPUT_DIR = os.path.join(BASE_PATH, 'input_images')
NUM_IMAGES = 60  # From Cell 2

def download_image(url, save_path):
    """Simple download helper from Cell 1"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(save_path)
        print(f"Success: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def download_coco_image(image_id):
    """COCO ID downloader from Cell 2"""
    base_url = "http://images.cocodataset.org/val2017/"
    formatted_id = str(image_id).zfill(12)
    url = f"{base_url}{formatted_id}.jpg"

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=2)
        if resp.status_code == 200:
            if len(resp.content) > 50000: # >50KB Filter
                save_path = os.path.join(INPUT_DIR, f"image_{formatted_id}.jpg")
                Image.open(BytesIO(resp.content)).save(save_path)
                return True
    except:
        pass
    return False

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # --- PART 1: The Specific Complex Scenes (Cell 1) ---
    print("Downloading specific complex scenes...")
    coco_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg", 
        "http://images.cocodataset.org/val2017/000000000139.jpg", 
        "http://images.cocodataset.org/val2017/000000289343.jpg"  
    ]
    for i, url in enumerate(coco_urls):
        filename = f"image_{i+1:03d}.jpg" # image_001.jpg, etc.
        download_image(url, os.path.join(INPUT_DIR, filename))

    # --- PART 2: The Bulk Download (Cell 2) ---
    print(f"\nDownloading remaining {NUM_IMAGES} images from COCO...")
    count = 0
    current_id = 139 # Starting ID from your notebook
    
    while count < NUM_IMAGES:
        if download_coco_image(current_id):
            count += 1
            print(f"[{count}/{NUM_IMAGES}] Downloaded image_{str(current_id).zfill(12)}.jpg")
        current_id += 1
        if current_id > 10000: break

    print(f"\nDownload Complete. Images in {INPUT_DIR}")

if __name__ == "__main__":
    main()
