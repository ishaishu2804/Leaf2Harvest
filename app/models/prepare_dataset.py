import os
import requests
import zipfile
from tqdm import tqdm
import shutil

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # Create data directory if it doesn't exist
    data_dir = os.path.join('data', 'plant_disease')
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the PlantVillage dataset (small version for testing)
    print("[INFO] Downloading dataset...")
    dataset_url = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    zip_path = os.path.join(data_dir, 'plant_disease.zip')
    
    try:
        download_file(dataset_url, zip_path)
        print("[INFO] Dataset downloaded successfully.")
        
        # Extract the dataset
        print("[INFO] Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up
        os.remove(zip_path)
        print("[INFO] Dataset preparation completed.")
        
    except Exception as e:
        print(f"[ERROR] Failed to download or prepare dataset: {e}")
        return

if __name__ == "__main__":
    main() 