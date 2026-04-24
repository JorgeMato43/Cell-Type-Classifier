# Import packages and modules
import os
import urllib.request
import zipfile

# Make folders

ROOT = "/content/livecell"
IMG_DIR = os.path.join(ROOT, "images")
ANN_DIR = os.path.join(ROOT, "annotations")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)

  
# Download images and annotation files into corresponding folders:
# Takes a couple of minutes

files_to_download = {
    "images_zip": (
        "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip",
        os.path.join(ROOT, "images.zip")
    ),
    "train_json": (
        "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json",
        os.path.join(ANN_DIR, "livecell_coco_train.json")
    ),
    "val_json": (
        "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json",
        os.path.join(ANN_DIR, "livecell_coco_val.json")
    ),
    "test_json": (
        "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json",
        os.path.join(ANN_DIR, "livecell_coco_test.json")
    ),
}

for name, (url, out_path) in files_to_download.items():
    if not os.path.exists(out_path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, out_path)
    else:
        print(f"Already exists: {out_path}")

print("Done downloading images")


# Unzip images.

zip_path = os.path.join(ROOT, "images.zip")

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(IMG_DIR)

print("Unzipped to:", IMG_DIR)

