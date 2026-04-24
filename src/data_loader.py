# Import packages and modules
import os
import urllib.request
import zipfile

def make_dir(base):
'''
Makes the necessary images and annotation folders at the base directory provided
and returns their directory address
'''
  ROOT = base
  IMG_DIR = os.path.join(ROOT, "images")
  ANN_DIR = os.path.join(ROOT, "annotations")

  os.makedirs(IMG_DIR, exist_ok=True)
  os.makedirs(ANN_DIR, exist_ok=True)

  print(ROOT)
  return ANN_DIR, IMG_DIR


def download_livecell_data(ROOT, ANN_DIR):
  ''' Download images and annotation files into corresponding folders
  Images are zipped and require unzipping. 
  Annotation files are in the json format and will go to the ANN_DIR directory provided'''

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

  print("Done.")

def unzip_images(ROOT, IMG_DIR):
  '''
  Unzips the images zip file downloaded by 'download_livecell_data' in the 
  IMG_DIR directory
  '''
  zip_path = os.path.join(ROOT, "images.zip")

  with zipfile.ZipFile(zip_path, "r") as zf:
      zf.extractall(IMG_DIR)

  print("Unzipped to:", IMG_DIR)
