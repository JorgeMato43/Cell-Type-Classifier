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

def train_val_test_data_dir_split(data_dir,
                                  test_data_dir=None,
                                  val_dir=None,
                                  train_dir=None,
                                  test_dir=None):
  '''
  Takes the data directory and the desired test, train,
  and validation data directories, and splits the data into said directories.
  '''

  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(val_dir, exist_ok=True)
  os.makedirs(test_dir, exist_ok=True)

  if train_dir is not None and val_dir is not None:

    for cls in os.listdir(data_dir):
        imgs = os.listdir(os.path.join(data_dir, cls))
        random.shuffle(imgs)

        split = int(0.8 * len(imgs))

        for i, img in enumerate(imgs):
            src = os.path.join(data_dir, cls, img)

            if i < split:
                dst = os.path.join(train_dir, cls)
            else:
                dst = os.path.join(val_dir, cls)

            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, os.path.join(dst, img))

  if test_dir is not None:
    for cls in os.listdir(data_dir):
      imgs = os.listdir(os.path.join(data_dir, cls))
      random.shuffle(imgs)

      for i, img in enumerate(imgs):
          src = os.path.join(data_dir, cls, img)

          dst = os.path.join(test_dir, cls)

          os.makedirs(dst, exist_ok=True)
          shutil.copy(src, os.path.join(dst, img))


def make_datasets(train_dir, val_dir, test_dir):
  '''
  Takes the training, validation, and testing data directories as inputs and returns the corresponding dataset and the number of different classes
  '''
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5])
  ])

  train_dataset = datasets.ImageFolder(train_dir, transform=transform)
  val_dataset = datasets.ImageFolder(val_dir, transform=transform)
  test_dataset = datasets.ImageFolder(test_out_dir, transform=transform)
  num_classes = len(train_dataset.classes)

  return train_dataset, val_dataset, test_dataset, num_classes

# Selecting a smaller set to train:


def adjust_dataset(train_dataset, val_dataset, test_dataset, 
                   train_size, val_size, test_size):

  '''
  Takes the train, validation, and test datasets along with their desired size 
  and returns datasets of those sizes with elements randomly selected from the input datasets.
  '''
  indices = random.sample(range(len(train_dataset)), train_size)
  train_dataset = Subset(train_dataset, indices)
  val_indices = random.sample(range(len(val_dataset)), val_size)
  val_dataset = Subset(val_dataset, val_indices)
  test_indices = random.sample(range(len(test_dataset)), test_size)
  test_dataset = Subset(test_dataset, test_indices)

  return train_dataset, val_dataset, test_dataset
