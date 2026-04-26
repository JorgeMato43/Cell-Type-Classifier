import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import random
import shutil
import matplotlib.pyplot as plt


def crop_cell(ann_file, img_dir, out_dir):
  '''
  Crops cells out of each image in the img_dir directory and places each individual 
  cropped cell in the out_dir directory using the corresponding ann_file
  '''
  os.makedirs(out_dir, exist_ok=True)

  coco = COCO(ann_file)

  for img_id in tqdm(coco.getImgIds()):
      img_info = coco.loadImgs(img_id)[0]
      img_path = os.path.join(img_dir, img_info['file_name'])

      img = cv2.imread(img_path)
      if img is None:
          continue

      # label mapping derived from filename
      label = img_info['file_name'].split('_')[0]

      ann_ids = coco.getAnnIds(imgIds=img_id)
      anns = coco.loadAnns(ann_ids)

      for ann in anns:
          mask = coco.annToMask(ann)

          x, y, w, h = map(int, ann['bbox'])

          crop = img[y:y+h, x:x+w]
          mask_crop = mask[y:y+h, x:x+w]

          if crop.shape[0] < 10 or crop.shape[1] < 10:
              continue

          crop = crop * mask_crop[:, :, None]

          # resize to CNN-friendly size
          crop = cv2.resize(crop, (224, 224))

          class_dir = os.path.join(out_dir, label)
          os.makedirs(class_dir, exist_ok=True)

          save_path = os.path.join(class_dir, f"{img_id}_{ann['id']}.png")
          cv2.imwrite(save_path, crop)



def train_val_test_data_dir_split(data_dir, 
                                  test_data_dir=None,
                                  train_dir=None, 
                                  val_dir=None, 
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



def segment_image(mask_generator, image_path, file_name):
  '''
  Shows the provided image with an outline segmenting the detected objects 
  '''
  img_path = os.path.join(
      image_path,
      file_name
  )

  image = cv2.imread(img_path)

  if image is None:
      raise ValueError(f"Failed to load image at {img_path}")

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  masks = mask_generator.generate(image)

  plt.imshow(image)
  for m in masks:
      plt.contour(m['segmentation'], colors='r', linewidths=0.5)
  plt.axis('off')
  plt.show()

