import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

# Cropping training and validation images to get individual cells

ann_file = "/content/livecell/annotations/livecell_coco_train.json"
img_dir = "/content/livecell/images/images/livecell_train_val_images"
out_dir = "/content/cell_crops/"

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
