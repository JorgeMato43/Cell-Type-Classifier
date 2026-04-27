### Use this code to download Live Cell data
!git clone https://github.com/JorgeMato43/Cell-Type-Classifier.git
%cd Cell-Type-Classifier
import sys
sys.path.append('/content/Cell-Type-Classifier')

from src.data_loader import *

ROOT = "/content/livecell"
ANN_DIR, IMG_DIR = make_dir(ROOT)

download_livecell_data(ROOT, ANN_DIR)
unzip_images(ROOT, IMG_DIR=IMG_DIR)
# Load COCO annotations
from pycocotools.coco import COCO

coco = COCO("/content/livecell/annotations/livecell_coco_train.json")

img_ids = coco.getImgIds()
img_info = coco.loadImgs(img_ids[0])[0]

print(img_info)
