# Cell Type Classifier Setup
This project consists of two main models working together to achieve the goal of classifying cells into their corresponding types. One model is a fine-tuned ResNet18, a Convolutional Neural Network with residual blocks from torchvision. The second model is segment anything from Meta (SAM). Follow these steps to get started using these models.

## System Requirements
- Python 10 or above
- Python 3.10+
- CUDA-enabled GPU (recommended)
- ~10GB free disk space

## Clone this repository with this code:
`!git clone https://github.com/JorgeMato43/Cell-Type-Classifier.git`

`%cd Cell-Type-Classifier`

`import sys`

`sys.path.append('/content/Cell-Type-Classifier')`

## Install dependencies and download and reach APIs
- To use SAM, clone the git repository and install SAM:

`!git clone https://github.com/facebookresearch/segment-anything.git`

`!pip install -e ./segment-anything`

You may need to restart the runtime after this. 

- Then download the `vit_b` checkpoint:

`!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

- Install other packages, including pycocotools:

`!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

`!pip install matplotlib opencv-python pycocotools`



