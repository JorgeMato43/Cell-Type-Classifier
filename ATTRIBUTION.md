This project was created using the following resources:

-Segment Anything Model (SAM) developed by Meta AI.
Repository: https://github.com/facebookresearch/segment-anything
Paper:
Kirillov et al., "Segment Anything", 2023
https://arxiv.org/abs/2304.02643

-LIVECell dataset for training and evaluation.
Source: https://www.nature.com/articles/s41592-021-01249-6
Dataset: [https://livecell-dataset.s3.amazonaws.com/](https://sartorius-research.github.io/LIVECell/)
Citation:
Edlund et al., "LIVECell—A large-scale dataset for label-free live cell segmentation", Nature Methods, 2021.

Summaries of libraries and APIs used (A full list can be found in the requirements.txt file):
Pytorch
random
Coco API (https://github.com/cocodataset/cocoapi) for loading LIVECell annotations and converting annotations to masks
google colab 
NumPy for numerical operations
Matplotlib for visualization
scikit-learn for dataset splitting



