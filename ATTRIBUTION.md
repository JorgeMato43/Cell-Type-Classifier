This project was created using the following resources:

-Segment Anything Model (SAM) developed by Meta AI.
Repository: https://github.com/facebookresearch/segment-anything

Paper: Kirillov et al., "Segment Anything", 2023   https://arxiv.org/abs/2304.02643

-LIVECell dataset for training and evaluation.

Source: https://www.nature.com/articles/s41592-021-01249-6

Dataset: [https://livecell-dataset.s3.amazonaws.com/](https://sartorius-research.github.io/LIVECell/)

Citation: Edlund et al., "LIVECell—A large-scale dataset for label-free live cell segmentation", Nature Methods, 2021.

- Summary of libraries and APIs used (A full list can be found in the requirements.txt file):
Pytorch, random,

Coco API (https://github.com/cocodataset/cocoapi) for loading LIVECell annotations and converting annotations to masks,
google colab,

NumPy for numerical operations,

Matplotlib for visualization,

scikit-learn for dataset splitting,


- AI Usage in this project:
AI was used in each of the following 3 categories:

I. Debug Functions written by me.
save_checkpoint(), load_checkpoint, unzip_images(), make_datasets(), code to show images in the SAM Jupiter notebook (both with Coco masks and with SAM masks)

II. Complement Functions written by me.
validate_sam(), train_one_epoch_sam(), crop_cell(), embedding_caching, download_livecell_data(), train_val_test_data_dir_split(), adjust_dataset()

III. Write code snippets that perform a specific task.
train_val_test_data_dir_split(), 

- Developed without the use of AI:
Project idea and design, literature review, choice of platform to develop the project, ablation study design and code and hyperparameter choice, visualization design, train_model() function, plot_training_curves(), plot_ablation_training_curves()



