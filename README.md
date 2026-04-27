# Cell-Type-Classifier
This project focused on training an image classifier that assigns classes to microscopy images of cropped cells instead of the entire image

## What it does
This project builds a pipeline for cell segmentation and classification using the LIVECell dataset, which contains over one million manually annotated microscopy images. Using the annotations, it creates cropped cell images from the original images. These crops are then fed into a fine-tuned ResNet18 model, a convolutional neural network with residual connections, to classify each cell by type. Finally, it fine-tunes Meta’s Segment Anything Model (SAM) to segment cells from microscopy images and generate individual cell crops without needing human annotations.

## Quick Start
To run this project, clone this repository into your workspace, then download all the data with the code in the loading data notebook located in the notebooks folder. All Jupiter notebooks incorporate these steps so that you can conduct or verify their results individually. Notice: Fine-tuning ResNet18 on 100,000 and SAM on 10,000 samples may take some time.

## Video Links
Project Overview: https://github.com/JorgeMato43/Cell-Type-Classifier/blob/main/Videos/Project%20Demo.mp4
Technical Walkthrough: https://github.com/JorgeMato43/Cell-Type-Classifier/blob/main/Videos/Technical%20Walkthrough.mp4

## Evaluation
### Ablation Study
This project conducted an ablation study to determine the training modality that best fit this task and the best learning rate. The two modalities tested were feature extraction and fine-tuning. Here I show the best training curves, this is, with the best learning rates for each modalities. The accuracy curve is shown in the right as a measure of model performance. 

Feature Extraction
<img width="1725" height="565" alt="image" src="https://github.com/user-attachments/assets/290f8519-60d7-4f38-b5da-84a281138fa5" />
Sample size: 5000  
best accuracy: 0.57

