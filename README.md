# Cell-Type-Classifier
This project focused on training an image classifier that assigns classes to microscopy images of cropped cells instead of the entire image

## What it does
This project builds a pipeline for cell segmentation and classification using the LIVECell dataset, which contains over one million manually annotated microscopy images. Using the annotations, it creates cropped cell images from the original images. These crops are then fed into a fine-tuned ResNet18 model, a convolutional neural network with residual connections, to classify each cell by type. Finally, it fine-tunes Meta’s Segment Anything Model (SAM) to segment cells from microscopy images and generate individual cell crops without needing human annotations.

## Quick Start
To run this project, clone this repository into your workspace, then download all the data with the code in the loading data notebook located in the notebooks folder. All Jupiter notebooks incorporate these steps so that you can conduct or verify their results individually. Notice: Fine-tuning ResNet18 on 100,000 and SAM on 10,000 samples may take some time.

## Video Links
Project Overview:
Technical Walkthrough:
