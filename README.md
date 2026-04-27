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
learning rate: 0.01
best accuracy: 0.57

Fine-tuning
<img width="1784" height="585" alt="image" src="https://github.com/user-attachments/assets/4bfd5598-d412-436f-8235-b37bc546abff" />
Sample size: 5000  
learning rate: 0.001
best accuracy: 0.77

Fine-tuning with a learning rate of 0.001 was the best training framework, so next, the model was fine-tuned on 100,000 with this learning rate.

<img width="1725" height="567" alt="image" src="https://github.com/user-attachments/assets/e48edaec-04ac-4125-93be-aa05621c077a" />

The model achieved an accuracy of 0.83 with 100,000 samples. It may seem that this is not much different from 0.77. However, the contexts in which a model like this could be deployed demand high accuracy. For example, a medical setting cannot afford a model that makes classification mistakes about a patient's cells, so any improvement in accuracy (and other performance metrics) is necessary. 

Next, I fine-tuned SAM with the goal of having an image segmentation model that could create masks around cells before ResNet18 classifies them. 
<img width="1725" height="566" alt="image" src="https://github.com/user-attachments/assets/14f5e59e-1ced-49bd-a0fd-ebb6708540db" />

Although training was stable, SAM's performance hardly improved. Deep inspection of the data proved that images of cells with blurry borders or whose shape was not round were hard for SAM to segment (image on the right). 
<img width="1789" height="604" alt="image" src="https://github.com/user-attachments/assets/d2518e70-5b6e-4252-bbbf-83bf9a031288" />




