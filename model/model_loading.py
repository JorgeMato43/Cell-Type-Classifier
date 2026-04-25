import torch.nn as nn
import torchvision.models as models

def get_resnet_model(num_classes, pretrained=True):
  '''
  Load resnet model from torchvision
  '''
  
  model = models.resnet18(pretrained=pretrained)

  # Replace final layer for classification
  model.fc = nn.Linear(model.fc.in_features, num_classes)

  return model

def load_model_checkpoint(path, num_classes):
  '''
  Load model weights and optimizer states from previous training
  '''
  model = get_resnet_model(num_classes)
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  return model
