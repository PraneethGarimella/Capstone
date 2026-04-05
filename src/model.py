import torch
import torch.nn as nn
import torchvision.models.segmentation as models

def get_model():
    model = models.deeplabv3_resnet50(pretrained=True)

    # Change classifier for 1 output channel
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    return model
