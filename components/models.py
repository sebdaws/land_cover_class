import os
import json
import torchvision
import torch.nn as nn

def load_hyperparameters(model_path):
    """
    Load hyperparameters from a JSON file located alongside the model weights.
    """
    hyperparams_path = os.path.join(os.path.dirname(model_path), "hyperparameters.json")
    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)
    return hyperparams

def add_input_channel(model, in_channels):
    """
    Modify the first convolution layer of a model to accept a different number of input channels.
    """
    if in_channels == 3:
        return model
        
    if hasattr(model, 'conv1'):  # ResNet
        out_channels = model.conv1.out_channels
        model.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    elif hasattr(model, 'features'):  # EfficientNet
        out_channels = model.features[0][0].out_channels
        model.features[0][0] = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
    return model

def load_model(model_name, num_classes, device, in_channels=3):
    """
    Load the specified model architecture with the correct number of output classes.
    """
    if model_name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b2':
        model = torchvision.models.efficientnet_b2(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b4':
        model = torchvision.models.efficientnet_b4(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b5':
        model = torchvision.models.efficientnet_b5(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet18":
        model = torchvision.models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit_b_16":
        model = torchvision.models.vit_b_16(weights="DEFAULT")
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'swin_t':
        model = torchvision.models.swin_t(weights="DEFAULT")
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = add_input_channel(model, in_channels)
    return model.to(device)