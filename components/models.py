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

def load_model(model_name, num_classes, device):
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
    return model.to(device)