import os
import json
import torchvision
import torch.nn as nn

def load_hyperparameters(model_path):
    """
    Load hyperparameters from a JSON file located in the same directory as the model weights.
    Expects a 'hyperparameters.json' file to exist alongside the model weights file.

    Parameters:
        model_path (str): Path to the model weights file

    Returns:
        dict: Dictionary containing model hyperparameters
    
    Raises:
        FileNotFoundError: If hyperparameters.json is not found
    """
    hyperparams_path = os.path.join(os.path.dirname(model_path), "hyperparameters.json")
    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)
    return hyperparams

def add_input_channel(model, in_channels):
    """
    Modifies a model's first convolution layer to accept a different number 
    of input channels while preserving pre-trained weights where possible.
    Currently supports ResNet and EfficientNet architectures.

    Parameters:
        model (nn.Module): The PyTorch model to modify
        in_channels (int): Desired number of input channels

    Returns:
        nn.Module: Modified model with updated input channels

    Notes:
        - If in_channels=3, returns the original model unchanged
        - For ResNet: Modifies the 'conv1' layer
        - For EfficientNet: Modifies the first conv layer in 'features'
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
    Loads and configures a pre-trained model for the specified number of classes
    and input channels. Supports various architectures from torchvision.models.

    Parameters:
        model_name (str): Name of the model architecture to load. Supported options: 
            ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
            'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
        num_classes (int): Number of output classes for the model
        device (torch.device): Device to load the model onto
        in_channels (int): Number of input channels for the model

    Returns:
        nn.Module: Configured model

    Raises:
        ValueError: If an unsupported model name is provided
    """
    # ResNet models
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet152":
        model = torchvision.models.resnet152(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # EfficientNet models
    elif model_name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b1":
        model = torchvision.models.efficientnet_b1(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b2":
        model = torchvision.models.efficientnet_b2(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b3":
        model = torchvision.models.efficientnet_b3(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b4":
        model = torchvision.models.efficientnet_b4(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b5":
        model = torchvision.models.efficientnet_b5(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b6":
        model = torchvision.models.efficientnet_b6(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b7":
        model = torchvision.models.efficientnet_b7(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = add_input_channel(model, in_channels)
    return model.to(device)