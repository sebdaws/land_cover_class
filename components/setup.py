import os
import random
from datetime import datetime

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T

from utils import get_device
from components.dataload import LandClassDataset
from components.models import load_model, load_hyperparameters
from utils.loss_functions import get_loss_func

def train_setup(args):
    """
    Sets up the training environment using the provided arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        trainloader: DataLoader for the training dataset.
        valloader: DataLoader for the validation dataset.
        model: Initialized model ready for training.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device for computation.
        run_dir: Directory to save experiment results.
    """
    # Set random seed for reproducibility
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Device setup
    device = get_device()
    print(f"Using device: {device}")

    # Define data transformations
    transform = T.Compose([
        T.Resize((100, 100)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor()
    ])

    # Load datasets and DataLoaders
    print('Loading training and validation datasets...')
    trainset = LandClassDataset(args.data_dir, transform=transform, split='train')
    if args.over_sample:
        class_weights = trainset.get_class_weights()
        sample_weights = [class_weights[label] for _, label in trainset]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valset = LandClassDataset(args.data_dir, transform=transform, split='val')
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Datasets successfully loaded')

    # Load model
    print('Loading model...')
    model = load_model(args.model_name, trainset.get_num_classes(), device)
    model = model.to(device)
    print(f'{args.model_name} loaded')

    # Define loss function and optimizer
    criterion = get_loss_func(args, trainset.get_num_classes(), trainset.get_class_weights(), device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.save_dir, args.model_name)
    run_dir = os.path.join(model_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    return trainloader, valloader, model, criterion, optimizer, device, run_dir

def test_setup(args):
    """
    Sets up the environment for testing a trained model on the Land Cover Dataset.

    Args:
        args: Parsed command-line arguments.

    Returns:
        dict: A dictionary containing the following keys:
            - 'device': The computation device.
            - 'testloader': DataLoader for the test dataset.
            - 'class_names': List of class names.
            - 'model': The loaded or default model.
            - 'output_dir': Directory for saving test results.
    """
    device = get_device()
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((100, 100)),
        T.ToTensor(),
    ])

    print("Loading test dataset...")
    testset = LandClassDataset(args.data_dir, transform=transform, split="test")
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_names = testset.get_class_names()
    print("Test dataset successfully loaded")

    print("Loading model...")
    if args.model_path:
        hyperparams = load_hyperparameters(args.model_path)
        model = load_model(hyperparams["model_name"], testset.get_num_classes(), device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {args.model_path}")

        output_dir = os.path.join(
            os.path.dirname(args.model_path), f"test_results_{os.path.dirname(args.model_path)}"
        )
    else:
        print(f'No model path provided. Loading default pretrained resnet18...')
        model = torchvision.models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, testset.get_num_classes())
        model = model.to(device)
        print(f"Default pretrained resnet18 loaded")
        output_dir = "./test_results"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created at {output_dir}")

    return testloader, model, class_names, output_dir, device

