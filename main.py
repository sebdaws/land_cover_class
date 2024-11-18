import os
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils.dataload import LandClassDataset

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def trainer(model, trainloader, valloader, criterion, optimizer, num_epochs, device):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
            
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 500 == 0:
                batch_loss = running_loss / (batch_idx + 1)
                batch_accuracy = correct / total
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(trainloader)}], "
                    f"Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.3f}")
        
        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.3f}")


        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        print('Validation Phase...')

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate validation loss and accuracy
        val_epoch_loss = val_loss / len(valloader)
        val_epoch_accuracy = val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.3f}")

        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            best_model_state = model.state_dict()

    return best_model_state, best_val_accuracy

def main():
    device = get_device()
    print(f"Using device: {device}")

    if os.path.isfile('split_file.csv') is False:
        with open('data/land_cover_representation/metadata.csv') as f:
            metadata = pd.read_csv(f)
        metadata = metadata[metadata['split_str'] == 'train']

    transform = T.Compose([
        T.Resize((100, 100)),
        T.ToTensor()
    ])

    trainset = LandClassDataset('data/land_cover_representation', transform=transform, split='train')
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

    valset = LandClassDataset('data/land_cover_representation', transform=transform, split='val')
    valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=4)


    # Load in ResNet18
    model = torchvision.models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 66)
    model = model.to(device)

    # Hyperparameter grid
    learning_rates = [0.01, 0.001, 0.0001]

    # Track the best configuration
    best_accuracy = 0.0
    best_lr = None
    best_model_state = None

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)#, num_workers=4)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)#, num_workers=4)

    # Load in ResNet18
    model = torchvision.models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 61)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    for lr in learning_rates:
        print(f'Training with learning rate of {lr}')
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train and validate
        model_state, val_accuracy = trainer(
            model, trainloader, valloader, criterion, optimizer, num_epochs=5, device=device
        )

        print(f'Learning rate: {lr}, Val Accuracy: {val_accuracy:.4f}')

        # Update the best configuration
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_lr = lr
            best_model_state = model_state
        

    print(f"Best configuration: {best_lr} with Validation Accuracy: {best_accuracy:.4f}")

    # Save the best model
    save_path = './weights/best_model.pth'
    torch.save({
        'model_state_dict': best_model_state,
        'best_lr': best_lr,
        'best_accuracy': best_accuracy
    }, save_path)

    print(f"Best model saved to {save_path}")


if __name__ == '__main__':
    main()