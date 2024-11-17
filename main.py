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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

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

            if (batch_idx + 1) % 100 == 0:
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

    # Save the trained model
    save_path = './weights/trained_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'train_loss': epoch_loss,
        'train_accuracy': epoch_accuracy,
        'val_loss': val_epoch_loss,
        'val_accuracy': val_epoch_accuracy
    }, save_path)

    print(f"Model saved to {save_path}")


