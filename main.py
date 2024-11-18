import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import random
import numpy as np

from utils.dataload import LandClassDataset

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def train(args, model, trainloader, valloader, criterion, optimizer, device):
    best_val_accuracy = 0.0
    for epoch in range(args.num_epochs):
        print(f'Epoch [{epoch+1}/{args.num_epochs}]')
        model.train()
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

            if (batch_idx + 1) % args.print_iter == 0:
                batch_loss = running_loss / (batch_idx + 1)
                batch_accuracy = correct / total
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Batch [{batch_idx+1}/{len(trainloader)}], "
                    f"Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.3f}")
        
        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct / total
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.3f}")


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
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.3f}\n")

        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            best_model_state = model.state_dict()

    return best_model_state, best_val_accuracy

def main():
    parser = ArgumentParser(description="Train a model on Land Cover Dataset")
    parser.add_argument('--data_dir', type=str, default='./data/land_cover_representation', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate used for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--save_path', type=str, default='./weights/best_model.pth', help='Path to save the best model')
    parser.add_argument('--seed', type=int, default=42, help='Set randomness seed')
    parser.add_argument('--print_iter', type=int, default=1000, help='Set number of iterations between printing updates in training')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((100, 100)),
        T.ToTensor()
    ])

    print('Loading training and validation datasets...')

    trainset = LandClassDataset(args.data_dir, transform=transform, split='train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valset = LandClassDataset(args.data_dir, transform=transform, split='val')
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Datasets successfully loaded')

    print('Loading model...')

    # Load in ResNet18
    model = torchvision.models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, trainset.get_num_classes())
    model = model.to(device)

    print('Model loaded')

    print("Calculating class weights...")
    class_weights = trainset.get_class_weights()
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
    print(class_weights)
    print(f"Class weights obtained")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f'Starting training loop, Epochs: {args.num_epochs}, Learning Rate: {args.lr}')
    # Train and validate
    model, val_accuracy = train(
        args, model, trainloader, valloader, criterion, optimizer, device=device
    )

    torch.save({
        'model_state_dict': model,
        'val_accuracy': val_accuracy
    }, args.save_path)

    print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    main()