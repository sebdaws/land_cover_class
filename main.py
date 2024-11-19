import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from argparse import ArgumentParser
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import json

from utils.dataload import LandClassDataset

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def calculate_metrics(y_true, y_pred, num_classes):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def train(args, model, trainloader, valloader, criterion, optimizer, device):
    metrics_df = pd.DataFrame()
    best_val_accuracy = 0.0
    for epoch in range(args.num_epochs):
        print(f'Epoch [{epoch+1}/{args.num_epochs}]')
        epoch_metrics = {'Epoch': epoch + 1}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = trainloader
            else:
                model.eval()
                loader = valloader

            running_loss = 0.0
            running_loss = 0.0
            correct = 0
            total = 0
            y_true = []
            y_pred = []

            # Process batches
            for batch_idx, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate loss
                running_loss += loss.item()

                # Store predictions and ground truths
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                if (batch_idx + 1) % args.print_iter == 0 and phase == 'train':
                    batch_loss = running_loss / (batch_idx + 1)
                    batch_accuracy = correct / total
                    print(f"Epoch [{epoch+1}/{args.num_epochs}], Batch [{batch_idx+1}/{len(trainloader)}], "
                        f"Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.3f}")

            # Calculate epoch metrics
            precision, recall, f1, accuracy = calculate_metrics(y_true, y_pred, trainloader.dataset.get_num_classes())
            epoch_loss = running_loss / len(loader)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")

            # Store metrics in dictionary
            epoch_metrics[f'{phase}_Loss'] = epoch_loss
            epoch_metrics[f'{phase}_Precision'] = precision
            epoch_metrics[f'{phase}_Recall'] = recall
            epoch_metrics[f'{phase}_F1-Score'] = f1
            epoch_metrics[f'{phase}_Accuracy'] = accuracy


            # Save the best model (validation phase only)
            if phase == 'val' and accuracy > best_val_accuracy:
                best_val_accuracy = accuracy
                best_model_state = model.state_dict()
        
        # Append the epoch metrics to the DataFrame
        metrics_df = pd.concat([metrics_df, pd.DataFrame([epoch_metrics])], ignore_index=True)

    return best_model_state, best_val_accuracy, metrics_df



        # model.train()
        # running_loss = 0.0
        # correct = 0
        # total = 0
            
        # for batch_idx, (images, labels) in enumerate(trainloader):
        #     images, labels = images.to(device), labels.to(device)
        #     optimizer.zero_grad()
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
            
        #     running_loss += loss.item()
            
        #     _, predicted = torch.max(outputs, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()

        #     if (batch_idx + 1) % args.print_iter == 0:
        #         batch_loss = running_loss / (batch_idx + 1)
        #         batch_accuracy = correct / total
        #         print(f"Epoch [{epoch+1}/{args.num_epochs}], Batch [{batch_idx+1}/{len(trainloader)}], "
        #             f"Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.3f}")
        
        # # Calculate epoch loss and accuracy
        # epoch_loss = running_loss / len(trainloader)
        # epoch_accuracy = correct / total
        
        # print(f"Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.3f}")


        # # Validation phase
        # model.eval()
        # val_loss = 0.0
        # val_correct = 0
        # val_total = 0

        # print('Validation Phase...')

        # with torch.no_grad():
        #     for images, labels in valloader:
        #         images, labels = images.to(device), labels.to(device)
        #         outputs = model(images)
        #         loss = criterion(outputs, labels)

        #         val_loss += loss.item()
        #         _, predicted = torch.max(outputs, 1)
        #         val_total += labels.size(0)
        #         val_correct += (predicted == labels).sum().item()

        # # Calculate validation loss and accuracy
        # val_epoch_loss = val_loss / len(valloader)
        # val_epoch_accuracy = val_correct / val_total
        # print(f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.3f}\n")

        # if val_epoch_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_epoch_accuracy
        #     best_model_state = model.state_dict()

    # return best_model_state, best_val_accuracy

def main():
    parser = ArgumentParser(description="Train a model on Land Cover Dataset")
    parser.add_argument('--data_dir', type=str, default='./data/land_cover_representation', help='Path to dataset')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate used for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--save_dir', type=str, default='./experiments', help='Directory to save the trained model')
    parser.add_argument('--seed', type=int, default=42, help='Set randomness seed')
    parser.add_argument('--print_iter', type=int, default=1000, help='Set number of iterations between printing updates in training')
    parser.add_argument('--balance_weights', action='store_true', help='Balance the class weights for training')
    parser.add_argument('--weights_smooth', type=float, default=0.1, help='Amount added to smooth class weights')
    parser.add_argument('--over_sample', action='store_true', help='Over-sample minority classes')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((100, 100)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor()
    ])

    print('Loading training and validation datasets...')

    trainset = LandClassDataset(args.data_dir, transform=transform, split='train')

    if args.over_sample:
        class_weights = trainset.get_class_weights()
        sample_weights = [class_weights[label] for _, label in trainset]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, sampler=sampler, num_workers=args.num_workers)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valset = LandClassDataset(args.data_dir, transform=transform, split='val')
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Datasets successfully loaded')

    print('Loading model...')

    if args.model_name == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, trainset.get_num_classes())
    else:
        raise ValueError('Script not designed for this model.')
    model = model.to(device)

    print('Model loaded')

    if args.balance_weights:
        print("Calculating class weights...")
        class_weights = trainset.get_class_weights()
        class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
        smoothed_weights = class_weights + args.weights_smooth
        print(f"Class weights obtained")

        criterion = nn.CrossEntropyLoss(weight=smoothed_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f'Starting training loop, Epochs: {args.num_epochs}, Learning Rate: {args.lr}')

    model, val_accuracy, metrics_df = train(
        args, model, trainloader, valloader, criterion, optimizer, device=device
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = os.path.join(args.save_dir, args.model_name)
    run_dir = os.path.join(model_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True) 

    model_weights_path = os.path.join(run_dir, "model_weights.pth")
    torch.save({
        'model_state_dict': model,
        'val_accuracy': val_accuracy
    }, model_weights_path)
    print(f"Model weights saved to {model_weights_path}")

    metrics_path = os.path.join(run_dir, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Save hyperparameters to JSON
    hyperparams = {
        "model_name": args.model_name,
        "dataset": args.data_dir,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "use_over_sampler": args.over_sample,
        "use_class_weights": args.balance_weights,
        "weights_smooth": args.weights_smooth,
        "final_val_accuracy": val_accuracy,
        "training_date": timestamp
    }

    hyperparams_path = os.path.join(run_dir, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Hyperparameters saved to {hyperparams_path}")
    
    # if args.save_path is None:
    #     save_path = f'./weights/resnet18_{str(args.num_epochs)}e_{str(args.lr)}lr'
    #     if args.balance_weights:    
    #         save_path += '_sw'
    #     save_path += '.pth'
    # else:
    #     save_path = args.save_path

    # torch.save({
    #     'model_state_dict': model,
    #     'val_accuracy': val_accuracy
    # }, save_path)

    # print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()