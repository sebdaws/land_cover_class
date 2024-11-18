import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils.dataload import LandClassDataset

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def test_model(args, model, testloader, class_names, device):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.3f}")
    
    if class_names:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    if args.confusion_matrix:
        cm = confusion_matrix(all_labels, all_predictions)
        # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

        # cm_display.plot()
        # plt.show()
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    return accuracy

def main():
    parser = ArgumentParser(description="Test a trained model on Land Cover Dataset")
    parser.add_argument('--data_dir', type=str, default='./data/land_cover_representation', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model weights')
    parser.add_argument('--confusion_matrix', action='store_true', help='Plot the confusion matrix')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((100, 100)),
        T.ToTensor()
    ])

    print('Loading test dataset...')
    testset = LandClassDataset(args.data_dir, transform=transform, split='test')
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Test dataset successfully loaded')

    print('Loading model...')
    if args.model_path:
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, testset.get_num_classes())
        model = model.to(device)

        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {args.model_path}")
    else:
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, testset.get_num_classes())
        model = model.to(device)
        print(f'Default pretrained model loaded')

    class_names = testset.get_class_names()

    print("Starting testing...")
    test_accuracy = test_model(args, model, testloader, class_names, device)

    print(f"Test accuracy: {test_accuracy:.3f}")

if __name__ == '__main__':
    main()