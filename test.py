import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils.dataload import LandClassDataset

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def save_test_results(output_dir, class_names, all_labels, all_predictions):
    """
    Save class-wise test metrics and confusion matrix to files.
    """
    # Generate classification report
    report = classification_report(
        all_labels, all_predictions, target_names=class_names, output_dict=True
    )
    # Save class-wise metrics to a CSV file
    metrics_df = pd.DataFrame(report).transpose()
    metrics_csv_path = os.path.join(output_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=True)
    print(f"Class-wise test metrics saved to {metrics_csv_path}")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="OrRd",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")

    # Save confusion matrix as an image
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")
    plt.close()

# def test_model(args, model, testloader, class_names, device):
#     """
#     Evaluate the model on the test dataset.
#     """
#     model.eval()  # Set the model to evaluation mode
#     correct = 0
#     total = 0
#     all_labels = []
#     all_predictions = []

#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
            
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             all_labels.extend(labels.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())

#     accuracy = correct / total
#     print(f"Test Accuracy: {accuracy:.3f}")
    
#     if class_names:
#         print("\nClassification Report:")
#         print(classification_report(all_labels, all_predictions, target_names=class_names))
    
#     if args.confusion_matrix:
#         cm = confusion_matrix(all_labels, all_predictions)
#         cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.title("Confusion Matrix")
#         plt.show()

#     return accuracy

def test_model(args, model, testloader, class_names, device, output_dir):
    """
    Evaluate the model on the test dataset and save results.
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

    # Save class-wise metrics and confusion matrix
    save_test_results(output_dir, class_names, all_labels, all_predictions)

    return accuracy

# def main():
#     parser = ArgumentParser(description="Test a trained model on Land Cover Dataset")
#     parser.add_argument('--data_dir', type=str, default='./data/land_cover_representation', help='Path to dataset')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
#     parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
#     parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model weights')
#     parser.add_argument('--confusion_matrix', action='store_true', help='Plot the confusion matrix')
#     args = parser.parse_args()

#     device = get_device()
#     print(f"Using device: {device}")

#     transform = T.Compose([
#         T.Resize((100, 100)),
#         T.ToTensor()
#     ])

#     print('Loading test dataset...')
#     testset = LandClassDataset(args.data_dir, transform=transform, split='test')
#     testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     print('Test dataset successfully loaded')

#     print('Loading model...')
#     if args.model_path:
#         model = torchvision.models.resnet18(weights=None)
#         model.fc = nn.Linear(model.fc.in_features, testset.get_num_classes())
#         model = model.to(device)

#         checkpoint = torch.load(args.model_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"Model loaded from {args.model_path}")
#     else:
#         model = torchvision.models.resnet18(weights='DEFAULT')
#         model.fc = nn.Linear(model.fc.in_features, testset.get_num_classes())
#         model = model.to(device)
#         print(f'Default pretrained model loaded')

#     class_names = testset.get_class_names()

#     print("Starting testing...")
#     test_accuracy = test_model(args, model, testloader, class_names, device)

#     print(f"Test accuracy: {test_accuracy:.3f}")

# if __name__ == '__main__':
#     main()

def main():
    parser = ArgumentParser(description="Test a trained model on Land Cover Dataset")
    parser.add_argument("--data_dir", type=str, default="./data/land_cover_representation", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model weights")
    parser.add_argument("--confusion_matrix", action="store_true", help="Plot the confusion matrix")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((100, 100)),
        T.ToTensor(),
    ])

    print("Loading test dataset...")
    testset = LandClassDataset(args.data_dir, transform=transform, split="test")
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("Test dataset successfully loaded")

    print("Loading model...")
    if args.model_path:
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, testset.get_num_classes())
        model = model.to(device)

        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {args.model_path}")

        # Define output directory based on model_path
        output_dir = os.path.join(
            os.path.dirname(args.model_path), f"test_results_{os.path.dirname(args.model_path)}"
        )
    else:
        model = torchvision.models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, testset.get_num_classes())
        model = model.to(device)
        print(f"Default pretrained model loaded")

        # Default output directory
        output_dir = "./test_results"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    class_names = testset.get_class_names()

    print("Starting testing...")
    test_accuracy = test_model(args, model, testloader, class_names, device, output_dir)

    print(f"Test accuracy: {test_accuracy:.3f}")


if __name__ == "__main__":
    main()