import os
import datetime
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_class_distribution(class_distribution, title="Class Distribution with Percentages", save_path=None):
    """
    Plot the class distribution as a bar chart with percentages annotated on each bar.
    
    Creates a visualization of class distribution in the dataset, with bars showing
    counts and percentage labels for classes with more than 0.25% representation.
    
    Args:
        class_distribution (pd.DataFrame): A DataFrame containing class counts with
            columns ['class', 'count']
        title (str, optional): The title of the plot. Defaults to "Class Distribution 
            with Percentages"
        save_path (str, optional): Path to save the plot. If None, displays the plot
            instead. Defaults to None.
    """
    classes = list(class_distribution.index)
    counts = list(class_distribution['count'])
    
    total_samples = sum(counts)
    percentages = [count / total_samples * 100 for count in counts]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='skyblue')
    
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        if percentage >= 0.25:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 10, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(classes, rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def save_test_results(output_dir, class_names, all_labels, all_predictions):
    """
    Save classification test results including metrics and confusion matrix.
    
    Generates and saves:
    1. A CSV file with class-wise test metrics (precision, recall, F1-score)
    2. A normalized confusion matrix visualization
    
    Args:
        output_dir (str): Directory path to save the results
        class_names (list): List of class names in the dataset
        all_labels (array-like): Ground truth labels
        all_predictions (array-like): Model predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate confusion matrix for accuracy calculation
    cm = confusion_matrix(all_labels, all_predictions)
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    
    # Generate classification report
    report = classification_report(
        all_labels, all_predictions, target_names=class_names, output_dict=True
    )
    
    # Create DataFrame and add accuracy column
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.insert(0, 'accuracy', 0.0)  # Add accuracy column after index
    
    # Fill accuracy values for each class
    for i, class_name in enumerate(class_names):
        metrics_df.loc[class_name, 'accuracy'] = class_accuracy[i]
    
    # Remove the last three rows (micro avg, macro avg, weighted avg)
    metrics_df = metrics_df.iloc[:-3]
    
    # Save class-wise metrics to CSV
    metrics_csv_path = os.path.join(output_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=True, index_label='Class')
    print(f"Class-wise test metrics saved to {metrics_csv_path}")

    # Generate normalized confusion matrix visualization
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 12))
    plt.tight_layout()
    
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="OrRd",
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.xlabel("Predicted", fontsize=12, labelpad=10)
    plt.ylabel("True", fontsize=12, labelpad=10)
    plt.title("Normalized Confusion Matrix", pad=20)
    
    plt.tight_layout()

    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path, bbox_inches='tight', dpi=300)
    print(f"Confusion matrix saved to {confusion_matrix_path}")
    plt.close()

def save_train(args, model, metrics_df, val_accuracy, device, training_time):
    """
    Save training artifacts including model weights, metrics, and hyperparameters.
    
    Creates a timestamped directory and saves:
    1. Model weights (best achieved during training)
    2. Training metrics history
    3. Hyperparameters configuration
    
    Args:
        args: Arguments containing training parameters
        model: The best model state dict achieved during training
        metrics_df (pd.DataFrame): DataFrame containing training metrics history
        val_accuracy (float): Best validation accuracy
        device (torch.device): Device used for training
        training_time (float): Total training time in seconds
        epoch (int, optional): Current epoch if training was interrupted
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = os.path.join(args.save_dir, args.model_name)
    run_dir = os.path.join(model_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True) 

    save_dict = {
        'model_state_dict': model,
        'val_accuracy': val_accuracy,
        'num_epochs': args.num_epochs
    }

    model_weights_path = os.path.join(run_dir, "model_weights.pth")
    torch.save(save_dict, model_weights_path)
    print(f"\nModel weights saved to {model_weights_path}")

    if metrics_df is not None:
        metrics_path = os.path.join(run_dir, "training_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")

    hyperparams = {
        "model_name": args.model_name,
        "dataset": args.data_dir,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "use_over_sampler": args.over_sample,
        "loss_function": args.loss_func,
        "weights_smooth": args.weights_smooth,
        "use_infrared": args.use_infrared,
        "final_val_accuracy": val_accuracy,
        "training_date": timestamp,
        "training_device": str(device),
        "num_workers": args.num_workers,
        "training_time_seconds": training_time,
        "training_time_formatted": f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s"
    }

    hyperparams_path = os.path.join(run_dir, "hyperparameters.json")

    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=4)

    print(f"Hyperparameters saved to {hyperparams_path}")

