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
    
    Args:
        class_distribution (DataFrame): A DataFrame of class counts.
        title (str): The title of the plot.
        save_path (str): Optional path to save the plot.
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
    
    # Create figure with adjusted size and margins
    plt.figure(figsize=(16, 12))
    plt.tight_layout()
    
    # Create heatmap with rotated labels and adjusted font sizes
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="OrRd",
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.xlabel("Predicted", fontsize=12, labelpad=10)
    plt.ylabel("True", fontsize=12, labelpad=10)
    plt.title("Normalized Confusion Matrix", pad=20)
    
    # Adjust margins to prevent cutoff
    plt.tight_layout()

    # Save confusion matrix as an image
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path, bbox_inches='tight', dpi=300)
    print(f"Confusion matrix saved to {confusion_matrix_path}")
    plt.close()

def save_train(args, model, metrics_df, val_accuracy):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
        "training_date": timestamp
    }

    hyperparams_path = os.path.join(run_dir, "hyperparameters.json")

    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Hyperparameters saved to {hyperparams_path}")