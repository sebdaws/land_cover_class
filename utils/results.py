import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

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
    plt.figure(figsize=(16, 10))
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