import matplotlib.pyplot as plt

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
    
    total_samples = sum(counts)  # Total samples in the dataset
    percentages = [count / total_samples * 100 for count in counts]  # Calculate percentage for each class
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='skyblue')
    
    # Annotate each bar with percentage
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        if percentage >= 0.25:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 10, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(classes, rotation=90)  # Rotate x-axis labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()