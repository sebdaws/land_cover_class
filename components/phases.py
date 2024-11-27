import torch
import pandas as pd

from utils.save import save_test_results
from utils.loss_functions import calculate_metrics


def run_phase(args, model, dataloader, phase, criterion=None, optimizer=None, device=None, class_names=None, output_dir=None):
    """
    General function to handle training, validation, or testing phases.

    Args:
        args: Arguments with configurations like epochs or print frequency.
        model: The model to evaluate or train.
        dataloader: DataLoader for the current phase.
        criterion: Loss function (required for 'train' and 'val').
        optimizer: Optimizer (required for 'train').
        device: Device for computation (e.g., 'cuda' or 'cpu').
        class_names: List of class names (optional, for test phase).
        output_dir: Directory to save test results (optional, for test phase).

    Returns:
        metrics: Dictionary of computed metrics.
    """
    if phase == 'train' and (criterion is None or optimizer is None):
        raise ValueError("Criterion and optimizer must be provided for the 'train' phase.")

    model.train() if phase == 'train' else model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        if phase == 'train':
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            if phase == 'train':
                loss.backward()
                optimizer.step()

        if phase == 'train' and (batch_idx + 1) % args.print_iter == 0:
            print(f"Batch [{batch_idx + 1}/{len(dataloader)}], "
                  f"Loss: {running_loss / (batch_idx + 1):.4f}, "
                  f"Accuracy: {correct / total:.3f}")

    # Calculate metrics for this phase
    phase_metrics = calculate_metrics(all_labels, all_predictions)
    phase_metrics['Accuracy'] = correct / total
    phase_metrics['Loss'] = running_loss / len(dataloader)

    # Add phase prefix to each metric and update the metrics dictionary
    metrics = {}
    for key, value in phase_metrics.items():
        metrics[f"{phase}_{key}"] = value
    
    print(f"{phase.capitalize()} Metrics: {metrics}")

    if args.phase == 'test' and class_names and output_dir:
        save_test_results(output_dir, class_names, all_labels, all_predictions)

    return model, metrics

def train(args, model, trainloader, valloader, criterion, optimizer, device):
    metrics_df = pd.DataFrame()
    best_val_accuracy = 0.0

    for epoch in range(args.num_epochs):
        print(f"Epoch [{epoch+1}/{args.num_epochs}]")
        model, epoch_train_metrics = run_phase(
            args=args,
            model=model, 
            dataloader=trainloader,
            phase='train',
            criterion=criterion,
            optimizer=optimizer,
            device=device)
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Validation...")
        model, epoch_val_metrics = run_phase(
            args=args,
            model=model,
            dataloader=valloader, 
            phase='val',
            criterion=criterion,
            device=device)

        combined_metrics = {**epoch_train_metrics, **epoch_val_metrics}
        epoch_df = pd.DataFrame([combined_metrics])
        metrics_df = pd.concat([metrics_df, epoch_df], ignore_index=True)
        
        if epoch_val_metrics['val_Accuracy'] > best_val_accuracy:
            best_val_accuracy = epoch_val_metrics['val_Accuracy']
            best_model_state = model.state_dict()

    return best_model_state, metrics_df, best_val_accuracy

def test(args, model, testloader, class_names, device, output_dir):
    _, _ = run_phase(
        args=args, 
        model=model, 
        dataloader=testloader, 
        phase='test', 
        device=device, 
        class_names=class_names, 
        output_dir=output_dir)
    
