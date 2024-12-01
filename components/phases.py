import torch
import pandas as pd
from tqdm import tqdm

from utils.save import save_test_results
from utils.loss_functions import calculate_metrics
from utils.progress import DoubleProgressBar

def run_phase(args, model, dataloader, phase, criterion=None, optimizer=None, device=None, class_names=None, output_dir=None, epoch=None, best_val_accuracy=0.0, pbar=None):
    """
    General function to handle training, validation, or testing phases.

    Args:
        args: Arguments with configurations like epochs or print frequency.
        model: The model to evaluate or train.
        dataloader: DataLoader for the current phase.
        criterion: Loss function
        optimizer: Optimizer (required for 'train').
        device: Device for computation (e.g., 'cuda' or 'cpu').
        class_names: List of class names (optional, for test phase).
        output_dir: Directory to save test results (optional, for test phase).
        epoch: Current epoch number (optional, for train phase).
        best_val_accuracy: Best validation accuracy achieved so far (optional, for train phase).

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

        if pbar is not None:
            if phase == 'test':
                pbar.update(1)
                pbar.set_postfix({
                    'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                    'Accuracy': f'{correct / total:.3f}'
                })
            else:
                pbar.update_epoch(
                    phase=phase,
                    epoch=epoch,
                    args=args,
                    postfix={
                        'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                        'Accuracy': f'{correct / total:.3f}'
                    }
                )
                pbar.update_total()
    
    if phase == 'test':
        pbar.close()
        save_test_results(output_dir, class_names, all_labels, all_predictions)
        return

    phase_metrics = calculate_metrics(all_labels, all_predictions)
    phase_metrics['Accuracy'] = correct / total
    phase_metrics['Loss'] = running_loss / len(dataloader)
    metrics = {f"{phase}_{key}": value for key, value in phase_metrics.items()}

    return model, metrics

def train(args, model, trainloader, valloader, criterion, optimizer, device):
    metrics_df = pd.DataFrame()
    best_val_accuracy = 0.0

    # Calculate total number of iterations for the overall progress bar
    total_iterations = args.num_epochs * (len(trainloader) + len(valloader))

    # Initialize double progress bar
    pbar = DoubleProgressBar(
        total_iterations=total_iterations,
        total_batches=len(trainloader)
    )

    for epoch in range(args.num_epochs):
        pbar.reset_epoch_bar(len(trainloader))
        model, epoch_train_metrics = run_phase(
            args=args,
            model=model, 
            dataloader=trainloader,
            phase='train',
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            best_val_accuracy=best_val_accuracy,
            pbar=pbar)
        
        pbar.reset_epoch_bar(len(valloader))
        model, epoch_val_metrics = run_phase(
            args=args,
            model=model,
            dataloader=valloader, 
            phase='val',
            criterion=criterion,
            device=device,
            epoch=epoch,
            best_val_accuracy=best_val_accuracy,
            pbar=pbar)

        if epoch_val_metrics['val_Accuracy'] > best_val_accuracy:
            best_val_accuracy = epoch_val_metrics['val_Accuracy']
            best_model_state = model.state_dict()

        pbar.update_total(postfix={'Best Val Acc': f'{best_val_accuracy:.3f}'})

        combined_metrics = {**epoch_train_metrics, **epoch_val_metrics}
        epoch_df = pd.DataFrame([combined_metrics])
        metrics_df = pd.concat([metrics_df, epoch_df], ignore_index=True)

    pbar.close()

    return best_model_state, metrics_df, best_val_accuracy

def test(args, model, testloader, criterion, class_names, device, output_dir):
    """Run test phase with a single progress bar."""
    # Initialize test progress bar
    test_bar = tqdm(
        total=len(testloader),
        desc="Testing Progress",
        position=0,
        leave=True,
        postfix={'Loss': '0.000', 'Accuracy': '0.000'}
    )

    run_phase(
        args=args, 
        model=model, 
        dataloader=testloader, 
        phase='test', 
        criterion=criterion,
        device=device, 
        class_names=class_names, 
        output_dir=output_dir,
        pbar=test_bar  # Pass the progress bar to run_phase
    )

    return
    
