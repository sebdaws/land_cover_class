import torch
import pandas as pd
from tqdm import tqdm

from utils.save import save_test_results
from utils.loss_functions import calculate_metrics
from utils.progress import DoubleProgressBar

def run_phase(args, model, dataloader, phase, criterion=None, optimizer=None, device=None, class_names=None, output_dir=None, epoch=None, best_val_accuracy=0.0, pbar=None):
    """
    Executes a single phase (train/validation/test) of the model pipeline.
    Handles forward/backward passes, metric calculation, and progress tracking.

    Parameters:
        args: Configuration object containing training parameters
        model (nn.Module): PyTorch model to train/evaluate
        dataloader (DataLoader): DataLoader for the current phase
        phase (str): One of ['train', 'val', 'test']
        criterion (nn.Module, optional): Loss function. Required for 'train' phase.
        optimizer (torch.optim.Optimizer, optional): Optimizer. Required for 'train' phase.
        device (torch.device, optional): Device to run computations on
        class_names (list[str], optional): Class names for test results output
        output_dir (Path, optional): Directory to save test results
        epoch (int, optional): Current epoch number for progress display
        best_val_accuracy (float, optional): Best validation accuracy so far
        pbar (ProgressBar, optional): Progress bar for updating training status

    Returns:
        For train/val phases:
            tuple: (model, metrics_dict) where metrics_dict contains phase metrics
        For test phase:
            None: Results are saved to output_dir

    Raises:
        ValueError: If criterion or optimizer are missing for train phase
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
    """
    Executes the complete training loop with validation.
    Tracks metrics, saves best model state, and displays progress.

    Parameters:
        args: Configuration object containing training parameters
        model (nn.Module): PyTorch model to train
        trainloader (DataLoader): DataLoader for training data
        valloader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run training on

    Returns:
        tuple: (
            best_model_state (OrderedDict): State dict of best model weights,
            metrics_df (DataFrame): DataFrame containing training/validation metrics per epoch,
            best_val_accuracy (float): Best validation accuracy achieved
        )
    """
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
    """
    Executes the test phase and saves results.
    Displays progress with a single progress bar.

    Parameters:
        args: Configuration object containing test parameters
        model (nn.Module): PyTorch model to evaluate
        testloader (DataLoader): DataLoader for test data
        criterion (nn.Module): Loss function
        class_names (list[str]): List of class names for result output
        device (torch.device): Device to run testing on
        output_dir (Path): Directory to save test results

    Returns:
        None: Results are saved to output_dir/test_results.csv
    """
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
    
