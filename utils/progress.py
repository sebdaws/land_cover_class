from tqdm import tqdm

class DoubleProgressBar:
    """
    Creates and manages two progress bars for nested training loops.
    
    This class implements a double progress bar system where one bar shows total
    training progress across all epochs, and another shows progress within the
    current epoch. Useful for providing visual feedback during model training.

    Args:
        total_iterations (int): Total number of iterations across all epochs
        total_batches (int): Number of batches in each epoch
    """
    def __init__(self, total_iterations, total_batches):
        """Initialize double progress bar with total iterations and batches."""
        self.total_bar = tqdm(
            total=total_iterations, 
            desc="Total Progress", 
            position=0, 
            leave=True,
            initial=0,
            postfix={'Best Val Acc': '0.000'}
        )
        self.epoch_bar = tqdm(
            total=total_batches, 
            desc="Epoch Progress", 
            position=1, 
            leave=False
        )
        
    def update_total(self, n=1, postfix=None):
        """
        Update the total (overall) progress bar.

        Args:
            n (int, optional): Number of steps to increment. Defaults to 1.
            postfix (dict, optional): Dictionary of additional stats to display.
                                    Typically used for metrics like accuracy.
        """
        self.total_bar.update(n)
        if postfix:
            self.total_bar.set_postfix(postfix)
        
    def update_epoch(self, phase, epoch, args, n=1, postfix=None):
        """
        Update the epoch progress bar.

        Args:
            phase (str): Current phase ('train' or 'val')
            epoch (int): Current epoch number (0-based)
            args: Arguments containing num_epochs
            n (int, optional): Number of steps to increment. Defaults to 1.
            postfix (dict, optional): Dictionary of additional stats to display.
                                    Typically used for batch-wise metrics.
        """
        self.epoch_bar.set_description(f"{phase.capitalize()} Epoch [{epoch + 1}/{args.num_epochs}]")
        self.epoch_bar.update(n)
        if postfix:
            self.epoch_bar.set_postfix(postfix)
            
    def reset_epoch_bar(self, total_batches):
        """
        Reset the epoch bar with new total.

        Args:
            total_batches (int): New total number of batches for the epoch bar
        """
        self.epoch_bar.close()
        self.epoch_bar = tqdm(
            total=total_batches,
            desc="Epoch Progress",
            position=1,
            leave=False
        )
            
    def close(self):
        """Close both progress bars."""
        self.epoch_bar.close()
        self.total_bar.close()