import tqdm

class DoubleProgressBar:
    def __init__(self, total_iterations, total_batches):
        """Initialize double progress bar with total iterations and batches."""
        self.total_bar = tqdm(
            total=total_iterations, 
            desc="Total Progress", 
            position=0, 
            leave=True,
            initial=0,
            postfix={'Best Val Acc': '0.000'}  # Initialize with 0
        )
        self.epoch_bar = tqdm(
            total=total_batches, 
            desc="Epoch Progress", 
            position=1, 
            leave=False
        )
        
    def update_total(self, n=1, postfix=None):
        """Update the total (overall) progress bar."""
        self.total_bar.update(n)
        if postfix:
            self.total_bar.set_postfix(postfix)
        
    def update_epoch(self, phase, epoch, args, n=1, postfix=None):
        """Update the epoch progress bar."""
        self.epoch_bar.set_description(f"{phase.capitalize()} Epoch [{epoch + 1}/{args.num_epochs}]")
        self.epoch_bar.update(n)
        if postfix:
            self.epoch_bar.set_postfix(postfix)
            
    def reset_epoch_bar(self, total_batches):
        """Reset the epoch bar with new total."""
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