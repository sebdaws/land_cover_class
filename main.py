from argparse import ArgumentParser
import pandas as pd
import time
import os

from components.setup import train_setup, test_setup
from components.phases import train, test
from utils.save import save_train

def main():
    parser = ArgumentParser(description="Train a model on Land Cover Dataset")
    parser.add_argument('--phase', type=str, required=True, choices=['train', 'test'], help='Phase to run')
    parser.add_argument('--data_dir', type=str, default='./data/land_cover_representation', help='Path to dataset')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate used for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--save_dir', type=str, default='./experiments', help='Directory to save the trained model')
    parser.add_argument('--seed', type=int, default=42, help='Set randomness seed')
    parser.add_argument('--print_iter', type=int, default=1000, help='Set number of iterations between printing updates in training')
    parser.add_argument('--loss_func', type=str, default='cross_entropy', choices=['cross_entropy', 'weighted_cross_entropy', 'focal', 'dice', 'kl_div'], help='Loss function to use for training.')
    parser.add_argument('--weights_smooth', type=float, default=0, help='Amount added to smooth class weights')
    parser.add_argument('--over_sample', action='store_true', help='Over-sample minority classes')
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model weights")
    parser.add_argument("--confusion_matrix", action="store_true", help="Plot the confusion matrix")
    parser.add_argument("--use_infrared", action="store_true", help="Use infrared bands")
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    if args.phase == 'train':
        start_time = time.time()
        
        trainloader, valloader, model, criterion, optimizer, device, start_epoch, metrics_df = train_setup(args)

        args.num_epochs = args.num_epochs + start_epoch

        best_model, metrics_df, val_accuracy = train(
            args=args, 
            model=model, 
            trainloader=trainloader, 
            valloader=valloader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device,
            start_epoch=start_epoch,
            metrics_df=metrics_df
        )
        
        training_time = time.time() - start_time
        save_train(args, best_model, metrics_df, val_accuracy, device, training_time)


    if args.phase == 'test':
        if not args.model_path:
            raise ValueError("A model path must be provided for testing.")
        
        testloader, model, criterion, class_names, output_dir, device = test_setup(args)
        metrics = test(
            args=args, 
            model=model, 
            testloader=testloader, 
            criterion=criterion, 
            class_names=class_names, 
            device=device, 
            output_dir=output_dir
        )
        
        summary_path = os.path.join(output_dir, "test_summary.csv")
        pd.DataFrame([metrics]).to_csv(summary_path, index=False)
        
        print("\nTest Results Summary")
        print("=" * 50)
        max_key_length = max(len(key[5:]) for key in metrics.keys())
        for metric, value in metrics.items():
            print(f"  {metric[5:]}:{' ' * (max_key_length - len(metric[5:]) + 4)}{value:.4f}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
