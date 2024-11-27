from argparse import ArgumentParser
import pandas as pd

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
    args = parser.parse_args()

    if args.phase == 'train':
        trainloader, valloader, model, criterion, optimizer, device, run_dir = train_setup(args)

        best_model, metrics_df, val_accuracy = train(
            args=args, 
            model=model, 
            trainloader=trainloader, 
            valloader=valloader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device
        )
        
        save_train(args, best_model, metrics_df, val_accuracy)


    if args.phase == 'test':
        testloader, model, class_names, output_dir, device = test_setup(args)
        test(args, model, testloader, class_names, device, output_dir)

if __name__ == "__main__":
    main()
