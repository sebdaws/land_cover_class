I'll help you create a README.md file that explains how to use your Land Cover Dataset training script based on the command-line arguments.

```markdown
# Land Cover Dataset Training

This repository contains a deep learning model training pipeline for land cover classification.

## Requirements

- Python 3.x
- PyTorch
- pandas
- Other dependencies (list them in requirements.txt)

## Usage

The main script can be run in two phases: training and testing.

### Basic Commands

Train a model:
```bash
python main.py --phase train
```

Test a trained model:
```bash
python main.py --phase test --model_path path/to/model/weights
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--phase` | str | Required | Phase to run: 'train' or 'test' |
| `--data_dir` | str | './data/land_cover_representation' | Path to dataset |
| `--model_name` | str | 'resnet18' | Model architecture to use |
| `--batch_size` | int | 32 | Batch size for training and validation |
| `--num_epochs` | int | 20 | Number of training epochs |
| `--lr` | float | 0.0001 | Learning rate for training |
| `--num_workers` | int | 0 | Number of workers for data loading |
| `--save_dir` | str | './experiments' | Directory to save trained models |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--print_iter` | int | 1000 | Print training updates every N iterations |
| `--loss_func` | str | 'cross_entropy' | Loss function for training (options: cross_entropy, weighted_cross_entropy, focal, dice, kl_div) |
| `--weights_smooth` | float | 0 | Smoothing factor for class weights |
| `--over_sample` | flag | False | Enable oversampling of minority classes |
| `--model_path` | str | None | Path to pre-trained model weights (for testing) |
| `--confusion_matrix` | flag | False | Generate confusion matrix during testing |

### Example Commands

Train a model with custom parameters:
```bash
python main.py \
    --phase train \
    --model_name resnet50 \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 0.001 \
    --loss_func focal \
    --over_sample
```

Test a model and generate confusion matrix:
```bash
python main.py \
    --phase test \
    --model_path ./experiments/best_model.pth \
    --confusion_matrix
```

## Output

- Trained models are saved in the specified `save_dir`
- Training metrics and logs are stored alongside the model
- Test results and confusion matrices (if requested) are generated in the output directory

## Notes

- Use `--over_sample` flag to handle class imbalance
- Different loss functions are available for handling various training scenarios
- Set appropriate `--num_workers` based on your system capabilities
```

This README provides a comprehensive overview of your script's functionality and how to use it. You might want to add:

1. A more detailed description of your dataset structure
2. Installation instructions
3. Any preprocessing steps required
4. Description of the output format
5. Examples of expected results

Let me know if you'd like me to expand any section!