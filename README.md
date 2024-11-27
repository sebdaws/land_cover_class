# Land Cover Dataset Training

This repository contains a deep learning model training pipeline for land cover classification.

## Installation

Create a new conda environment and install requirements:
```bash
conda create -n landcover
conda activate landcover
pip install -r requirements.txt
```

### Download
The land cover dataset can be downloaded from [insert_data_source_link].

### Data Structure
After downloading, place the data in the `./data` directory with the following structure:
```
data/
└── land_cover_representation/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### Data Preprocessing

#### Class Balancing
The dataset includes classes with varying sample sizes. Use the class balancing script to group low-count classes and create balanced train/val/test splits:
```bash
python scripts/balance_classes.py \
    --metadata_path ../data/land_cover_representation/metadata.csv \
    --save_path ../data/land_cover_representation/metadata_balanced.csv \
    --min_count 2000 \
    --train_split 0.8 \
    --test_split 0.1
```

Arguments for balance_classes.py:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--metadata_path` | str | '../data/land_cover_representation/metadata.csv' | Path to original metadata file |
| `--save_path` | str | '../data/land_cover_representation/metadata_balanced.csv' | Path to save the balanced metadata file |
| `--min_count` | int | 2000 | Classes with fewer samples than this will be grouped into "Other" |
| `--train_split` | float | 0.8 | Proportion of data for training set |
| `--test_split` | float | 0.1 | Proportion of data for test set (validation gets the remainder) |
| `--seed` | int | 42 | Random seed for reproducibility |

The script will:
1. Read the original metadata file
2. Group classes with fewer than `min_count` samples into an "Other" category
3. Create stratified train/validation/test splits
4. Save the new balanced metadata file


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
    --model_name efficientnet_b0 \
    --batch_size 64 \
    --num_epochs 10 \
    --lr 0.001 \
    --loss_func kl_div \
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
