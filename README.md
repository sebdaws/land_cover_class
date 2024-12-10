# Land Cover Dataset Training

This repository contains a deep learning model training pipeline for land cover classification.

## Dataset Overview

The land cover classification dataset used in this project is derived from the work of Jean et al. [1]. It covers a 2500 square kilometer area of Central Valley, CA, USA, and consists of NAIP (National Agriculture Imagery Program) aerial imagery with 4 spectral bands (R,G,B,IR) at 0.6m resolution. The dataset includes 61 land cover classes.

## Installation

Create a new conda environment and install requirements:
```bash
conda create -n landcover python=3.10
conda activate landcover
```
To install pytorch with CUDA support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
Otherwise:
```bash
conda install pytorch torchvision
```
Install the rest of the requirements:
```
pip install -r requirements.txt
```

## Dataset

This project uses the land cover classification dataset from Jean et al. [1], which covers a 2500 square kilometer area of Central Valley, CA, USA. The dataset consists of NAIP (National Agriculture Imagery Program) aerial imagery with 4 spectral bands (R,G,B,IR) at 0.6m resolution with 61 land cover classes.

### Download
The dataset can be downloaded from [[this link](https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg15/land_cover_representation.html#download)]. Download the 'land_cover_representation.zip' file and create a `./data` folder in the root directory in which to unzip it. This should result in the following structure:
```
data/
└── land_cover_representation/
    ├── tiles/
    ├── metadata.csv/

```

#### Class Balancing
The dataset includes classes with varying sample sizes. The 'balance_classes.py' script groups low-count classes and creates balanced train/val/test splits:
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
4. Save the new `metadata_balanced.csv` file which is the default file used in training and testing


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
| `--use_infrared` | flag | False | Use infrared bands in addition to RGB |
| `--resume_from` | str | None | Path to checkpoint to resume training from |

### Supported Model Architectures

The training pipeline supports several common CNN architectures through torchvision:

| Architecture | Description | Parameters | Input Size |
|--------------|-------------|------------|------------|
| `resnet18` | Lightweight ResNet variant, good for initial experiments | 11.7M | 224x224 |
| `resnet34` | Medium ResNet variant with more capacity | 21.8M | 224x224 |
| `resnet50` | Popular ResNet variant, good balance of speed/accuracy | 25.6M | 224x224 |
| `resnet101` | Deeper ResNet with high capacity | 44.5M | 224x224 |
| `resnet152` | Deepest ResNet variant with maximum capacity | 60.2M | 224x224 |
| `efficientnet_b0` | Smallest EfficientNet, very efficient | 5.3M | 224x224 |
| `efficientnet_b1` | Slightly larger than b0, more accuracy | 7.8M | 240x240 |
| `efficientnet_b2` | Good balance for medium datasets | 9.2M | 260x260 |
| `efficientnet_b3` | Higher accuracy, still efficient | 12.0M | 300x300 |
| `efficientnet_b4` | Large model with strong performance | 19.0M | 380x380 |
| `efficientnet_b5` | Very large model for complex tasks | 30.0M | 456x456 |
| `efficientnet_b6` | Higher capacity for challenging datasets | 43.0M | 528x528 |
| `efficientnet_b7` | Maximum capacity EfficientNet | 66.0M | 600x600 |

Select the model architecture using the `--model_name` argument. For example:
```bash
python main.py --phase train --model_name efficientnet_b0
```

Notes on model selection:
- ResNet18/34 and EfficientNet-B0/B1 are good starting points for initial experiments
- ResNet50 and EfficientNet-B2/B3 offer good balance of speed and accuracy
- Larger models (ResNet101/152, EfficientNet-B4+) require more GPU memory and training time

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

## Data Limitations

It's important to note that the dataset labels are based on the mode of the class within each tile. This means that the most frequent class within a tile is used as the label for that tile. As a result, there is some inherent inaccuracy in the data due to:
- Labelling errors: The dataset contains some incorrect labels due to human error in the annotation process
- Mixed pixels: A single tile may contain multiple land cover types, but only the most frequent type is labeled.
- Boundary effects: Tiles on the boundary of different land cover types may be misclassified.

These factors should be considered when:
- Interpreting model performance metrics
- Setting expectations for accuracy
- Analyzing prediction errors


## References

[1] N. Jean, S. Wang, A. Samar, G. Azzari, D. Lobell, and S. Ermon. Tile2Vec: Unsupervised representation learning for spatially distributed data. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):3967–3974, Jul. 2019.
