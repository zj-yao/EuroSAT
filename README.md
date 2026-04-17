# EuroSAT MLP From Scratch

This project implements a three-layer MLP classifier for EuroSAT RGB images using only `NumPy` for tensor math. Automatic differentiation, backpropagation, SGD, learning-rate decay, model selection, hyperparameter search, testing, weight visualization, and error analysis are implemented inside the repository.

## Project Layout

```text
src/
  autograd.py   # Tensor graph and backprop
  dataset.py    # EuroSAT discovery, split, normalization, batching
  layers.py     # Linear layer and loss
  metrics.py    # Accuracy and confusion matrix
  model.py      # Three-layer MLP
  optim.py      # SGD and step LR scheduler
  search.py     # Random search entrypoint
  test.py       # Evaluation entrypoint
  train.py      # Training entrypoint
  utils.py      # Plotting, logging, serialization helpers
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Expected dataset layout:

```text
EuroSAT_RGB/
  AnnualCrop/
  Forest/
  HerbaceousVegetation/
  Highway/
  Industrial/
  Pasture/
  PermanentCrop/
  Residential/
  River/
  SeaLake/
```

## Train

Default training keeps the original `64x64` images. If CPU training is too slow, pass `--image_size 32`.

```bash
python -m src.train \
  --data_dir ./EuroSAT_RGB \
  --epochs 30 \
  --batch_size 128 \
  --lr 0.01 \
  --hidden_dims 128 64 \
  --activation relu \
  --weight_decay 1e-4
```

Useful CPU-friendly variant:

```bash
python -m src.train \
  --data_dir ./EuroSAT_RGB \
  --image_size 32 \
  --epochs 30 \
  --batch_size 128 \
  --hidden_dims 128 64
```

## Hyperparameter Search

```bash
python -m src.search \
  --data_dir ./EuroSAT_RGB \
  --trials 8 \
  --epochs 20
```

The search script samples from:

- learning rate: `0.1, 0.03, 0.01, 0.003`
- hidden dimensions: `(128, 64)` or `(256, 128)`
- weight decay: `0, 1e-4, 1e-3`
- activation: `relu` or `tanh`

By default it retrains the best configuration after search. Disable that with `--skip_retrain_best`.

## Test

```bash
python -m src.test \
  --data_dir ./EuroSAT_RGB \
  --checkpoint ./outputs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.npz
```

## Generated Artifacts

Each training run creates:

- `history.json`
- `history.csv`
- `plots/training_curves.png`
- `plots/first_layer_weights.png`
- `checkpoints/best_model.npz`

Each test run creates:

- `metrics.json`
- `plots/confusion_matrix.png`
- `plots/misclassified_examples.png`

## Notes

- No auto-diff framework is used.
- Checkpoints store model weights, normalization stats, class names, and the exact train/val/test splits so testing reproduces the same partition.
- For quick smoke tests, all scripts accept `--max_train_samples`, `--max_val_samples`, and `--max_test_samples`.

