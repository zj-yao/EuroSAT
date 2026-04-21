# EuroSAT MLP From Scratch

This project implements a three-layer MLP classifier for EuroSAT RGB images using only `NumPy` for tensor math. Automatic differentiation, backpropagation, SGD with momentum, learning-rate decay, model selection, hyperparameter search, data augmentation, testing, weight visualization, and error analysis are implemented inside the repository.

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

Recommended final training configuration:

```bash
python -m src.train \
  --data_dir ./EuroSAT_RGB \
  --epochs 60 \
  --batch_size 128 \
  --lr 0.005 \
  --momentum 0.9 \
  --weight_decay 1e-3 \
  --step_size 10 \
  --gamma 0.5 \
  --hidden_dims 512 256 \
  --activation relu \
  --image_size 48 \
  --augment \
  --min_crop_scale 0.85 \
  --brightness_jitter 0.1 \
  --run_name improved_aug48_momentum_v2_60ep
```

Baseline CPU-friendly variant:

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

## Best Result

- Best validation accuracy: `72.40%`
- Test accuracy: `73.46%`
- Test loss: `0.7372`
- Final checkpoint (uploaded separately): `outputs/improved_aug48_momentum_v2_60ep/checkpoints/best_model.npz`

## Test

```bash
python -m src.test \
  --data_dir ./EuroSAT_RGB \
  --checkpoint ./outputs/improved_aug48_momentum_v2_60ep/checkpoints/best_model.npz \
  --run_name improved_aug48_momentum_v2_60ep_test
```

## Submission Links

- Public GitHub Repo: https://github.com/zj-yao/EuroSAT
- Model weights: https://drive.google.com/file/d/12x7tM2sZAQbbmJRM7hQVw4cYtuewwr1B/view?usp=drive_link

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
