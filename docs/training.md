# Training

```python
from deepvoid.train import run, run_seeds, evaluate_run

out = run("configs/tng_dm_baseline.yaml")                       # one run
summary = run_seeds("configs/tng_dm_baseline.yaml", [1, 2, 3])  # mean and std over seeds
evaluate_run(out)                                               # re-score a finished run
```

From a shell: `python -m deepvoid.train configs/tng_dm_baseline.yaml [--seeds 1 2 3]`.

## Settings

A config only needs `name`, `data.inputs` and `data.labels`; everything else
has a default (`src/deepvoid/config.py`). Misspelled settings raise an error
instead of being ignored. The full config used, defaults included, is saved
as `config.yaml` in the run folder.

| Setting | Default | What it does |
|---|---|---|
| `name` | required | Run folder name, under `output` |
| `output` | `runs` | Parent folder for runs (use a Drive path on Colab) |
| `seed` | 12 | Seed for weights and for the crops drawn each step |
| `data.inputs` | required | Input datasets from the catalog; several share one label set |
| `data.labels` | required | Mask dataset from the catalog |
| `data.split` | 0.7/0.15/0.15, buffer 16, axis 0 | Train, validation and test slabs |
| `model.depth`, `model.base_filters` | 4, 16 | U-Net size |
| `model.batch_norm`, `model.bn_momentum` | true, 0.99 | Batch normalization |
| `model.dropout` | 0 | Dropout at the bottom of the U-Net |
| `training.crop` | 96 | Training crop side in cells |
| `training.batch_size` | 8 | Crops per step |
| `training.steps_per_epoch` | 200 | Steps between validation passes |
| `training.val_batches` | 32 | Fixed validation crops, in batches |
| `training.epochs` | 150 | Upper limit; early stopping usually ends sooner |
| `training.rotate` | true | Random cube symmetry per batch (48 rotations and reflections) |
| `training.optimizer` | AdamW, lr 1e-3, weight decay 1e-4, clipnorm 1.0 | `name` is `adamw` or `adam`; `clipnorm` clips the gradient norm |
| `training.schedule` | cosine, 3 warmup epochs, min 1% of lr | `cosine` (linear warmup, then cosine decay) or `constant` |
| `training.ema` | 0.999 | Decay of the weight average; `null` turns it off |
| `training.loss.name` | `scce` | `scce` or `dice_scce` (cross-entropy plus Dice) |
| `training.loss.class_weights` | null | `null`, `auto`, or a list of 4 weights |
| `training.loss.dice_weight` | 1.0 | Weight of the Dice term in `dice_scce` |
| `training.early_stopping` | `val_mcc`, patience 15 | Score to track and epochs to wait without improvement |
| `training.mixed_precision` | true | float16 compute, float32 weights |
| `training.xla` | true | Compile the training step with XLA |
| `training.resume` | true | Continue an interrupted run in the same folder |
| `evaluate.crop`, `evaluate.margin` | 128, 16 | Crop size and discarded edge for full-box prediction |
| `evaluate.tta` | `none` | Test-time augmentation: `none`, `flips` (8x) or `full` (48x) |

## What happens in a run

1. The labels are split into slabs, and input normalization is fit on the
   training slab.
2. With `class_weights: auto`, weights proportional to f^(-1/2) are computed
   from the training slab's class fractions f and scaled so the average
   voxel weight is 1.
3. Each step draws `batch_size` crops from the training slab on the GPU and
   applies one random cube symmetry to the batch. The crops for a given seed,
   epoch and step are always the same.
4. After each step the weight average (EMA) is updated.
5. After each epoch the averaged weights score the fixed validation crops
   (loss, accuracy, MCC, void F1, halo F1). The best epoch by the monitored
   score is saved as `best.keras`, and training stops after `patience`
   epochs without improvement.
6. At the end, `best.keras` predicts the full box and is scored on the test
   slab, written to `test_scores.json`.

The learning rate follows its schedule whether or not training stops early.
The schedule is set when a run starts, so changing `training.epochs` when
resuming changes when training ends but not the schedule.

## Watching a run

- **`log.csv`** in the run folder gets a row per epoch with the training loss
  and accuracy, the validation scores and the learning rate. The whole file
  is rewritten each epoch, so the copy on Google Drive stays current.
- **TensorBoard** shows the same numbers as curves, plus a validation slice
  each epoch with the truth above and the prediction below. Logs are written
  to local disk and copied to `tensorboard/` in the run folder each epoch.
  In Colab, start it before training:
  `%tensorboard --logdir /tmp/deepvoid_tensorboard`.
- The cell output prints one line per epoch with the monitored score and the
  best so far.

On track looks like validation MCC rising over the first 10 to 20 epochs and
the predicted slice coming to resemble the truth.

## Resuming

Every epoch saves `last.keras` (weights and optimizer state), `ema.npz` and
`state.json` (epoch, best score, epochs without improvement). If a run is
interrupted (a Colab timeout, a disconnect, the computer sleeping), run the
same config again: it picks up after the last completed epoch, with the same
learning-rate schedule and the same sequence of crops. At most the epoch in
progress is lost.

Resuming refuses to continue if any setting other than `training.epochs` or
`evaluate` has changed; start a new run with a new name instead.

## Several seeds

`run_seeds(config, [1, 2, 3])` trains one run per seed, in folders
`<name>_seed1`, `<name>_seed2`, ..., and returns the mean and standard
deviation of each test score. Differences between configs smaller than the
seed-to-seed spread are not meaningful. Each seed's run resumes on its own,
so the whole call can simply be rerun after an interruption.
