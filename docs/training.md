# Training

```python
from deepvoid.train import run, run_seeds, evaluate_run, spacing_sweep

out = run("configs/tng_dm_baseline.yaml")                       # one run
summary = run_seeds("configs/tng_dm_baseline.yaml", [1, 2, 3])  # mean and std over seeds
evaluate_run(out)                                               # re-score a finished run
spacing_sweep(out, "tng300-3/subhalos/L10", [3, 5, 7, 10])     # FiLM check, see below
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
| `model.conditioning` | null | Condition on tracer spacing (FiLM), e.g. `{units: 32, noise: 0.1}`; see below |
| `training.crop` | 96 | Training crop side in cells |
| `training.batch_size` | 8 | Crops per step |
| `training.steps_per_epoch` | 200 | Steps between validation passes |
| `training.val_crop` | 64 | Validation crop side; must fit inside the validation slab (70 cells for a 512^3 box with the default split) |
| `training.val_batches` | 32 | Fixed validation crops per input, in batches |
| `training.epochs` | 80 | Length of the learning-rate schedule and upper limit on training |
| `training.rotate` | true | Random cube symmetry per batch (48 rotations and reflections) |
| `training.init_from` | null | Start from a trained model: a `.keras` file or a run folder (its `best.keras`), relative to `output` |
| `training.curriculum` | null | Sampling weights over inputs that change with the epoch; see below |
| `training.optimizer` | AdamW, lr 1e-3, weight decay 1e-4, clipnorm 1.0 | `name` is `adamw` or `adam`; `clipnorm` clips the gradient norm |
| `training.schedule` | cosine, 3 warmup epochs, min 1% of lr | `cosine` (linear warmup, then cosine decay) or `constant` |
| `training.ema` | 0.999 | Decay of the weight average; `null` turns it off |
| `training.loss.name` | `scce` | `scce` or `dice_scce` (cross-entropy plus Dice) |
| `training.loss.class_weights` | null | `null`, `auto`, or a list of 4 weights |
| `training.loss.dice_weight` | 1.0 | Weight of the Dice term in `dice_scce` |
| `training.early_stopping` | `val_mcc`, patience 15, min_delta 0.002 | Score to track, and epochs to wait without a gain of at least `min_delta` |
| `training.mixed_precision` | true | float16 compute, float32 weights |
| `training.xla` | true | Compile the training step with XLA |
| `training.resume` | true | Continue an interrupted run in the same folder |
| `evaluate.crop`, `evaluate.margin` | 128, 16 | Crop size and discarded edge for full-box prediction |
| `evaluate.tta` | `none` | Test-time augmentation: `none`, `flips` (8x) or `full` (48x) |
| `evaluate.inputs` | null | Inputs to score on the test slab; null means the training inputs |
| `evaluate.labels` | null | Mask on a finer grid to score against; predictions are upsampled to it (see below) |

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
   (loss, accuracy, MCC, void F1, halo F1). These are 64^3 by default,
   smaller than the training crops, because the validation slab is thinner
   than the training slab. The best epoch by the monitored
   score is saved as `best.keras` (any gain counts). Training stops after
   `patience` epochs in which the score has not beaten its last reference
   value by at least `min_delta`, so a score creeping up by tiny amounts
   does not keep a run going.
6. At the end, `best.keras` predicts the full box and is scored on the test
   slab, written to `test_scores.json`.

The learning rate follows its schedule whether or not training stops early,
so `training.epochs` should be close to how long a run actually needs: if
early stopping ends a run well before `epochs`, the learning rate never got
low. The first subhalo run (spacing 10, from scratch) peaked at epoch 11
and stopped at 26 with the learning rate still at 94% of its maximum,
because `epochs` was 150. The dark-matter baseline ran 106 epochs of 150.
The default is now 80, and the mixed subhalo runs use 60.
The schedule is set when a run starts, so changing `training.epochs` when
resuming changes when training ends but not the schedule.

## Several inputs, spacing and curriculum

`data.inputs` can list several fields that share one label set, such as
subhalo fields at tracer spacings 3, 5, 7 and 10 h^-1 Mpc
(`configs/subhalo/`). Each crop in a training batch comes from its own
randomly chosen input, with equal probability unless a curriculum says
otherwise. Mixing within the batch matters with batch normalization:
when every batch came from a single input, training looked fine but the
running averages used at inference matched none of the inputs, and
validation fell apart (the first `C_mixed` run). Each input gets
its own normalization, fit on the training slab. Validation crops are drawn
from every input, and with more than one input the log also has
`val_mcc_<label>` for each (`val_mcc_L3`, ...). `val_mcc` itself is pooled
over all inputs. `test_scores.json` has one entry per input scored.

**Starting from a trained model.** `training.init_from` copies weights from
a saved model, layer by layer, matching names and shapes; the run log says
how many layers were copied. Every layer stays trainable. Use a lower
learning rate than from scratch (`configs/subhalo/B_finetune_L10.yaml` uses
2e-4 with one warmup epoch).

**Spacing conditioning (FiLM).** With `model.conditioning`, the model also
takes the tracer spacing lambda = n^(-1/3), read from each input's
`spacing` in the catalog. log(lambda) goes through a two-layer network
with `units` outputs, and after every normalization layer the features are
scaled and shifted by amounts computed from it (`models/film.py`). Those
layers start at zero, so the conditioned model starts out identical to the
plain one, and a plain model's weights can be loaded into it with
`init_from`. During training lambda is multiplied by exp(N(0, `noise`)),
so the model cannot rely on its exact value.

To check the model uses lambda, `spacing_sweep(out, name, spacings)`
scores one input while telling the model each of the given spacings, and
writes `spacing_sweep_<label>.json`. If the scores barely change when the
spacing is wrong, the conditioning is doing nothing.

**Curriculum.** `training.curriculum` changes how often each input is drawn:

```yaml
curriculum: {start: [4, 3, 2, 1], end: [1, 1, 1, 1], epochs: 30}
```

The weights (one per input, in the order of `data.inputs`) move linearly
from `start` to `end` over `epochs` epochs and stay at `end` after that.
All weights must be positive, so every input is seen from the first epoch
on; only the mix changes. Training on one input at a time tends to forget
the earlier ones, which this avoids.

## Training on a coarser grid

A model can train on a coarser grid, so a crop of the same cell count
covers more of the box, and still be scored at full resolution.
`scripts/make_coarse.py` (notebook 07) writes 256^3 inputs and a 256^3
mask recomputed from the dark matter with the same sigma_eff. A config
then sets `evaluate.labels` to the 512^3 mask: each 256^3 prediction fills
its 2x2x2 fine cells and is scored on the same test slab at 512^3.
Halving `data.split.buffer` (8 instead of 16) makes the 256^3 slabs exactly
half the 512^3 ones. See `configs/subhalo/A256_scratch_L10.yaml`.

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
