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
| `data.smoothing` | null | Add a Gaussian-smoothed copy of each input as a second channel: `{sigma_spacing: 0.5}` (sigma as a fraction of the tracer spacing) or `{sigma_mpc: 5}`; `keep_raw: false` drops the raw channel |
| `model.depth`, `model.base_filters` | 4, 16 | U-Net size |
| `model.batch_norm`, `model.bn_momentum` | true, 0.99 | Batch normalization |
| `model.dropout` | 0 | Dropout at the bottom of the U-Net |
| `model.conditioning` | null | Condition on tracer spacing (FiLM), e.g. `{units: 32, noise: 0.1}`; see below |
| `model.deep_supervision` | false | Extra training outputs at each coarser level; see below |
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
| `training.loss.void_weight` | 0 | Weight of an added void-vs-rest term; see below |
| `training.early_stopping` | `val_void_mcc`, patience 15, min_delta 0.002 | Score to track (`val_void_mcc`: void against everything else; `val_mcc`: all four classes), and epochs to wait without a gain of at least `min_delta` |
| `training.mixed_precision` | true | float16 compute, float32 weights |
| `training.xla` | true | Compile the training step with XLA |
| `training.resume` | true | Continue an interrupted run in the same folder |
| `evaluate.crop`, `evaluate.margin` | 128, 32 | Crop size and discarded edge for full-box prediction. The margin should cover the model's effective receptive field: with 16 cells, tile seams showed in the λ = 10 predictions (notebook 08). Re-score an older run with `evaluate_run(out, margin=32)` |
| `evaluate.tta` | `none` | Test-time augmentation: `none`, `flips` (8x) or `full` (48x) |
| `evaluate.inputs` | null | Inputs to score on the test slab; null means the training inputs |
| `evaluate.labels` | null | Mask on a finer grid to score against; predictions are upsampled to it (see below) |
| `evaluate.void_threshold` | null | Optional: also score the test slab with a void-score threshold (`tune`: chosen on the validation slab; or a number). The predicted classes are always the highest-scoring ones |
| `evaluate.void_objective` | `mcc` | What the tuned threshold maximizes on the validation slab: `mcc`, `f1` or `iou` (void against everything else) |

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
   slab, written to `test_scores.json`, with a figure (`test_figure.png`): a slice through the middle of the test slab (input, truth, prediction, wrong voxels) and the confusion matrix for each input.

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

## Void scores and the void threshold

Voids are the science product, so void-vs-rest scores come first:
`void_mcc` and `void_iou` (void against all other classes) are in every
score file, `val_void_mcc` is logged each epoch (and per input with
several inputs), and it is the default early-stopping score.

The network's outputs are class scores, not probabilities (see
`docs/conventions.md`). The predicted class is the one with the highest
score (argmax), as is standard for multi-class segmentation; all headline
scores and figures use it.

Every score file also records, for void against the rest, the void
precision, recall and MCC at every threshold on the void score, on the
validation and test slabs (`void_curve`). These describe the scores and do
not change any prediction; notebook 06 draws them as precision-recall
curves.

Optionally, `evaluate.void_threshold` also scores the test slab with a
void-score threshold: void wherever the void score is at least t,
otherwise the highest-scoring other class. With `tune`, t maximizes
`evaluate.void_objective` on the validation slab (never the test slab).
The results go under `tuned`. This trades void purity against completeness
and is meant for analysis (e.g. choosing an operating point for a void
catalog), not as the default decision rule.

`void_reliability` in the same file says whether the void score is
calibrated on the test slab: voxels binned by void score, with each bin's
mean score, the fraction labelled void, and the expected calibration error.
The test figure shows the argmax prediction, the test-slab void curve and
this reliability curve.

## Void-vs-rest loss and deep supervision

**Void-vs-rest loss.** With `training.loss.void_weight: w`, every voxel's
loss gains w times the binary cross-entropy of its void score s_v against
its void label: -[1(void) log s_v + (1 - 1(void)) log(1 - s_v)]. The
four-class loss is unchanged; the added term trains the void score directly
for the two-class decision voids are judged on.

**Deep supervision.** With `model.deep_supervision: true`, the U-Net gets
one extra output per level below full resolution: after each decoder block
at 1/2, 1/4, ... resolution and after the bottleneck, so depth d gives d - 1
extra outputs (a depth-4 model has them at 1/2, 1/4 and 1/8; depth 5 adds
1/16). Each is trained against the class fractions in the blocks of the
full-resolution labels it covers (the one-hot labels averaged over 2^3,
4^3, ... voxels), so blocks that straddle a boundary keep both classes. The
losses are weighted 1, 1/2, 1/4, ... from full resolution down, normalized
to sum to 1. The coarse outputs exist only during training: validation,
`best.keras` and everything downstream use the full-resolution output
alone. The training crop and validation crop must be divisible by
2^(depth - 1), as for any U-Net of that depth.

Planned, not yet implemented: weighting the loss by distance to the
void boundary.

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
