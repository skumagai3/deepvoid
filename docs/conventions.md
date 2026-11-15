# Conventions

These choices define what the model learns. Change them deliberately, and
record the change in a dataset's metadata.

## Units and axes

- Lengths in $h^{-1}\,\mathrm{Mpc}$, masses in $h^{-1}\,M_\odot$, velocities in $\mathrm{km\,s^{-1}}$ (peculiar).
- Arrays are $(x, y, z)$, C order. Boxes are periodic.
- Densities are stored as $\rho/\bar\rho$.
- Mean tracer separation: $\lambda = n^{-1/3}$, with tracers selected as the $N$
  most massive, $N = (L_\mathrm{box}/\lambda)^3$.

## Classes

0 void, 1 wall, 2 filament, 3 halo (`deepvoid.labels`).

## T-web labels

Following the dissertation (Sec. 5.3):

- Effective smoothing $\sigma_\mathrm{eff}^2 = \sigma^2 + \sigma_G^2$, with
  $\sigma_G = L_\mathrm{box}/N$. All labels use
  $\sigma_\mathrm{eff} = 1\,h^{-1}\,\mathrm{Mpc}$, which needs cells smaller than
  $1\,h^{-1}\,\mathrm{Mpc}$. `labels.smoothing_sigma` computes $\sigma$ for a given grid.
- Tidal tensor $T_{ij} = \partial_i \partial_j \phi$ from the normalized Poisson equation, $\nabla^2 \phi = \delta$.
- Eigenvalue threshold $\lambda_\mathrm{th} = 0.65$; class = number of eigenvalues above it.
- Labels always come from the full real-space matter field, whatever the
  input (sparse tracers, redshift space, survey cuts).

Legacy masks: TNG300-3 at $512^3$ used $\sigma = 2.4$ cells
($\sigma_\mathrm{eff} \approx 1.04\,h^{-1}\,\mathrm{Mpc}$); `deepvoid.labels` reproduces it on 100% of voxels. The
dissertation's Bolshoi mask agrees with a recomputation from the Bolshoi
density on only 88% of voxels, so `bolshoi/mask` now points to a mask
recomputed with the convention above; the old one is `bolshoi/mask-legacy`.

## Gridding

Cloud-in-cell, grid points at integer multiples of the cell size
(`deepvoid.grid.cic`).

## Splits and normalization

- Train, validation and test are slabs of the box separated by buffer slabs,
  chosen before any crops are drawn (`deepvoid.split`). Crops never leave
  their slab.
- The test slab is not used for model selection, early stopping or threshold
  tuning.
- Input normalization, $\log_{10}(\rho/\bar\rho + 10^{-3})$ then standardized, is
  fit on the training slab and saved with each run.
- Training crops ($96^3$ by default) are drawn inside the training slab, and the
  model is chosen by validation void MCC (Matthews correlation coefficient;
  void against the other three
  classes) on fixed $64^3$ crops from the validation slab. Reported scores come
  from full-box prediction scored on the test slab only.

## Model outputs: class scores, not probabilities

- For each voxel the network outputs the softmax of its last layer: four
  non-negative numbers summing to 1. These are **class scores**; the first
  is the **void score**. Training makes them small for wrong classes, but
  nothing makes them probabilities, and the code and docs do not call them
  that.
- **Predicted class**: the class with the highest score (the default and
  the standard for multi-class segmentation), or, optionally, with a void
  threshold $t$, void wherever the void score is at least $t$ and otherwise the
  non-void class with the highest score. A threshold needs only that a
  higher void score means "more void-like"; it does not need the scores to
  be probabilities. It is chosen on the validation slab, never the test slab.
- **Calibration** is an empirical property of a model on a given set of
  voxels. Bin the voxels by void score; in each bin compare the mean void
  score with the fraction of those voxels labelled void in the T-web mask.
  If these agree, the void score can be read as the frequency with which
  voxels given that score are void, for that data only (one simulation, one
  tracer sample, one test slab). The expected calibration error is

  $$\mathrm{ECE} = \sum_b \frac{n_b}{N}\,\left|f_b - \bar s_b\right|,$$

  where bin $b$ holds $n_b$ of the $N$ voxels, $f_b$ is the fraction of them
  labelled void and $\bar s_b$ is their mean void score. The same holds for
  the top-class version, which compares the highest score with the fraction
  of voxels whose predicted class is correct.
- **Entropy** of the class scores, $H = -\sum_c s_c \log_2 s_c$ (0 to 2 bits),
  describes how evenly the scores are spread. It is not a probabilistic
  uncertainty unless the scores have been shown to be calibrated.
- **IoU** for a class, with $P$ the predicted and $T$ the true voxels of that class:

  $$\mathrm{IoU} = \frac{|P \cap T|}{|P \cup T|} = \frac{TP}{TP + FP + FN}, \qquad
  F_1 = \frac{2\,TP}{2\,TP + FP + FN}, \qquad
  \mathrm{IoU} = \frac{F_1}{2 - F_1}.$$

  $F_1$ is the Dice coefficient. **MCC** for one class against the rest uses
  the same $TP$, $FP$, $FN$, $TN$:

  $$\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.$$

  The four-class MCC is Gorodkin's (2004) generalization.

## Files

- New volumes: compressed `.npz` with the array and a JSON header
  (`deepvoid.io.save_volume`). Densities float16 where the range allows,
  masks uint8.
- Legacy `.fvol` files are read as-is (`deepvoid.io.read_fvol`).
