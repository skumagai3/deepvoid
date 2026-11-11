# Conventions

These choices define what the model learns. Change them deliberately, and
record the change in a dataset's metadata.

## Units and axes

- Lengths in h^-1 Mpc, masses in h^-1 Msun, velocities in km/s (peculiar).
- Arrays are (x, y, z), C order. Boxes are periodic.
- Densities are stored as rho / rho_bar.
- Mean tracer separation: lambda = n^(-1/3), with tracers selected as the N most
  massive, N = (L_box / lambda)^3.

## Classes

0 void, 1 wall, 2 filament, 3 halo (`deepvoid.labels`).

## T-web labels

Following the dissertation (Sec. 5.3):

- Effective smoothing sigma_eff^2 = sigma^2 + sigma_G^2 with sigma_G = L_box / N.
  All labels use sigma_eff = 1 h^-1 Mpc, which needs cells smaller than
  1 h^-1 Mpc. `labels.smoothing_sigma` computes sigma for a given grid.
- Tidal tensor from the normalized Poisson equation, del^2 phi = delta.
- Eigenvalue threshold lambda_th = 0.65; class = number of eigenvalues above it.
- Labels always come from the full real-space matter field, whatever the
  input (sparse tracers, redshift space, survey cuts).

Legacy masks: TNG300-3 at 512^3 used sigma = 2.4 cells (sigma_eff about
1.04 h^-1 Mpc); Bolshoi at 640^3 used sigma = 0.916 h^-1 Mpc.

## Gridding

Cloud-in-cell, grid points at integer multiples of the cell size
(`deepvoid.grid.cic`).

## Splits and normalization

- Train, validation and test are slabs of the box separated by buffer slabs,
  chosen before any crops are drawn (`deepvoid.split`). Crops never leave
  their slab.
- The test slab is not used for model selection, early stopping or threshold
  tuning.
- Input normalization (log10(rho/rho_bar + 1e-3), then standardized) is fit on
  the training slab and saved with each run.

## Files

- New volumes: compressed `.npz` with the array and a JSON header
  (`deepvoid.io.save_volume`). Densities float16 where the range allows,
  masks uint8.
- Legacy `.fvol` files are read as-is (`deepvoid.io.read_fvol`).
