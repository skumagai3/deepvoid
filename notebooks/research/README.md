# Research notebooks

The notebooks behind the results. They read TNG300-3 and Bolshoi grids from
the author's Google Drive (the names in `src/deepvoid/catalog.yaml`) and write
runs there, so they will not run elsewhere as they are. The same grids are on Zenodo:
`deepvoid.data.fetch()` downloads them and writes a catalog, and setting a
config's `catalog` to that file runs it on the downloaded copies. Configs are
in `configs/research/`.

| Notebook | Purpose |
|---|---|
| `00_data_check.ipynb` | Load the IllustrisTNG (TNG) and Bolshoi simulations, check alignment and class fractions, recompute the masks |
| `01_smoke_test.ipynb` | Tiny train, resume and score run to check the training code |
| `02_tng_baseline.ipynb` | Dark-matter baseline on TNG300-3 |
| `03_bolshoi_transfer.ipynb` | The TNG baseline applied to Bolshoi without retraining |
| `04_subhalo_smoke_test.ipynb` | Tiny run checking FiLM (feature-wise linear modulation, the spacing conditioning), curriculum and `init_from` before the subhalo runs |
| `05_subhalo_train.ipynb` | Trains one config from `configs/research/subhalo/` per session |
| `06_subhalo_compare.ipynb` | Re-scores finished runs; four-class and void scores by run and spacing, void precision–recall curves, and the FiLM spacing sweep |
| `07_coarse_grid.ipynb` | $256^3$ inputs and mask for coarse-grid runs, and the score ceiling of a $256^3$ grid |
| `08_interpretability.ipynb` | Activation maps, centered kernel alignment (CKA) across spacings, a probe for $\lambda$, effective receptive field, error and score-entropy maps, accuracy vs distance to the nearest tracer, reliability of the class scores |
| `09_figures.ipynb` | Figures from data, finished runs and trained models, including the README summary figure |
| `10_pointclouds.ipynb` | 3D views for Potree: the subhalo samples colored by T-web (tidal-tensor) class, and a $\lambda = 10$ slab with void/wall errors, void surfaces and dark matter |
| `11_package_model.ipynb` | Packages the run behind the README results as a model release, checking it predicts exactly as the run does |
| `12_cross_simulation.ipynb` | The packaged TNG subhalo model on Bolshoi halos, scored against Bolshoi's own T-web labels |
| `13_redshift_space.ipynb` | Redshift-space subhalo grids (z as line of sight) from the TNG300-1 catalog, checks against the older grids, and the real-space model scored on redshift-space inputs |

Planned:

| Notebook | Purpose |
|---|---|
| `14_survey.ipynb` | Survey masks and radial selection |
| `15_calibration.ipynb` | Reliability of the class scores and temperature scaling |
