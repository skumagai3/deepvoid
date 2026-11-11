# Notebooks

Thin Colab front-ends. Each one installs the package, loads a config, and
calls into `deepvoid`; the logic lives in the package, not in cells.

Each notebook needs a Colab secret `GITHUB_TOKEN` with read access to the
repo while it is private.

| Notebook | Purpose |
|---|---|
| `00_data_check.ipynb` | Load TNG and Bolshoi, check alignment and class fractions, recompute the masks |
| `01_smoke_test.ipynb` | Tiny train, resume and score run to check the training code |
| `02_tng_baseline.ipynb` | Dark-matter baseline on TNG300-3 |
| `03_bolshoi_transfer.ipynb` | The TNG baseline applied to Bolshoi without retraining |
| `04_subhalo_smoke_test.ipynb` | Tiny run checking FiLM, curriculum and `init_from` before the subhalo runs |
| `05_subhalo_train.ipynb` | Trains one config from `configs/subhalo/` per session |
| `06_subhalo_compare.ipynb` | Test MCC by run and spacing, and the FiLM spacing sweep |
| `07_coarse_grid.ipynb` | 256³ inputs and mask for coarse-grid runs, and the score ceiling of a 256³ grid |
| `08_interpretability.ipynb` | Activation maps, CKA across spacings, a probe for λ, effective receptive field, error and uncertainty maps, accuracy vs distance to the nearest tracer, calibration |

Planned:

| Notebook | Purpose |
|---|---|
| `08_redshift_space.ipynb` | Redshift-space inputs with real-space labels |
| `09_survey.ipynb` | Survey masks and radial selection |
| `10_confidence.ipynb` | Temperature scaling and reliability diagrams |
