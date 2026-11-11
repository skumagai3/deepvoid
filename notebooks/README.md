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

Planned:

| Notebook | Purpose |
|---|---|
| `03_evaluate.ipynb` | Full-box prediction, held-out scores, confusion matrices, slices |
| `04_cross_sim.ipynb` | Train on one simulation, test on another |
| `05_redshift_space.ipynb` | Redshift-space inputs with real-space labels |
| `06_survey.ipynb` | Survey masks and radial selection |
| `07_confidence.ipynb` | Temperature scaling and reliability diagrams |
