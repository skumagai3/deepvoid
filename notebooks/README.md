# Notebooks

Thin Colab front-ends. Each one installs the package, loads a config, and
calls into `deepvoid`; the logic lives in the package, not in cells.

Planned:

| Notebook | Purpose |
|---|---|
| `00_data_inventory.ipynb` | Load each catalog entry, check shapes, alignment and class fractions |
| `01_labels.ipynb` | Build or rebuild T-web masks with the shared convention |
| `02_train.ipynb` | Train from a config |
| `03_evaluate.ipynb` | Full-box prediction, held-out scores, confusion matrices, slices |
| `04_cross_sim.ipynb` | Train on one simulation, test on another |
| `05_redshift_space.ipynb` | Redshift-space inputs with real-space labels |
| `06_survey.ipynb` | Survey masks and radial selection |
| `07_confidence.ipynb` | Temperature scaling and reliability diagrams |
