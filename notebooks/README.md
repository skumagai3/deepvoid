# Notebooks

Written for Google Colab with a GPU runtime. Each one installs the package and
calls into it; the logic lives in `src/deepvoid`, not in the cells.

| Notebook | What it does |
|---|---|
| `01_quickstart.ipynb` | The whole pipeline on a small sample that ships with the repository: grids, labels, training, scores, figures |
| `02_your_simulation.ipynb` | A template for training on your own matter density and tracers |
| `03_use_trained_model.ipynb` | The published trained model, no training: download it, predict a whole box, score it on the held-out slab, and how to run it on your own box |

`research/` holds the notebooks behind the results in the README and
`docs/overview.md`. They read the author's data files from Google Drive, so
they document how the results were made but will not run elsewhere as they are.
