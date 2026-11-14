# Notebooks

Written for Google Colab with a GPU runtime. Each one installs the package and
calls into it; the logic lives in `src/deepvoid`, not in the cells.

| Notebook | What it does |
|---|---|
| `01_quickstart.ipynb` | The whole pipeline on a small sample that ships with the repository: grids, labels, training, scores, figures |
| `02_your_simulation.ipynb` | A template for training on your own matter density and tracers |

`research/` holds the notebooks behind the results in the README and
`docs/overview.md`. They read the author's data files from Google Drive, so
they document how the results were made but will not run elsewhere as they are.
