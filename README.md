# DeepVoid

DeepVoid labels every voxel of a simulated universe as void, wall, filament
or halo with a 3D U-Net. It learns the tidal-tensor (T-web) classification
of the full dark-matter field and can recover it from much sparser tracers,
such as the most massive subhalos.

This repository is a clean rewrite of the research code behind the DeepVoid
paper and dissertation, organized so that new simulations and observational
effects (redshift-space distortions, survey masks, selection functions) can
be added without touching the rest.

Status: early. The data, labeling, splitting, scoring and calibration code
is in place and tested; the training path (pipeline, models, losses,
train) is written but has not yet been run end to end.

## Layout

```
configs/              one YAML per experiment
src/deepvoid/
  catalog.yaml        named datasets: where each file lives and what it is
  volume.py           Volume: a periodic grid plus its metadata
  io.py               legacy .fvol and compact .npz files
  catalog.py          load datasets by name
  sources/            simulation readers (TNG, Bolshoi, Indra, Quijote)
  observe/            tracer selection, redshift space, survey effects
  grid.py             cloud-in-cell gridding
  labels/             T-web classification
  split.py            spatial train / val / test slabs
  preprocess.py       input normalization
  pipeline.py         random crops drawn on the GPU
  models/, losses.py  3D U-Net and losses
  train.py            training from a config
  predict.py          full-box prediction from overlapping crops
  evaluate.py         confusion matrices, F1, MCC
  calibrate.py        temperature scaling, reliability
scripts/              one-off utilities
notebooks/            Colab front-ends
docs/conventions.md   units, classes, smoothing, splits
```

Every input is a `Volume` with metadata saying which simulation, tracer,
spacing and observational effects produced it, so results always carry
their provenance.

## Quick start (Colab)

```python
from google.colab import drive
drive.mount("/content/drive")
!pip install -q "git+https://github.com/skumagai3/deepvoid.git"

from deepvoid import Catalog
from deepvoid.labels import tweb_labels

cat = Catalog()                      # data root: /content/drive/MyDrive/data
dm = cat.load("tng300-3/dm")
mask = tweb_labels(dm)               # sigma_eff = 1 h^-1 Mpc, lambda_th = 0.65
```

Set `DEEPVOID_DATA` to use a different data root.

## Conventions

See [docs/conventions.md](docs/conventions.md). In short: T-web labels with
an effective smoothing of 1 h^-1 Mpc and an eigenvalue threshold of 0.65,
computed from the real-space matter field; train, validation and test are
separate slabs of the box; normalization is fit on the training slab.

## Data and credits

The IllustrisTNG simulations are provided by the IllustrisTNG collaboration
(https://www.tng-project.org): Springel et al. 2018, MNRAS 475, 676;
Pillepich et al. 2018, MNRAS 475, 648; Nelson et al. 2018, MNRAS 475, 624;
Naiman et al. 2018, MNRAS 477, 1206; Marinacci et al. 2018, MNRAS 480, 5113.
Data release: Nelson et al. 2019, ComAC 6, 2, doi:10.1186/s40668-019-0028-x.

The Bolshoi simulation (Klypin, Trujillo-Gomez & Primack 2011, ApJ 740, 102)
is distributed through CosmoSim and Skies & Universes.
