# DeepVoid trained model: TNG300-3 subhalos

*Text for the Zenodo record of the DeepVoid model release. The files are
made by `scripts/make_model_release.py`.*

## Description

A trained [DeepVoid](https://github.com/skumagai3/deepvoid) network: a 3D
U-Net that labels each cell of a periodic simulation box as void, wall,
filament or halo, from the positions of sparse tracers such as halos or
galaxies.

The network was trained on IllustrisTNG (TNG300-3-Dark, redshift 0)
subhalos at mean separations of 3, 5, 7 and $10\,h^{-1}\,\mathrm{Mpc}$ at
once, on a $512^3$ grid. The target labels are T-web classes computed from
the full dark-matter density (effective smoothing $1\,h^{-1}\,\mathrm{Mpc}$,
eigenvalue threshold 0.65). The training grids are in the DeepVoid data
release, https://doi.org/10.5281/zenodo.24233069.

| File | Contents |
|---|---|
| `model.keras` | The trained network (Keras) |
| `model.json` | How inputs must be built: cell size, smoothing, normalization for each trained spacing, prediction crop and margin |
| `config.yaml` | The full training configuration |
| `normalization.json` | Input normalization fitted on the training slab |
| `test_scores.json` | Scores on the held-out test slab |
| `log.csv` | Training and validation scores for each epoch |
| `MANIFEST.txt` | Size and SHA-256 checksum of every file |

**Use.** With the `deepvoid` package:

```python
from deepvoid import pretrained

model = pretrained.load()
scores = model.predict(field, box_size=box_size, spacing=spacing)
classes = scores.argmax(-1)    # 0 void, 1 wall, 2 filament, 3 halo
```

The input must be a periodic box gridded like the training data: cells of
$0.40\,h^{-1}\,\mathrm{Mpc}$, and the mass of the most massive tracers,
enough of them for the given mean separation, assigned by cloud-in-cell.

**Scores.** Matthews correlation coefficient (MCC) on a slab of the TNG box
never used for training, model selection or tuning, and on the whole
Bolshoi box (Bolshoi halos selected the same way, scored against Bolshoi's
own T-web labels; nothing from Bolshoi was used in training):

| Mean separation ($h^{-1}\,\mathrm{Mpc}$) | 3 | 5 | 7 | 10 |
|---|---|---|---|---|
| TNG test slab, four-class MCC | 0.885 | 0.842 | 0.781 | 0.683 |
| TNG test slab, void MCC | 0.918 | 0.880 | 0.825 | 0.736 |
| Bolshoi, four-class MCC | 0.881 | 0.832 | 0.769 | 0.679 |
| Bolshoi, void MCC | 0.913 | 0.869 | 0.813 | 0.729 |

Two retrainings with other random seeds match the TNG scores to within
0.002.

## Credit

The model was trained on IllustrisTNG data and tested on Bolshoi data.
Please cite the DeepVoid paper,

- Kumagai, S., Vogeley, M. S., Aragon-Calvo, M. A., Douglass, K. A.,
  BenZvi, S., & Neyrinck, M. 2026, DeepVoid: A Deep Learning Void Detector,
  ApJ, 998, 85, https://doi.org/10.3847/1538-4357/ae2c80

and the IllustrisTNG data release and introductory papers:

- Nelson, D., et al. 2019, Computational Astrophysics and Cosmology, 6, 2
- Springel, V., et al. 2018, MNRAS, 475, 676
- Pillepich, A., et al. 2018, MNRAS, 475, 648
- Nelson, D., et al. 2018, MNRAS, 475, 624
- Naiman, J. P., et al. 2018, MNRAS, 477, 1206
- Marinacci, F., et al. 2018, MNRAS, 480, 5113

## License

CC BY 4.0.
