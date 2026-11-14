# DeepVoid input grids from IllustrisTNG and Bolshoi

*Text for the Zenodo record of the DeepVoid data release. The files are
made by `scripts/make_data_release.py`.*

## Description

Gridded density fields and cosmic-web class masks used to train and test
[DeepVoid](https://github.com/skumagai3/deepvoid), a 3D U-Net that labels
each cell of a simulated volume as void, wall, filament or halo.

**These grids are derived from simulation data produced and released by
others: the IllustrisTNG project (TNG300-3-Dark) and the Bolshoi
simulation (via the CosmoSim database).** We only gridded the particles and
halos and computed the class masks. Please credit the simulations as listed
below whenever you use these files.

All grids cover the full periodic simulation box at redshift 0, with axis
order (x, y, z) and lengths in h⁻¹ Mpc.

| File | Contents |
|---|---|
| `tng300-3-dark_dm_n512.npz` | TNG300-3-Dark, dark-matter density from all particles; 512³ cells in a 205 h⁻¹ Mpc box |
| `tng300-3-dark_tweb_n512.npz` | T-web classes from that density (0 void, 1 wall, 2 filament, 3 halo): Gaussian smoothing of 2.4 cells, eigenvalue threshold 0.65 |
| `tng300-3-dark_subhalos_L{3,5,7,10}_n512.npz` | Mass density of the most massive subhalos, selected so that their mean separation is 3, 5, 7 or 10 h⁻¹ Mpc |
| `bolshoi_dm_n640.npz` | Bolshoi, dark-matter density from all 2048³ particles (mean separation 0.122 h⁻¹ Mpc); 640³ cells in a 250 h⁻¹ Mpc box |
| `bolshoi_tweb_n640.npz` | T-web classes from that density: effective smoothing 1 h⁻¹ Mpc, eigenvalue threshold 0.65 |
| `bolshoi_halos_L{3,5,7,10}_n640.npz` | Bolshoi halos at mean separations of 3, 5, 7 or 10 h⁻¹ Mpc |
| `MANIFEST.txt` | Size and SHA-256 checksum of every file |

**Format.** Each `.npz` opens with NumPy alone:

```python
import numpy as np
grid = np.load("tng300-3-dark_dm_n512.npz")["data"]
```

`data` holds the values (uint8 for class masks, float32 otherwise);
`header` is a JSON string with the box size and the file's provenance;
`fvol_header` keeps the header of the file the grid was converted from.
The conversion is lossless: each original file can be rebuilt byte for byte
as `fvol_header` followed by `data` as float32, and its SHA-256 checksum is
stored in `header` under `source_sha256`.

**Cosmologies.** TNG300: Ωm = 0.3089, ΩΛ = 0.6911, Ωb = 0.0486,
σ8 = 0.8159, ns = 0.9667, h = 0.6774. Bolshoi: Ωm = 0.27, ΩΛ = 0.73,
Ωb = 0.0469, σ8 = 0.82, ns = 0.95, h = 0.70.

## Credit

**IllustrisTNG.** Any use of the TNG300 data should cite the data release
paper and the introductory papers:

- Nelson, D., et al. 2019, Computational Astrophysics and Cosmology, 6, 2
- Springel, V., et al. 2018, MNRAS, 475, 676
- Pillepich, A., et al. 2018, MNRAS, 475, 648
- Nelson, D., et al. 2018, MNRAS, 475, 624
- Naiman, J. P., et al. 2018, MNRAS, 477, 1206
- Marinacci, F., et al. 2018, MNRAS, 480, 5113

The data come from https://www.tng-project.org.

**Bolshoi.** Cite the simulation, the CosmoSim Bolshoi dataset and, for the
halo grids, the halo finder:

- Klypin, A. A., Trujillo-Gomez, S., & Primack, J. 2011, ApJ, 740, 102
- CosmoSim Bolshoi dataset, https://doi.org/10.17876/cosmosim/bolshoi
- Klypin, A., & Holtzman, J. 1997, arXiv:astro-ph/9712217 (Bound Density
  Maxima halo finder)

The CosmoSim database asks publications to include its acknowledgment
paragraph, given at https://www.cosmosim.org/cms/credits/.

**DeepVoid.** If these grids are useful to you, please also cite

- Kumagai, S., Vogeley, M. S., Aragon-Calvo, M. A., Douglass, K. A.,
  BenZvi, S., & Neyrinck, M. 2025, DeepVoid: A Deep Learning Void Detector,
  arXiv:2504.21134

## License

Our processing (the gridding and the class masks) is released under
CC BY 4.0. The Bolshoi data are released by CosmoSim under CC0 1.0. The
IllustrisTNG data remain subject to the terms of the IllustrisTNG project.
