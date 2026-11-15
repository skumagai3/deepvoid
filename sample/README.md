# Sample data

`tng300-3-dark_n128.npz` is a small sample for `notebooks/01_quickstart.ipynb`:
the IllustrisTNG TNG300-3-Dark simulation at redshift 0 (snapshot 99), a
periodic box of $205\,h^{-1}\,\mathrm{Mpc}$, on a grid of $128^3$ cells ($1.6\,h^{-1}\,\mathrm{Mpc}$ each).

| Key | Contents |
|---|---|
| `dm` | Dark-matter density, $\rho/\bar\rho$, all particles (cloud-in-cell). float16 |
| `subhalos_L7` | Mass density of the most massive subhalos, kept so that their mean separation is $7\,h^{-1}\,\mathrm{Mpc}$, as $\rho/\bar\rho$. 92.6% of cells are empty. float16 |
| `box_size` | 205.0 ($h^{-1}\,\mathrm{Mpc}$) |

Axis order is (x, y, z). The arrays are the same as in
[deepvoid-demo](https://github.com/skumagai3/deepvoid-demo), whose data card
describes how they were made.

At this resolution the cells are larger than the $1\,h^{-1}\,\mathrm{Mpc}$ smoothing of the
DeepVoid label convention, so the sample is for checking that the pipeline
runs, not for results.

## Credit

The IllustrisTNG simulations are provided by the IllustrisTNG collaboration
(https://www.tng-project.org). If you use this sample, cite the TNG100/TNG300
introductory papers (Springel et al. 2018, MNRAS 475, 676; Pillepich et al.
2018, MNRAS 475, 648; Nelson et al. 2018, MNRAS 475, 624; Naiman et al. 2018,
MNRAS 477, 1206; Marinacci et al. 2018, MNRAS 480, 5113) and the public data
release (Nelson et al. 2019, Computational Astrophysics and Cosmology 6, 2).
