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
| `05_subhalo_train.ipynb` | Trains one config from `configs/research/subhalo/` (or `rsd/`, `papersplit/`) per session |
| `06_subhalo_compare.ipynb` | Re-scores finished runs; four-class and void scores by run and spacing, void precision–recall curves, and the FiLM spacing sweep |
| `07_coarse_grid.ipynb` | $256^3$ inputs and mask for coarse-grid runs, and the score ceiling of a $256^3$ grid |
| `08_interpretability.ipynb` | Activation maps, centered kernel alignment (CKA) across spacings, a probe for $\lambda$, effective receptive field, error and score-entropy maps, accuracy vs distance to the nearest tracer, reliability of the class scores; receptive field along and across the line of sight for the real- and redshift-space models |
| `09_figures.ipynb` | Figures from data, finished runs and trained models, including the README summary figure |
| `10_pointclouds.ipynb` | 3D views for Potree: the subhalo samples colored by T-web (tidal-tensor) class, and a $\lambda = 10$ slab with void/wall errors, void surfaces and dark matter |
| `11_package_model.ipynb` | Packages the run behind the README results as a model release, checking it predicts exactly as the run does |
| `12_cross_simulation.ipynb` | The packaged TNG subhalo model on Bolshoi halos, scored against Bolshoi's own T-web labels |
| `13_redshift_space.ipynb` | Redshift-space subhalo grids (z as line of sight) from the TNG300-1 catalog, checks against the older grids, and the real-space model scored on redshift-space inputs |
| `14_information_limit.ipynb` | How much of the depth-5 models' error is in voxels far from any tracer: class shares by distance, the model against a distance-only classifier, MCC if far voxels were right, and training vs test slab scores (real and redshift space) |
| `15_smdpl.ipynb` | SMDPL density grid from CosmoSim's TAP service in 27 chunks (needs a `COSMOSIM_TOKEN` Colab secret), checked and assembled to 512³ |
| `16_smdpl_scoring.ipynb` | The real- and redshift-space models on SMDPL (whole box; redshift space along x, y and z), scored on SMDPL's 0.78 h⁻¹ Mpc grid, with the TNG test slab scored on a 0.80 h⁻¹ Mpc grid as the reference; measures where SMDPL's grid points sit |
| `17_void_catalogs.ipynb` | Void catalogs from void voxels on the whole TNG or Bolshoi box: thresholded cores, distance basins with a persistence cut, the same on a score-averaged distance, and maximal empty spheres, each run on the truth and on the predictions with the same settings; persistence diagrams, size functions, matching |
| `18_void_definitions.ipynb` | T-web basins, DeepVoid, VIDE (pyVIDE) and VAST VoidFinder on the same box (TNG, SMDPL or Bolshoi) at λ = 3, 5, 10: size functions, slices, true T-web composition of each catalog's voids, stability across spacing, radial profiles around void centers, merge trees |
| `19_bolshoi_halos.ipynb` | Bolshoi BDM halo positions (z = 0) from CosmoSim's TAP service; where Bolshoi's grid points sit and which halo selection matches the dissertation's halo grids; feeds notebook 18 |
| `20_vweb.ipynb` | V-web (velocity-shear) labels on TNG300-3-Dark: dark-matter particles from the TNG API, smoothed velocity field, V-web against f × T-web eigenvalues, slices at several thresholds, class fractions and void percolation against the threshold, confusion matrix against the T-web labels |

Planned:

| Notebook | Purpose |
|---|---|
| `21_survey.ipynb` | Survey masks and radial selection |
| `22_calibration.ipynb` | Reliability of the class scores and temperature scaling |
