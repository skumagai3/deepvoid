# Research configs

The experiments behind the results, kept as they were run. The dataset names
in them refer to the author's files (`src/deepvoid/catalog.yaml`). For a
commented config to start from, use `configs/template.yaml`.

| Config | What it is |
|---|---|
| `tng_dm_baseline.yaml` | Full dark-matter density in, T-web classes out (TNG300-3) |
| `subhalo/A_*` | One model at subhalo spacing 10 h⁻¹ Mpc, from scratch; `smooth2ch` adds the smoothed channel, `smoothonly` uses only it, `A256` is on a 256³ grid |
| `subhalo/B_*` | The dark-matter baseline fine-tuned at spacing 10 |
| `subhalo/C_mixed*` | One model on spacings 3, 5, 7 and 10 at once; suffixes add the smoothed channel (`smooth2ch`), a fifth U-Net level (`depth5`), a void-vs-rest loss term (`voidloss`) or deep supervision (`deepsup`) |
| `subhalo/C_film*` | As `C_mixed`, with the model told the spacing (FiLM), with or without a dense-to-sparse curriculum |
| `papersplit/*` | The first paper's subcube split, for the test in `docs/overview.md`, section 7 |

`subhalo/C_mixed_smooth2ch_depth5.yaml` is the model reported in the README.
