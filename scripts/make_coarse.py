"""Write 256^3 versions of the TNG300-3 inputs and a 256^3 T-web mask.

    python scripts/make_coarse.py            # uses the packaged catalog and its data root

Inputs (dark matter and the subhalo fields) are averaged over 2x2x2 blocks.
The mask is recomputed from the 256^3 dark-matter density with the same
sigma_eff = 1 h^-1 Mpc convention as the 512^3 mask, rather than by
majority vote of the 512^3 mask, which would erase the smallest halo
regions. Files go to <data root>/TNG/n256/ and match the `tng300-3/n256/...`
entries in catalog.yaml. Existing files are left alone.
"""

from __future__ import annotations

import sys

from deepvoid import Catalog, save_volume
from deepvoid.labels import tweb_labels
from deepvoid.resample import coarsen

FACTOR = 2
SOURCES = ["tng300-3/dm"] + [f"tng300-3/subhalos/L{L}" for L in (3, 5, 7, 10)]


def main(catalog: Catalog) -> None:
    for source in SOURCES:
        target = source.replace("tng300-3/", "tng300-3/n256/")
        path = catalog.path(target)
        if path.exists():
            print(f"exists: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        coarse = coarsen(catalog.load(source), FACTOR)
        save_volume(coarse, path, dtype="float32")
        print(f"wrote {path} ({coarse.n}^3)")
        if source == "tng300-3/dm":
            mask_path = catalog.path("tng300-3/n256/mask")
            if mask_path.exists():
                print(f"exists: {mask_path}")
                continue
            mask = tweb_labels(coarse, sigma_eff=1.0, threshold=0.65)
            save_volume(mask, mask_path)
            fractions = [float((mask.data == c).mean()) for c in range(4)]
            print(f"wrote {mask_path}; class fractions {[round(f, 3) for f in fractions]}")


if __name__ == "__main__":
    main(Catalog(*sys.argv[1:2]))
