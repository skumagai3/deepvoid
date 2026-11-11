"""Convert legacy .fvol files listed in catalog.yaml to compact .npz volumes.

    python scripts/convert_legacy.py OUT_DIR [NAME_PREFIX]

Densities are stored as rho / rho_bar in float16 when they fit (float32
otherwise) and masks as uint8. The originals are left untouched; once the
new files are checked, point catalog.yaml at them.
"""

import sys
from pathlib import Path

from deepvoid import Catalog, save_volume


def main(out_dir: str, prefix: str = "") -> None:
    catalog = Catalog()
    out_dir = Path(out_dir)
    for name in catalog.names(prefix):
        src = catalog.path(name)
        dst = out_dir / (name.replace("/", "__") + ".npz")
        if dst.exists():
            print(f"skip   {name}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        vol = catalog.load(name)
        if vol.meta.get("field") == "density":
            vol.data = vol.data / vol.data.mean()
            vol.meta["units"] = "rho/rho_bar"
        try:
            save_volume(vol, dst)
        except ValueError:
            save_volume(vol, dst, dtype="float32")
        print(f"wrote  {name}: {src.stat().st_size / 1e6:.0f} MB -> "
              f"{dst.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
