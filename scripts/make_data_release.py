"""Build the DeepVoid data release: lossless .npz copies of the grids.

Usage (on Colab, with Google Drive mounted):

    python scripts/make_data_release.py /content/drive/MyDrive/data /content/drive/MyDrive/deepvoid_release

Each .fvol grid (a 256-byte header followed by float32 values) is written as
a compressed .npz holding

    data          the values: uint8 for class masks whose values are all
                  0, 1, 2 or 3, float32 otherwise
    header        JSON readable by deepvoid.io.load_volume: box size and
                  provenance, including the SHA-256 of the original file
    fvol_header   the original 256 header bytes

The original file is then rebuilt from the .npz alone, as fvol_header + data
as float32, and its SHA-256 compared with the original's. Any mismatch stops
the script. Files already in .npz form are copied unchanged and checksummed.
A MANIFEST.txt with every file's size and SHA-256 is written last.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

TNG_BOX = 205.0       # h^-1 Mpc
BOLSHOI_BOX = 250.0   # h^-1 Mpc

# (source file under the data folder, released file name, box size, field, spacing)
FILES = [
    ("TNG/DM_DEN_snap99_Nm=512.fvol", "tng300-3-dark_dm_n512.npz", TNG_BOX, "density", 0.33),
    ("TNG/TNG300-3-Dark-mask-Nm=512-th=0.65-sig=2.4.fvol", "tng300-3-dark_tweb_n512.npz", TNG_BOX, "mask", None),
    ("TNG/subs1_mass_Nm512_L3_d_None_smooth.fvol", "tng300-3-dark_subhalos_L3_n512.npz", TNG_BOX, "density", 3.0),
    ("TNG/subs1_mass_Nm512_L5_d_None_smooth.fvol", "tng300-3-dark_subhalos_L5_n512.npz", TNG_BOX, "density", 5.0),
    ("TNG/subs1_mass_Nm512_L7_d_None_smooth.fvol", "tng300-3-dark_subhalos_L7_n512.npz", TNG_BOX, "density", 7.0),
    ("TNG/subs1_mass_Nm512_L10_d_None_smooth.fvol", "tng300-3-dark_subhalos_L10_n512.npz", TNG_BOX, "density", 10.0),
    ("Bolshoi/Bolshoi_halo_CIC_640_L=0.122.fvol", "bolshoi_dm_n640.npz", BOLSHOI_BOX, "density", 0.122),
    ("Bolshoi/bolshoi_mask_tweb_th0.65_seff1.npz", "bolshoi_tweb_n640.npz", BOLSHOI_BOX, "mask", None),
    ("Bolshoi/Bolshoi_halo_CIC_640_L=3.0.fvol", "bolshoi_halos_L3_n640.npz", BOLSHOI_BOX, "density", 3.0),
    ("Bolshoi/Bolshoi_halo_CIC_640_L=5.0.fvol", "bolshoi_halos_L5_n640.npz", BOLSHOI_BOX, "density", 5.0),
    ("Bolshoi/Bolshoi_halo_CIC_640_L=7.0.fvol", "bolshoi_halos_L7_n640.npz", BOLSHOI_BOX, "density", 7.0),
    ("Bolshoi/Bolshoi_halo_CIC_640_L=10.0.fvol", "bolshoi_halos_L10_n640.npz", BOLSHOI_BOX, "density", 10.0),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(src: Path, dst: Path, box: float, field: str, spacing: float | None) -> str:
    raw = src.read_bytes()
    original = sha256(raw)
    head = raw[:256]
    nx, ny, nz = (int(v) for v in np.frombuffer(head[12:24], dtype="<i4"))
    values = np.frombuffer(raw, dtype="<f4", offset=256)
    if values.size != nx * ny * nz:
        raise ValueError(f"{src}: header says {nx}x{ny}x{nz}, file holds {values.size} values")
    values = values.reshape(nx, ny, nz)
    if field == "mask":
        if not np.all(np.isin(values, [0, 1, 2, 3])):
            raise ValueError(f"{src}: a mask with values other than 0, 1, 2, 3")
        data = values.astype(np.uint8)
    else:
        data = values
    meta = {"field": field, "source_file": src.name, "source_sha256": original}
    if spacing is not None:
        meta["spacing"] = spacing
    header = json.dumps({"box_size": box, "meta": meta})
    np.savez_compressed(dst, data=data, header=np.array(header),
                        fvol_header=np.frombuffer(head, dtype=np.uint8))
    with np.load(dst) as f:
        rebuilt = f["fvol_header"].tobytes() + f["data"].astype("<f4").tobytes()
    if sha256(rebuilt) != original:
        raise RuntimeError(f"{dst}: the original cannot be rebuilt exactly; stopping")
    return original


def main(data_root: str, out: str) -> None:
    data_root, out = Path(data_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    missing = [rel for rel, *_ in FILES if not (data_root / rel).exists()]
    if missing:
        raise FileNotFoundError("not found under " + str(data_root) + ": " + ", ".join(missing))
    rows = []
    for rel, name, box, field, spacing in FILES:
        src, dst = data_root / rel, out / name
        if src.suffix == ".npz":
            shutil.copyfile(src, dst)
            note = "copied unchanged"
        else:
            convert(src, dst, box, field, spacing)
            note = "rebuilds the original byte for byte"
        size = dst.stat().st_size
        rows.append((name, size, file_sha256(dst), rel))
        print(f"{name}: {src.stat().st_size / 1e6:.0f} MB -> {size / 1e6:.1f} MB, {note}", flush=True)
    with open(out / "MANIFEST.txt", "w") as f:
        f.write("# file  bytes  sha256  source\n")
        for name, size, digest, rel in rows:
            f.write(f"{name}  {size}  {digest}  {rel}\n")
    total = sum(r[1] for r in rows)
    print(f"done: {len(rows)} files, {total / 1e9:.2f} GB, in {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
