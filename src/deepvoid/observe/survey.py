"""Survey effects: footprint masks, radial selection and lightcones.

Planned, not implemented. Intended pieces:

- ``apply_footprint(tracers, mask)``: drop tracers outside an angular mask,
  with the observer placed in or outside the box.
- ``radial_selection(tracers, nz)``: thin tracers to follow a target n(z).
- ``lightcone(snapshots, ...)``: stitch snapshots into a cone.

Masked regions also need to reach the model, either as an extra input
channel or as voxels excluded from the loss, so these functions should
return the survey window alongside the tracers.
"""
