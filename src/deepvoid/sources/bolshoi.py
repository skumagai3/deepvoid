"""Bolshoi (Klypin, Trujillo-Gomez & Primack 2011).

Only pre-built grids are used so far (see ``catalog.yaml``). A halo-catalog
reader belongs here once redshift-space or survey experiments need Bolshoi.
"""

from . import Cosmology

BOX_SIZE = 250.0  # h^-1 Mpc
COSMOLOGY = Cosmology(omega_m=0.27, h=0.70)  # WMAP5
