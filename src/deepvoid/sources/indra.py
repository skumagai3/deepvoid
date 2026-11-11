"""Indra (Falck et al. 2021), on SciServer.

Not implemented yet. Indra boxes are 1 h^-1 Gpc with 1024^3 particles, so a
512^3 grid has 1.95 h^-1 Mpc cells, larger than the 1 h^-1 Mpc effective
smoothing used for T-web labels. Using Indra with the same label convention
needs a grid of at least ~1024^3. Density grids are built on SciServer by
``indra_grids.py`` in the deepvoid-sciserver repo.
"""

BOX_SIZE = 1000.0  # h^-1 Mpc
