# Data directory

This directory is populated by the pipeline and is **not** tracked in git.

- `raw/` — cached source downloads, never modified after they land:
  the Overture Maps street extract, the four USGS 3DEP 1 m lidar tiles, the
  1/3 arc-second cross-check tile, and the neighborhood boundaries.
  Populate with `python -m sf_flat_routes download` (~725 MB).
- `processed/` — cached intermediate products: the clipped DEM mosaic, the
  edge table, per-edge elevation profiles, the directed edge metrics, the
  neighborhood pair matrix and the route store.
  Populate with `python -m sf_flat_routes build-network` and
  `python -m sf_flat_routes analyze`.

Nothing here needs to be kept: every file is rebuilt deterministically from
the sources listed by `python -m sf_flat_routes sources`.
