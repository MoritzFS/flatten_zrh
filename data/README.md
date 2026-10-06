# Data directory

This directory is populated by the pipeline and is **not** tracked in git.

- `raw/` — cached source downloads, never modified after they land: the
  Overture Maps street, place and base-theme extracts, the 316 swisstopo
  swissALTI3D 2 m tiles (`dem/`), the DHM25 cross-check window, the City of
  Zurich's statistical quarters, swisstopo's building addresses inside the
  study area, and the OpenStreetMap incline tags used in validation.
  Populate with `python -m zrh_flat_routes download` (~400 MB). Files the
  publishers' hosts will not serve to the build environment come from this
  repository's `source-data` release instead (`raw/mirror/`), which
  `.github/workflows/mirror-data.yml` fills from the publishers.
- `processed/` — cached intermediate products: the clipped DEM mosaic, the
  edge table, per-edge elevation profiles, the directed edge metrics, the
  quarter pair matrix and the route store.
  Populate with `python -m zrh_flat_routes build-network` and
  `python -m zrh_flat_routes analyze`.

Nothing here needs to be kept: every file is rebuilt deterministically from
the sources listed by `python -m zrh_flat_routes sources`.
