# Vendored third-party assets

`leaflet-1.9.4.min.js` and `leaflet-1.9.4.css` are Leaflet 1.9.4, obtained
from the npm registry (`https://registry.npmjs.org/leaflet/-/leaflet-1.9.4.tgz`,
retrieved upstream 2026-09-16, checked byte-identical to the registry's
tarball 2026-10-06) and redistributed unmodified under the BSD 2-Clause
licence (Copyright (c) 2010-2023 Volodymyr Agafonkin, (c) 2010-2011
CloudMade). The licence text is `LICENSE-leaflet.txt`, from the same
tarball; the build copies it into `site/` beside the hashed Leaflet files.

They are inlined into `outputs/zrh_flat_routes_map.html` and
`outputs/zrh_flat_route_finder.html` so that those pages are genuinely
self-contained files: it opens and works with
no CDN and no network access, apart from the optional raster basemap tiles.
