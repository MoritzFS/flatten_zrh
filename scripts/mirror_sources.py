"""Mirror the Swiss source datasets into one directory, for a GitHub release.

The analysis needs two datasets whose publishers' hosts are not reachable
from every build environment: swisstopo's swissALTI3D elevation model
(data.geo.admin.ch) and the City of Zurich's statistical quarters
(ogd.stadt-zuerich.ch). This script runs in GitHub Actions, which can reach
both, and writes the files unmodified next to a manifest recording where each
one came from, when, and its checksum. The workflow
``.github/workflows/mirror-data.yml`` then attaches them to the
``source-data`` release, from which ``python -m zrh_flat_routes download``
falls back when the publisher's own host is unreachable.

It also saves the publishers' licence pages and catalogue records, so the
terms the data was taken under are on file next to it.

Usage: python scripts/mirror_sources.py OUT_DIR [--skip-alti]
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

#: Zurich city with a margin, lon/lat (the city spans 8.448-8.626 E,
#: 47.320-47.435 N).
BBOX = (8.425, 47.305, 8.650, 47.450)

STAC = "https://data.geo.admin.ch/api/stac/v0.9"
ALTI_COLLECTION = "ch.swisstopo.swissalti3d"
#: Ground sample distance of the swissALTI3D product mirrored (metres).
ALTI_GSD = "2"

#: DHM25 matrix model (25 m, interpolated from the National Map's contours,
#: so independent of the lidar behind swissALTI3D); used only as a
#: cross-check. Distributed as one national ASCII grid in LV03.
DHM25_URL = "https://cms.geo.admin.ch/ogd/topography/DHM25_MM_ASCII_GRID.zip"
#: LV03 window around Zurich (EPSG:21781 metres: xmin, ymin, xmax, ymax).
DHM25_WINDOW = (673000, 238000, 692000, 257000)

ZH_WFS = "https://www.ogd.stadt-zuerich.ch/wfs/geoportal"
ZH_LAYERS = ("Statistische_Quartiere", "Stadtkreise")

#: Catalogue records and licence pages kept for provenance.
PAGES = {
    "swisstopo_terms_en.html": "https://www.swisstopo.admin.ch/en/terms-of-use-free-geodata-and-geoservices",
    "swisstopo_source_reference_en.html": "https://www.swisstopo.admin.ch/en/source-reference-ogd-swisstopo",
    "swisstopo_ogd_conditions.html": "https://www.swisstopo.admin.ch/ogd-conditions",
    "swisstopo_swissalti3d_en.html": "https://www.swisstopo.admin.ch/en/height-model-swissalti3d",
    "swisstopo_dhm25_en.html": "https://www.swisstopo.admin.ch/en/height-model-dhm25",
    "stac_collections.json": f"{STAC}/collections?limit=1000",
    "stac_swissalti3d.json": f"{STAC}/collections/{ALTI_COLLECTION}",
    "ckan_swissalti3d.json": "https://ckan.opendata.swiss/api/3/action/package_show?id=swissalti3d",
    "ckan_dhm25.json": "https://ckan.opendata.swiss/api/3/action/package_show?id=dhm25",
    "ckan_addresses.json": "https://ckan.opendata.swiss/api/3/action/package_show?id=amtliches-verzeichnis-der-gebaudeadressen",
    "swisstopo_addresses_en.html": "https://www.swisstopo.admin.ch/en/official-directory-of-building-addresses",
    "overture_attribution.html": "https://docs.overturemaps.org/attribution/",
    "ckan_statistische_quartiere.json": "https://ckan.opendata.swiss/api/3/action/package_show?id=statistische-quartiere2",
    "ckan_search_quartiere.json": "https://ckan.opendata.swiss/api/3/action/package_search?q=statistische%20quartiere%20z%C3%BCrich&rows=20",
    "stadtzh_dataset_quartiere.html": "https://data.stadt-zuerich.ch/dataset/geo_statistische_quartiere",
    "stadtzh_ckan_quartiere.json": "https://data.stadt-zuerich.ch/api/3/action/package_show?id=geo_statistische_quartiere",
    "stadtzh_ckan_stadtkreise.json": "https://data.stadt-zuerich.ch/api/3/action/package_show?id=geo_stadtkreise",
    "opendata_swiss_terms.html": "https://opendata.swiss/en/terms-of-use",
}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "flatten_zrh source mirror (+https://github.com/MoritzFS/flatten_zrh)"
MANIFEST: list[dict] = []


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def fetch(url: str, dest: Path, retries: int = 4, **kw) -> Path:
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, timeout=(30, 300), **kw)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            MANIFEST.append({
                "file": str(dest), "url": r.url, "retrieved": now(),
                "bytes": len(r.content),
                "sha256": hashlib.sha256(r.content).hexdigest(),
                "content_type": r.headers.get("Content-Type", ""),
                "last_modified": r.headers.get("Last-Modified", ""),
            })
            return dest
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"  retry {attempt} for {url}: {exc}", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def guarded(label: str, fn, *args):
    print(f"== {label}", flush=True)
    try:
        return fn(*args)
    except Exception:
        traceback.print_exc()
        MANIFEST.append({"failed": label, "error": traceback.format_exc(limit=2)})
        return None


# --------------------------------------------------------------------------
def stac_items(collection: str, bbox) -> list[dict]:
    url = f"{STAC}/collections/{collection}/items"
    params = {"bbox": ",".join(map(str, bbox)), "limit": 100}
    items = []
    while url:
        r = SESSION.get(url, params=params, timeout=120)
        r.raise_for_status()
        page = r.json()
        items += page.get("features", [])
        url = next((l["href"] for l in page.get("links", []) if l.get("rel") == "next"), None)
        params = None
    return items


def mirror_swissalti3d(out: Path) -> None:
    items = stac_items(ALTI_COLLECTION, BBOX)
    print(f"  {len(items)} STAC items", flush=True)
    (out / "provenance").mkdir(parents=True, exist_ok=True)
    (out / "provenance" / "stac_swissalti3d_items.json").write_text(json.dumps(items, indent=1))
    # One asset per 1 km tile: the 2 m GeoTIFF of the most recent edition.
    best: dict[str, tuple[str, str, dict]] = {}
    for it in items:
        for key, a in it.get("assets", {}).items():
            if not key.endswith(".tif") or f"_{ALTI_GSD}_2056_" not in key:
                continue
            # swissalti3d_<year>_<E>-<N>_<gsd>_2056_5728.tif
            parts = key.split("_")
            year, tile = parts[1], parts[2]
            if tile not in best or year > best[tile][0]:
                best[tile] = (year, key, a)
    print(f"  {len(best)} tiles at {ALTI_GSD} m", flush=True)
    tdir = out / "swissalti3d"

    def get(entry):
        year, key, a = entry
        dest = tdir / key
        fetch(a["href"], dest)
        return key

    with ThreadPoolExecutor(max_workers=8) as pool:
        done = list(pool.map(get, best.values()))
    print(f"  downloaded {len(done)} tiles", flush=True)
    tar = out / f"swissalti3d_{ALTI_GSD}m_zurich.tar"
    with tarfile.open(tar, "w") as tf:
        for p in sorted(tdir.glob("*.tif")):
            tf.add(p, arcname=p.name)
    print(f"  wrote {tar.name} ({tar.stat().st_size / 1e6:.1f} MB)", flush=True)


def mirror_quarters(out: Path) -> None:
    for layer in ZH_LAYERS:
        base = f"{ZH_WFS}/{layer}"
        cap = out / "stadtzh" / f"{layer}_capabilities.xml"
        fetch(base, cap, params={"service": "WFS", "request": "GetCapabilities",
                                 "version": "1.1.0"})
        import re
        names = re.findall(r"<(?:wfs:)?Name>([^<]+)</(?:wfs:)?Name>", cap.read_text("utf-8", "replace"))
        print(f"  {layer}: feature types {names}", flush=True)
        for name in names:
            short = name.split(":")[-1]
            for srs, tag in (("EPSG:4326", "wgs84"), ("EPSG:2056", "lv95")):
                dest = out / "stadtzh" / f"{short}_{tag}.geojson"
                try:
                    fetch(base, dest, params={
                        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
                        "typename": name, "outputFormat": "GeoJSON", "srsName": srs})
                    head = dest.read_bytes()[:200]
                    print(f"    {dest.name}: {dest.stat().st_size} bytes, {head[:80]!r}", flush=True)
                except Exception as exc:
                    print(f"    {name} {srs}: {exc}", flush=True)


def mirror_dhm25(out: Path) -> None:
    """Clip the national DHM25 grid to Zurich and keep it as a GeoTIFF."""
    import zipfile

    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds

    z = out / "dhm25" / "DHM25_MM_ASCII_GRID.zip"
    fetch(DHM25_URL, z)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        print(f"  zip members: {names}", flush=True)
        grid = next(n for n in names if n.lower().endswith((".asc", ".agr", ".txt"))
                    and "readme" not in n.lower())
        zf.extract(grid, out / "dhm25")
        for n in names:
            if n.lower().endswith(".prj") or "readme" in n.lower() or n.lower().endswith(".pdf"):
                zf.extract(n, out / "dhm25")
    src_path = out / "dhm25" / grid
    with rasterio.open(src_path) as src:
        print(f"  {grid}: {src.width}x{src.height} res={src.res} bounds={src.bounds} crs={src.crs}",
              flush=True)
        win = from_bounds(*DHM25_WINDOW, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        transform = src.window_transform(win)
        nodata = src.nodata
    dest = out / "dhm25_zurich_lv03.tif"
    with rasterio.open(dest, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0],
                       count=1, dtype="float32", crs="EPSG:21781", transform=transform,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(arr.astype("float32"), 1)
    print(f"  wrote {dest.name} {arr.shape} range {np.nanmin(arr):.1f}-{np.nanmax(arr):.1f}",
          flush=True)
    src_path.unlink()
    z.unlink()


#: Every OpenStreetMap way in the study area that carries an ``incline`` tag.
#: Mappers mostly copy these from gradient warning signs, which makes them
#: the closest thing to published street gradients for Zurich; validation
#: compares them with the computed grades. (c) OpenStreetMap contributors,
#: ODbL 1.0.
OVERPASS = ("https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter")
INCLINE_QUERY = (
    "[out:json][timeout:180];"
    f"way[\"highway\"][\"incline\"]({BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]});"
    "out tags geom;"
)


def mirror_osm_incline(out: Path) -> None:
    last = None
    for url in OVERPASS:
        try:
            fetch(url, out / "osm_incline_zurich.json", retries=2,
                  params={"data": INCLINE_QUERY})
            n = len(json.loads((out / "osm_incline_zurich.json").read_text())["elements"])
            print(f"  {n} ways with an incline tag (from {url})", flush=True)
            return
        except Exception as exc:
            last = exc
            print(f"  {url}: {exc}", flush=True)
    raise RuntimeError(f"no Overpass endpoint answered: {last}")


#: swisstopo's official directory of building addresses (OGD). Distributed
#: for the whole country; the mirror keeps the rows inside the study area.
ADDRESS_COLLECTION = "ch.swisstopo.amtliches-gebaeudeadressverzeichnis"
#: The study area in LV95 (EPSG:2056 metres: xmin, ymin, xmax, ymax).
LV95_WINDOW = (2673000, 1238000, 2692000, 1257000)


def mirror_addresses(out: Path) -> None:
    import csv
    import io
    import zipfile

    r = SESSION.get(f"{STAC}/collections/{ADDRESS_COLLECTION}/items", timeout=120)
    r.raise_for_status()
    items = r.json()["features"]
    (out / "provenance").mkdir(parents=True, exist_ok=True)
    (out / "provenance" / "stac_addresses_items.json").write_text(json.dumps(items, indent=1))
    fetch(f"{STAC}/collections/{ADDRESS_COLLECTION}",
          out / "provenance" / "stac_addresses.json")
    assets = [(k, a) for it in items for k, a in it.get("assets", {}).items()]
    print(f"  assets: {[k for k, _ in assets]}", flush=True)
    key, asset = next((k, a) for k, a in assets
                      if "csv" in k.lower() and "2056" in k)
    z = out / "addresses" / key
    fetch(asset["href"], z)
    with zipfile.ZipFile(z) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        print(f"  {key}: members {zf.namelist()}", flush=True)
        with zf.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
            sample = text.readline()
            delim = ";" if sample.count(";") > sample.count(",") else ","
            header = next(csv.reader([sample], delimiter=delim))
            print(f"  columns: {header}", flush=True)
            ix = {c.upper(): i for i, c in enumerate(header)}
            ex = next(ix[c] for c in ("ADR_EASTING", "GKODE", "E", "EASTING") if c in ix)
            nx = next(ix[c] for c in ("ADR_NORTHING", "GKODN", "N", "NORTHING") if c in ix)
            dest = out / "swisstopo_addresses_zurich.csv"
            kept = 0
            with open(dest, "w", encoding="utf-8", newline="") as fo:
                w = csv.writer(fo, delimiter=delim)
                w.writerow(header)
                for row in csv.reader(text, delimiter=delim):
                    try:
                        e, n = float(row[ex]), float(row[nx])
                    except (ValueError, IndexError):
                        continue
                    if (LV95_WINDOW[0] <= e <= LV95_WINDOW[2]
                            and LV95_WINDOW[1] <= n <= LV95_WINDOW[3]):
                        w.writerow(row)
                        kept += 1
    print(f"  kept {kept} addresses in the study area", flush=True)
    z.unlink()


def mirror_pages(out: Path) -> None:
    for fname, url in PAGES.items():
        try:
            fetch(url, out / "provenance" / fname, retries=2)
            print(f"  ok {fname}", flush=True)
        except Exception as exc:
            print(f"  FAILED {fname}: {exc}", flush=True)
            MANIFEST.append({"failed": fname, "url": url, "error": str(exc)})


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "mirror")
    out.mkdir(parents=True, exist_ok=True)
    guarded("licence pages and catalogue records", mirror_pages, out)
    guarded("Stadt Zurich statistical quarters", mirror_quarters, out)
    if "--skip-alti" not in sys.argv:
        guarded("swissALTI3D", mirror_swissalti3d, out)
    guarded("DHM25", mirror_dhm25, out)
    guarded("OpenStreetMap incline tags", mirror_osm_incline, out)
    guarded("swisstopo building addresses", mirror_addresses, out)
    (out / "provenance").mkdir(exist_ok=True)
    (out / "provenance" / "manifest.json").write_text(json.dumps(MANIFEST, indent=1))
    with tarfile.open(out / "stadtzh_boundaries.tar.gz", "w:gz") as tf:
        if (out / "stadtzh").exists():
            tf.add(out / "stadtzh", arcname="stadtzh")
    with tarfile.open(out / "provenance.tar.gz", "w:gz") as tf:
        tf.add(out / "provenance", arcname="provenance")
    failed = [m for m in MANIFEST if "failed" in m]
    print(f"done; {len(failed)} failure(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
