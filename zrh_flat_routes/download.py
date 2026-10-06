"""Data acquisition with local caching.

Everything is cached under ``data/raw`` and re-downloaded only when missing
(or when ``force=True``), so re-running the pipeline never re-fetches the
340 MB of elevation tiles or re-scans the Overture release.

The Overture read is the interesting part: the transportation theme is tens
of gigabytes spread over a hundred-odd Parquet files.  Each file carries
per-row-group statistics on the ``bbox`` struct column, and Overture writes
rows in spatial order, so we read the footers (cheap, a couple of range
requests each), keep only the row groups whose bounding box intersects
Zurich, and read just those.

The Swiss datasets come from their publishers -- swisstopo's STAC API for
swissALTI3D, the City of Zurich's WFS for the statistical quarters -- and
fall back to the ``source-data`` release of this repository when those hosts
are unreachable (see ``sources.MIRROR_RELEASE``).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from . import sources
from .config import RAW_DIR, STUDY_BBOX
from .utils import configure_gdal_for_proxy, get_logger, human_bytes, progress, step

log = get_logger("zrh_flat_routes.download")

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

DEM_DIR = RAW_DIR / "dem"
SEGMENTS_PARQUET = RAW_DIR / "overture_segments_zrh.parquet"
CONNECTORS_PARQUET = RAW_DIR / "overture_connectors_zrh.parquet"
PLACES_PARQUET = RAW_DIR / "overture_places_zrh.parquet"
#: Overture base theme (OpenStreetMap): mapped parks, schools, stations...
BASE_PARQUETS = {typ: RAW_DIR / f"overture_{typ}_zrh.parquet"
                 for typ in ("land_use", "infrastructure", "land")}
#: Overture base-theme water: the lake and rivers, drawn on the maps.
WATER_PARQUET = RAW_DIR / "overture_water_zrh.parquet"
#: The 34 statistical quarters, in LV95 as the City publishes them.
NEIGHBORHOODS_GEOJSON = RAW_DIR / "zrh_quarters.geojson"
#: DHM25 cut to a window around Zurich (LV03); validation only.
DHM25_TIF = RAW_DIR / "dhm25_zurich_lv03.tif"
#: OpenStreetMap ways with an incline tag; validation only.
OSM_INCLINE_JSON = RAW_DIR / "osm_incline_zurich.json"
#: swisstopo's official building addresses inside the study area.
ADDRESSES_CSV = RAW_DIR / "swisstopo_addresses_zurich.csv"
MIRROR_DIR = RAW_DIR / "mirror"

#: Columns pulled from the Overture segment table. Everything unused is left
#: on the server -- the nested route/destination columns are large.
SEGMENT_COLUMNS = [
    "id", "names", "subtype", "class", "subclass", "connectors",
    "road_flags", "rail_flags", "access_restrictions", "road_surface",
    "speed_limits", "level_rules", "geometry", "bbox", "sources",
]
CONNECTOR_COLUMNS = ["id", "geometry", "bbox"]
PLACE_COLUMNS = ["id", "names", "taxonomy", "basic_category", "confidence",
                 "operating_status", "sources", "geometry", "bbox"]
BASE_COLUMNS = ["id", "names", "subtype", "class", "geometry", "bbox"]


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
def _list_s3_keys(bucket: str, prefix: str, suffix: str = ".parquet") -> list[str]:
    """List keys under a public S3 prefix via the REST list-objects-v2 API."""
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(bucket + "/", params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall("s3:Contents", _S3_NS):
            key = c.find("s3:Key", _S3_NS).text
            if key.endswith(suffix):
                keys.append(key)
        if root.findtext("s3:IsTruncated", default="false", namespaces=_S3_NS) == "true":
            token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
        else:
            break
    return keys


def _download_file(url: str, dest: Path, force: bool = False,
                   retries: int = 5, params: dict | None = None) -> Path:
    """Stream a URL to disk with retries and an atomic rename."""
    if dest.exists() and not force:
        log.info("cached %s (%s)", dest.name, human_bytes(dest.stat().st_size))
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, params=params, stream=True,
                              timeout=(30, 300)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                written = 0
                with open(tmp, "wb") as fh:
                    bar = progress(r.iter_content(chunk_size=1 << 20),
                                   desc=f"  {dest.name}",
                                   total=(total >> 20) + 1 if total else None,
                                   unit="MB")
                    for chunk in bar:
                        fh.write(chunk)
                        written += len(chunk)
            if total and written < total:
                raise IOError(f"short read: {written} of {total} bytes")
            tmp.replace(dest)
            log.info("downloaded %s (%s)", dest.name, human_bytes(dest.stat().st_size))
            return dest
        except Exception as exc:  # network flakiness is expected
            tmp.unlink(missing_ok=True)
            if attempt == retries or _blocked(exc):
                raise
            import time
            log.warning("attempt %d/%d for %s failed (%s); retrying in %.0fs",
                        attempt, retries, dest.name, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _blocked(exc: Exception) -> bool:
    """A refusal no retry will fix: proxy denial, 4xx, unreachable host."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return 400 <= exc.response.status_code < 500
    return isinstance(exc, (requests.exceptions.ProxyError,
                            requests.exceptions.SSLError))


def _from_mirror(asset: str, force: bool = False) -> Path:
    """Fetch one asset of the ``source-data`` release (see ``sources``)."""
    return _download_file(f"{sources.MIRROR_RELEASE}/{asset}", MIRROR_DIR / asset,
                          force=force)


# --------------------------------------------------------------------------
# Overture
# --------------------------------------------------------------------------
def _bbox_stat_columns(metadata) -> dict[str, int]:
    cols = {c["path_in_schema"]: i
            for i, c in enumerate(metadata.row_group(0).to_dict()["columns"])}
    return {k: cols[k] for k in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}


def _matching_row_groups(metadata, bbox) -> list[int]:
    lon_min, lon_max, lat_min, lat_max = bbox
    ix = _bbox_stat_columns(metadata)
    out = []
    for rg in range(metadata.num_row_groups):
        g = metadata.row_group(rg)
        xmax = g.column(ix["bbox.xmax"]).statistics.max
        xmin = g.column(ix["bbox.xmin"]).statistics.min
        ymax = g.column(ix["bbox.ymax"]).statistics.max
        ymin = g.column(ix["bbox.ymin"]).statistics.min
        if xmax >= lon_min and xmin <= lon_max and ymax >= lat_min and ymin <= lat_max:
            out.append(rg)
    return out


def _read_overture_type(overture_type: str, columns: list[str], dest: Path,
                        bbox=STUDY_BBOX, force: bool = False,
                        prefix: str = sources.OVERTURE_PREFIX) -> Path:
    """Row-group-pruned read of one Overture type (any theme)."""
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    if dest.exists() and not force:
        log.info("cached %s (%s)", dest.name, human_bytes(dest.stat().st_size))
        return dest

    keys = _list_s3_keys(sources.OVERTURE_BUCKET, f"{prefix}/type={overture_type}/")
    log.info("overture %s: %d parquet files in release %s",
             overture_type, len(keys), sources.OVERTURE_RELEASE)

    fs = fsspec.filesystem("https")

    def scan(key: str):
        url = f"{sources.OVERTURE_BUCKET}/{key}"
        with fs.open(url, block_size=8 << 20) as fh:
            md = pq.ParquetFile(fh).metadata
            return key, _matching_row_groups(md, bbox)

    with step(f"scanning {len(keys)} {overture_type} footers for the Zurich bbox", log):
        with ThreadPoolExecutor(max_workers=16) as pool:
            scanned = list(pool.map(scan, keys))
    hits = [(k, rgs) for k, rgs in scanned if rgs]
    n_rg = sum(len(r) for _, r in hits)
    log.info("overture %s: %d row group(s) in %d file(s) intersect Zurich",
             overture_type, n_rg, len(hits))
    if not hits:
        raise RuntimeError(f"no Overture {overture_type} row groups intersect {bbox}")

    tables = []
    for key, rgs in progress(hits, desc=f"  reading {overture_type}", unit="file"):
        url = f"{sources.OVERTURE_BUCKET}/{key}"
        with fs.open(url, block_size=16 << 20) as fh:
            t = pq.ParquetFile(fh).read_row_groups(rgs, columns=columns)
        tables.append(t)
    table = pa.concat_tables(tables)

    # Row groups are coarse; clip precisely to the study bbox.
    bb = table.column("bbox").combine_chunks()
    xmin = np.asarray(bb.field("xmin")); xmax = np.asarray(bb.field("xmax"))
    ymin = np.asarray(bb.field("ymin")); ymax = np.asarray(bb.field("ymax"))
    lon_min, lon_max, lat_min, lat_max = bbox
    keep = ((xmax >= lon_min) & (xmin <= lon_max)
            & (ymax >= lat_min) & (ymin <= lat_max))
    table = table.filter(pa.array(keep))
    log.info("overture %s: %d features inside the study bbox", overture_type,
             table.num_rows)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(dest)
    log.info("wrote %s (%s)", dest.name, human_bytes(dest.stat().st_size))
    return dest


def download_street_network(force: bool = False) -> tuple[Path, Path]:
    seg = _read_overture_type("segment", SEGMENT_COLUMNS, SEGMENTS_PARQUET, force=force)
    con = _read_overture_type("connector", CONNECTOR_COLUMNS, CONNECTORS_PARQUET,
                              force=force)
    return seg, con


def download_places(force: bool = False) -> tuple[Path, Path]:
    """Places and addresses for the route page's offline search.

    Neither is needed by the analysis; the route page degrades to
    intersection-only search when they are missing.
    """
    places = _read_overture_type("place", PLACE_COLUMNS, PLACES_PARQUET,
                                 force=force, prefix=sources.OVERTURE_PLACES_PREFIX)
    for typ, dest in list(BASE_PARQUETS.items()) + [("water", WATER_PARQUET)]:
        _read_overture_type(typ, BASE_COLUMNS, dest, force=force,
                            prefix=sources.OVERTURE_BASE_PREFIX)
    return places, download_addresses(force=force)


def download_addresses(force: bool = False) -> Path:
    """swisstopo's building-address register, cut to the study area."""
    import csv
    import io
    import zipfile

    if ADDRESSES_CSV.exists() and not force:
        log.info("cached %s", ADDRESSES_CSV.name)
        return ADDRESSES_CSV
    try:
        r = requests.get(f"{sources.STAC_ROOT}/collections/"
                         f"{sources.ADDRESSES_COLLECTION}/items", timeout=60)
        r.raise_for_status()
        href = next(a["href"] for it in r.json()["features"]
                    for k, a in it["assets"].items()
                    if "csv" in k.lower() and "2056" in k)
        z = _download_file(href, RAW_DIR / "addresses_ch.csv.zip", retries=2)
        (x0, y0, x1, y1) = sources.LV95_WINDOW
        with zipfile.ZipFile(z) as zf, \
                open(ADDRESSES_CSV, "w", encoding="utf-8", newline="") as fo:
            member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(member) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
                first = text.readline()
                delim = ";" if first.count(";") > first.count(",") else ","
                header = next(csv.reader([first], delimiter=delim))
                ix = {c.upper(): i for i, c in enumerate(header)}
                ex, nx = ix["ADR_EASTING"], ix["ADR_NORTHING"]
                w = csv.writer(fo, delimiter=delim)
                w.writerow(header)
                for row in csv.reader(text, delimiter=delim):
                    try:
                        e, n = float(row[ex]), float(row[nx])
                    except (ValueError, IndexError):
                        continue
                    if x0 <= e <= x1 and y0 <= n <= y1:
                        w.writerow(row)
        z.unlink()
        return ADDRESSES_CSV
    except Exception as exc:
        log.warning("swisstopo address register unavailable (%s); using the "
                    "source-data mirror", exc)
        return _from_mirror(ADDRESSES_CSV.name, force=force).replace(ADDRESSES_CSV)


# --------------------------------------------------------------------------
# Elevation
# --------------------------------------------------------------------------
def _stac_items(collection: str, bbox) -> list[dict]:
    lon_min, lon_max, lat_min, lat_max = bbox
    url = f"{sources.STAC_ROOT}/collections/{collection}/items"
    params = {"bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}", "limit": 100}
    items: list[dict] = []
    while url:
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        page = r.json()
        items += page.get("features", [])
        url = next((ln["href"] for ln in page.get("links", [])
                    if ln.get("rel") == "next"), None)
        params = None
    return items


def swissalti3d_assets(items: list[dict], gsd: str = sources.SWISSALTI3D_GSD) -> dict:
    """One GeoTIFF asset per 1 km tile: the most recent edition at ``gsd`` m.

    Asset keys look like ``swissalti3d_2026_2683-1247_2_2056_5728.tif``.
    """
    best: dict[str, tuple[str, str, str]] = {}
    for it in items:
        for key, a in it.get("assets", {}).items():
            if not key.endswith(".tif") or f"_{gsd}_2056_" not in key:
                continue
            _, year, tile = key.split("_")[:3]
            if tile not in best or year > best[tile][0]:
                best[tile] = (year, key, a["href"])
    return {key: href for _, key, href in best.values()}


def download_dem(force: bool = False, include_dhm25: bool = True) -> list[Path]:
    """Fetch the swissALTI3D 2 m tiles that cover the study area."""
    import tarfile

    DEM_DIR.mkdir(parents=True, exist_ok=True)
    have = sorted(DEM_DIR.glob("swissalti3d_*.tif"))
    if have and not force:
        log.info("cached %d swissALTI3D tiles", len(have))
    else:
        try:
            assets = swissalti3d_assets(
                _stac_items(sources.SWISSALTI3D_COLLECTION, STUDY_BBOX))
            log.info("swissALTI3D: %d tiles at %s m", len(assets),
                     sources.SWISSALTI3D_GSD)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda kv: _download_file(kv[1], DEM_DIR / kv[0], force=force),
                              assets.items()))
        except Exception as exc:
            log.warning("swisstopo STAC unreachable (%s); using the source-data "
                        "mirror", exc)
            tar = _from_mirror(f"swissalti3d_{sources.SWISSALTI3D_GSD}m_zurich.tar")
            with tarfile.open(tar) as tf:
                tf.extractall(DEM_DIR, filter="data")
    if include_dhm25:
        try:
            download_dhm25(force=force)
        except Exception as exc:
            log.warning("optional DHM25 cross-check unavailable (%s)", exc)
    return sorted(DEM_DIR.glob("swissalti3d_*.tif"))


def download_dhm25(force: bool = False) -> Path:
    """DHM25 around Zurich: the national grid, cut to a window (LV03)."""
    if DHM25_TIF.exists() and not force:
        log.info("cached %s", DHM25_TIF.name)
        return DHM25_TIF
    try:
        import zipfile

        import rasterio
        from rasterio.windows import from_bounds

        z = _download_file(sources.DHM25_URL, RAW_DIR / "DHM25_MM_ASCII_GRID.zip",
                           retries=2)
        with zipfile.ZipFile(z) as zf:
            grid = next(n for n in zf.namelist()
                        if n.lower().endswith((".asc", ".agr", ".txt"))
                        and "readme" not in n.lower())
            zf.extract(grid, RAW_DIR / "dhm25")
        src_path = RAW_DIR / "dhm25" / grid
        with rasterio.open(src_path) as src:
            win = from_bounds(673000, 238000, 692000, 257000,
                              transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win)
            transform, nodata = src.window_transform(win), src.nodata
        with rasterio.open(DHM25_TIF, "w", driver="GTiff", width=arr.shape[1],
                           height=arr.shape[0], count=1, dtype="float32",
                           crs="EPSG:21781", transform=transform, nodata=nodata,
                           compress="deflate") as dst:
            dst.write(arr.astype("float32"), 1)
        src_path.unlink()
        z.unlink()
    except Exception as exc:
        log.warning("DHM25 from swisstopo unavailable (%s); using the mirror", exc)
        _from_mirror(DHM25_TIF.name).replace(DHM25_TIF)
    return DHM25_TIF


# --------------------------------------------------------------------------
# Quarters
# --------------------------------------------------------------------------
def download_neighborhoods(force: bool = False) -> Path:
    """The City of Zurich's 34 statistical quarters, in LV95."""
    import tarfile

    if NEIGHBORHOODS_GEOJSON.exists() and not force:
        log.info("cached %s", NEIGHBORHOODS_GEOJSON.name)
        return NEIGHBORHOODS_GEOJSON
    try:
        return _download_file(sources.QUARTERS_WFS, NEIGHBORHOODS_GEOJSON, force=force,
                              retries=2, params={
                                  "service": "WFS", "version": "1.1.0",
                                  "request": "GetFeature",
                                  "typename": sources.QUARTERS_TYPENAME,
                                  "outputFormat": "GeoJSON",
                                  "srsName": "EPSG:2056"})
    except Exception as exc:
        log.warning("Stadt Zurich WFS unreachable (%s); using the source-data "
                    "mirror", exc)
        tar = _from_mirror("stadtzh_boundaries.tar.gz", force=force)
        member = f"stadtzh/{sources.QUARTERS_TYPENAME}_lv95.geojson"
        with tarfile.open(tar) as tf:
            data = tf.extractfile(member).read()
        NEIGHBORHOODS_GEOJSON.write_bytes(data)
        return NEIGHBORHOODS_GEOJSON


# --------------------------------------------------------------------------
# OpenStreetMap incline tags (validation only)
# --------------------------------------------------------------------------
def download_osm_incline(force: bool = False) -> Path:
    if OSM_INCLINE_JSON.exists() and not force:
        log.info("cached %s", OSM_INCLINE_JSON.name)
        return OSM_INCLINE_JSON
    lon_min, lon_max, lat_min, lat_max = STUDY_BBOX
    query = ("[out:json][timeout:180];"
             f'way["highway"]["incline"]({lat_min},{lon_min},{lat_max},{lon_max});'
             "out tags geom;")
    for url in sources.OVERPASS_URLS:
        try:
            return _download_file(url, OSM_INCLINE_JSON, force=force, retries=2,
                                  params={"data": query})
        except Exception as exc:
            log.warning("Overpass %s unavailable (%s)", url, exc)
    return _from_mirror(OSM_INCLINE_JSON.name, force=force).replace(OSM_INCLINE_JSON)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def download_all(force: bool = False) -> dict[str, object]:
    configure_gdal_for_proxy()
    out: dict[str, object] = {}
    with step("downloading street network (Overture)", log):
        out["segments"], out["connectors"] = download_street_network(force=force)
    with step("downloading the statistical quarters (Stadt Zurich)", log):
        out["neighborhoods"] = download_neighborhoods(force=force)
    with step("downloading places and addresses (Overture)", log):
        try:
            out["places"], out["addresses"] = download_places(force=force)
        except Exception as exc:  # optional: the route page can do without
            log.warning("places/addresses unavailable (%s); the route page "
                        "will offer intersection search only", exc)
    with step("downloading swisstopo elevation (swissALTI3D, DHM25)", log):
        out["dem"] = download_dem(force=force)
    with step("downloading OpenStreetMap incline tags", log):
        try:
            out["osm_incline"] = download_osm_incline(force=force)
        except Exception as exc:  # optional: validation skips the check
            log.warning("OpenStreetMap incline tags unavailable (%s)", exc)
    return out
