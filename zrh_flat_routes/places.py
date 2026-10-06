"""Offline place index for the route page: addresses, places, and the base.

A static page cannot call a geocoding API without carrying a billed key in
public, and should not need one for a single city anyway, so the index is
built here and shipped with the page.
Three sources, all already in hand or fetched the same way as the streets:

* **Street intersections** are derived in the browser from the graph itself
  ("Langstrasse & Josefstrasse"); nothing to pack.
* **Mapped features** come from Overture's base theme, which is OpenStreetMap
  data: parks, playgrounds, schools, hospitals, plazas, stations, piers,
  bridges, viewpoints, peaks and beaches, each placed on its mapped outline.
  These are authoritative and are never pruned.
* **Places** come from Overture's places theme (Meta, Microsoft, Foursquare,
  AllThePlaces and other POI data):
  landmarks, museums, shops, cafes and so on. The feed is noisy -- the same
  name recurs at several spots, some of them nowhere near the real thing --
  so a record is kept only where nearby records corroborate it, and it is
  dropped when a mapped feature already carries its name.
* **Addresses** come from swisstopo's official directory of building
  addresses (Amtliches Verzeichnis der Gebäudeadressen, open government
  data), deduplicated to one point per street number. Overture distributes
  the same register, but its records label the licence only as proprietary,
  so it is taken from swisstopo directly.

The hillshade base is rendered from the same swissALTI3D DEM the analysis
uses and reprojected to WGS84 so it overlays correctly, then
palette-quantised: it is a quiet grey image and does not need 24-bit colour.
"""
from __future__ import annotations

import base64
import io
import re

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, STUDY_BBOX
from .download import ADDRESSES_CSV, BASE_PARQUETS, PLACES_PARQUET
from .utils import get_logger, step

log = get_logger("zrh_flat_routes.places")

HILLSHADE_PNG = PROCESSED_DIR / "hillshade_light.png"
#: Elevation (m) at which the hillshade tint starts, and the rise over which
#: it reaches full strength (upstream used sea level and 260 m for SF).
HILLSHADE_BASE_M = 392.0
HILLSHADE_SPAN_M = 300.0
#: Colour of the lake and rivers on the hillshade.
WATER_RGB = (214, 226, 234)

#: Overture primary categories kept, grouped for display. Anything not listed
#: is dropped unless it is a landmark-like category matched by _KEEP_RE.
CATEGORY_GROUPS = {
    "park": ["park", "garden", "playground", "beach", "dog_park", "hiking_trail",
             "scenic_point", "nature_preserve", "botanical_garden", "national_park",
             "state_park", "plaza", "picnic_ground"],
    "landmark": ["landmark_and_historical_building", "monument", "tourist_attraction",
                 "museum", "art_museum", "history_museum", "science_museum",
                 "aquarium", "zoo", "stadium_arena", "observatory", "lighthouse",
                 "historical_site", "memorial", "pier", "marina", "amusement_park",
                 "theatre", "performing_arts_theatre", "concert_hall", "cinema",
                 "library", "public_library"],
    "transit": ["train_station", "light_rail_station", "subway_station",
                "bus_station", "ferry_terminal", "transit_station", "cable_car_station",
                "metro_station", "public_transportation"],
    "school": ["school", "university", "college_university", "high_school",
               "elementary_school", "middle_school", "community_college", "campus"],
    "civic": ["city_hall", "courthouse", "post_office", "hospital", "fire_station",
              "police_station", "community_center", "recreation_center",
              "swimming_pool", "public_swimming_pool", "church_cathedral", "synagogue",
              "mosque", "temple", "farmers_market", "public_market"],
    "food": ["restaurant", "cafe", "coffee_shop", "bakery", "bar", "pub", "brewery",
             "ice_cream_shop", "pizza_restaurant", "taco_restaurant", "diner",
             "dessert_shop", "tea_room", "wine_bar", "cocktail_bar", "food_court"],
    "shop": ["grocery_store", "supermarket", "bookstore", "shopping_center",
             "hardware_store", "bicycle_shop", "pharmacy", "farmers_market",
             "convenience_store", "department_store", "record_store", "florist"],
    "lodging": ["hotel", "hostel", "bed_and_breakfast"],
}
_GROUP_OF = {c: g for g, cs in CATEGORY_GROUPS.items() for c in cs}

#: Overture base-theme (OpenStreetMap) classes kept, with the kind shown in
#: the search list. Keyed by (type, class).
BASE_CLASSES = {
    ("land_use", "park"): "park", ("land_use", "dog_park"): "dog park",
    ("land_use", "playground"): "playground", ("land_use", "garden"): "garden",
    ("land_use", "allotments"): "community garden", ("land_use", "plaza"): "plaza",
    ("land_use", "pedestrian"): "plaza", ("land_use", "school"): "school",
    ("land_use", "kindergarten"): "school", ("land_use", "university"): "university",
    ("land_use", "college"): "college", ("land_use", "hospital"): "hospital",
    ("land_use", "golf_course"): "golf course", ("land_use", "stadium"): "stadium",
    ("land_use", "marina"): "marina", ("land_use", "recreation_ground"): "park",
    ("land_use", "national_park"): "park", ("land_use", "military"): "landmark",
    ("land_use", "protected_landscape_seascape"): "park",
    # Zurich navigates by tram and bus stop: "meet at Bellevue" means the
    # stop, and the POI feed's own "Bellevue" is a restaurant in Affoltern
    ("infrastructure", "platform"): "tram/bus stop",
    ("infrastructure", "bus_stop"): "tram/bus stop",
    ("infrastructure", "stop_position"): "tram/bus stop",
    ("infrastructure", "railway_station"): "station",
    ("infrastructure", "subway_station"): "station",
    ("infrastructure", "ferry_terminal"): "ferry", ("infrastructure", "pier"): "pier",
    ("infrastructure", "bridge"): "bridge", ("infrastructure", "viewpoint"): "viewpoint",
    ("infrastructure", "observation"): "landmark",
    ("infrastructure", "communication_tower"): "landmark",
    ("land", "peak"): "peak", ("land", "hill"): "hill", ("land", "beach"): "beach",
    ("land", "island"): "island", ("land", "islet"): "island",
}
_KEEP_RE = re.compile(r"park|garden|museum|station|landmark|monument|library|"
                      r"theat|school|universit|college|church|cathedral|hospital|"
                      r"beach|plaza|square|pier|market|stadium|trail|overlook", re.I)

def _support(names: pd.Series, lon: np.ndarray, lat: np.ndarray,
             all_names: pd.Series, all_lon: np.ndarray, all_lat: np.ndarray,
             radius_m: float = 300.0) -> np.ndarray:
    """How many nearby records mention each name.

    A real landmark is surrounded by records that borrow its name (upstream's
    example: 'Dolores Park' amid 'Dolores Park Cafe', 'Dolores Park Tennis
    Courts'...); a stray copy of the name dropped elsewhere has none of that.
    The feed carries such strays with full confidence, so the name alone
    cannot pick the right one.
    """
    cell = radius_m / 111000.0
    grid: dict[tuple[int, int], list[int]] = {}
    low = all_names.str.lower().to_numpy()
    for i, (x, y) in enumerate(zip(all_lon, all_lat)):
        grid.setdefault((int(x / cell), int(y / cell)), []).append(i)
    out = np.zeros(len(names), dtype=int)
    for k, (n, x, y) in enumerate(zip(names.str.lower(), lon, lat)):
        cx, cy = int(x / cell), int(y / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    o = low[j]
                    if o != n and n in o and abs(all_lon[j] - x) * 88000 < radius_m \
                            and abs(all_lat[j] - y) * 111000 < radius_m:
                        out[k] += 1
    return out


_CITY_SUFFIXES = {"zürich", "zurich", "zuerich", "zh", "zürich zh", "zurich zh",
                  "zürich ch", "zurich ch", "zürich schweiz", "zurich switzerland",
                  "schweiz", "switzerland", "ch"}
# 'X Zürich' and 'X Zurich' are city suffixes even without a comma ('Kunsthaus
# Zürich' is the Kunsthaus); a bare trailing 'ZH', 'CH' or 'Schweiz' only
# after one.
_CORE_RE = re.compile(
    r"(?:[\s,\-/]+(?:zürich|zurich|zuerich)(?:[\s,]+(?:zh|ch|schweiz|switzerland))?"
    r"|[,\-/]\s*(?:zh|ch|schweiz|switzerland))\s*$", re.I)


def _fold(s: str) -> str:
    """Lower case without diacritics, as the page's search folds text:
    'Zürich', 'Zurich' and 'Zuerich' all become 'zurich'."""
    import unicodedata
    s = unicodedata.normalize("NFD", str(s).lower().replace("ß", "ss"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"([aou])e", r"\1", s)


def _core(name: str) -> str:
    """'Kunsthaus Zürich' -> 'kunsthaus'; folded, so that the POI feed's
    'Zurich HB' and the mapped 'Zürich HB' are recognised as one name."""
    return _fold(_CORE_RE.sub("", str(name)).strip())


def _prune_variants(df: pd.DataFrame, radius_m: float = 500.0) -> pd.DataFrame:
    """Drop 'Lindenhof, Zürich' when 'Lindenhof' is 200 m away.

    The places feed carries many user-typed variants of the same name. A
    record is dropped when a better-supported kept name is a prefix of it
    (at a word boundary) and the two points are within ``radius_m``.
    """
    df = df.sort_values(["support", "conf"], ascending=False).reset_index(drop=True)
    kept_idx = []
    by_first: dict[str, list[int]] = {}
    lon = df["lon"].to_numpy(); lat = df["lat"].to_numpy()
    names = df["name"].tolist(); groups = df["group"].tolist()
    support = df["support"].to_numpy()
    for i, n in enumerate(names):
        # split on punctuation too: the first word of 'Platzspitz, Zürich'
        # is 'platzspitz', not 'platzspitz,'
        first = re.split(r"[\s,\-/(]+", n.strip())[0].lower()
        dup = False
        for j in by_first.get(first, ()):
            k = names[j]
            if not (n.startswith(k) and len(n) > len(k) and n[len(k)] in " ,-/("):
                continue
            # a city suffix never marks a different place; anything else
            # (a branch, a sub-area) only when it is close by
            rest = n[len(k):].strip(" ,-/()").lower()
            if rest in _CITY_SUFFIXES:
                r = 6000.0                       # 'X, Zürich' anywhere
            elif groups[i] != groups[j]:
                continue                         # 'Lindenhof Cafe' is a cafe
            elif support[j] >= 3 and support[i] == 0:
                r = 6000.0                       # a same-kind variant of a well-known name
            else:
                r = radius_m
            if (abs(lon[i] - lon[j]) * 88000 < r
                    and abs(lat[i] - lat[j]) * 111000 < r):
                dup = True
                break
        if not dup:
            kept_idx.append(i)
            by_first.setdefault(first, []).append(i)
    out = df.iloc[kept_idx].reset_index(drop=True)
    log.info("places: %d near-duplicate name variants pruned", len(df) - len(out))
    return out


def build_base() -> pd.DataFrame:
    """Named OpenStreetMap features from the Overture base theme."""
    import pyarrow.parquet as pq
    import shapely
    frames = []
    for typ, path in BASE_PARQUETS.items():
        if not path.exists():
            continue
        t = pq.read_table(path, columns=["names", "class", "geometry"]).to_pandas()
        name = t["names"].map(lambda n: (n or {}).get("primary") if isinstance(n, dict) else None)
        kind = [BASE_CLASSES.get((typ, c)) for c in t["class"]]
        keep = name.notna() & pd.Series(kind, index=t.index).notna()
        geom = shapely.from_wkb(t.loc[keep, "geometry"].values)
        pts = shapely.get_coordinates(shapely.point_on_surface(geom)) if len(geom) else np.zeros((0, 2))
        area = np.where(shapely.get_type_id(geom) >= 3, shapely.area(geom), 0.0) if len(geom) else []
        frames.append(pd.DataFrame({
            "name": name[keep].to_numpy(), "group": np.asarray(kind, dtype=object)[keep.to_numpy()],
            "lon": pts[:, 0], "lat": pts[:, 1], "area": area,
        }))
    if not frames:
        return pd.DataFrame(columns=["name", "group", "lon", "lat", "area"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["name"].str.len() >= 3]
    # a stop is mapped once per platform and once per stopping position:
    # stand each name at the median of its points, so the pin lands between
    # the platforms rather than on one of them
    stops = df["group"] == "tram/bus stop"
    if stops.any():
        med = (df[stops].groupby("name")[["lon", "lat"]].median()
               .reset_index().assign(group="tram/bus stop", area=0.0))
        df = pd.concat([df[~stops], med], ignore_index=True)
    # a bridge is mapped once per carriageway and a park once per polygon
    # ring: keep the largest outline under each name
    df = (df.sort_values("area", ascending=False)
            .drop_duplicates(["name", "group"]).reset_index(drop=True))
    log.info("base features: %d named (%s)", len(df),
             ", ".join(f"{g} {n}" for g, n in df["group"].value_counts().head(8).items()))
    return df


def build_places() -> dict:
    """Compact place list: names, display kind, coordinates."""
    import pyarrow.parquet as pq
    base = build_base()
    t = pq.read_table(PLACES_PARQUET).to_pandas()
    names = t["names"].map(lambda n: (n or {}).get("primary") if isinstance(n, dict) else None)
    # Overture's places schema files a record under ``taxonomy.primary``
    # (formerly ``categories.primary``, with the same vocabulary)
    cats = t["taxonomy"].map(lambda c: (c or {}).get("primary") if isinstance(c, dict) else None)
    if "operating_status" in t.columns:
        closed = t["operating_status"].fillna("open").ne("open")
        cats = cats.where(~closed, None)
    conf = pd.to_numeric(t["confidence"], errors="coerce").fillna(0)
    import shapely
    geom = shapely.from_wkb(t["geometry"].values)
    lon = np.array([g.x for g in geom]); lat = np.array([g.y for g in geom])

    group = cats.map(lambda c: (_GROUP_OF.get(c) or ("landmark" if _KEEP_RE.search(c) else None)) if isinstance(c, str) else None)
    support = _support(names.fillna(""), lon, lat, names.fillna(""), lon, lat)
    keep = names.notna() & (names.str.len() >= 3) & group.notna() & (conf >= 0.6)
    # a famous place with no useful category still deserves a slot, and so
    # does anything the records around it keep mentioning (upstream's
    # example: San Francisco's Ferry Building is filed under farming
    # services, after its market)
    keep |= names.notna() & cats.isna() & (conf >= 0.9)
    keep |= names.notna() & (support >= 5) & (conf >= 0.6)
    # Foursquare's records come under Apache 2.0, whose notice obligations
    # the page does not carry; the rest of the feed is CDLA Permissive 2.0
    # or CC0. In Zurich every Foursquare record has no other source.
    if "sources" in t.columns:
        fsq = t["sources"].map(lambda ss: any((e or {}).get("dataset") == "Foursquare"
                                              for e in (ss if ss is not None else [])))
        keep &= ~fsq.to_numpy()
        log.info("places: %d Foursquare records left out", int(fsq.sum()))
    df = pd.DataFrame({"name": names, "group": group.fillna("landmark"),
                       "conf": conf, "lon": lon, "lat": lat, "support": support})[keep]
    df = df[(df["lon"].between(STUDY_BBOX[0], STUDY_BBOX[1]))
            & (df["lat"].between(STUDY_BBOX[2], STUDY_BBOX[3]))]
    df = df.sort_values(["support", "conf"], ascending=False)
    # one record per name and kind, and 'Zurich HB' is 'Zürich HB': keep the
    # best-supported record's position under the properly spelled name
    df["core"] = df["name"].map(_fold)
    proper = (df[df["name"].str.contains(r"[äöüÄÖÜàéèç]", regex=True)]
              .drop_duplicates(["core", "group"])
              .set_index(["core", "group"])["name"])
    df = df.drop_duplicates(["core", "group"]).reset_index(drop=True)
    better = [proper.get((c, g)) for c, g in zip(df["core"], df["group"])]
    df["name"] = [b or n for b, n in zip(better, df["name"])]
    df = df.drop(columns="core")
    # the mapped feature wins over any POI record of the same name, or of a
    # trailing part of it ('Rieterpark' for 'Museum Rietberg Rieterpark')
    mapped = set(_core(n) for n in base["name"])
    tails = set()
    for n in mapped:
        words = n.split()
        for k in range(1, len(words)):
            tail = " ".join(words[k:])
            if len(tail) >= 8:
                tails.add(tail)
    def superseded(n: str) -> bool:
        c = _core(n)
        head = re.split(r"\s*[,\-/(]\s*", c, maxsplit=1)[0]   # 'Landesmuseum, Platzspitz'
        return c in mapped or c in tails or head in mapped or head in tails
    dup = df["name"].map(superseded)
    log.info("places: %d records superseded by mapped features", int(dup.sum()))
    df = _prune_variants(df[~dup].reset_index(drop=True))
    df = pd.concat([base[["name", "group", "lon", "lat"]], df[["name", "group", "lon", "lat"]]],
                   ignore_index=True)
    df = df[(df["lon"].between(STUDY_BBOX[0], STUDY_BBOX[1]))
            & (df["lat"].between(STUDY_BBOX[2], STUDY_BBOX[3]))]
    df = df.sort_values(["name"]).reset_index(drop=True)
    log.info("places: %d kept of %d POI records plus %d mapped features (%s)",
             len(df) - len(base), len(t), len(base),
             ", ".join(f"{g} {n}" for g, n in df["group"].value_counts().head(12).items()))
    groups = sorted(set(df["group"]))
    return {
        "names": df["name"].tolist(),
        "group": [groups.index(g) for g in df["group"]],
        "groups": groups,
        "lon": np.round(df["lon"].to_numpy(), 5).tolist(),
        "lat": np.round(df["lat"].to_numpy(), 5).tolist(),
    }


def _address_columns(header: list[str]) -> dict:
    """Locate the street, number and LV95 coordinate columns of the register."""
    up = {c.upper(): c for c in header}

    def pick(*cands):
        return next((up[c] for c in cands if c in up), None)
    cols = {"street": pick("STN_LABEL", "STRNAME", "STREET"),
            "number": pick("ADR_NUMBER", "DEINR", "NUMBER"),
            "e": pick("ADR_EASTING", "GKODE", "DKODE", "EASTING"),
            "n": pick("ADR_NORTHING", "GKODN", "DKODN", "NORTHING"),
            "status": pick("ADR_STATUS", "STATUS")}
    missing = [k for k, v in cols.items() if v is None and k != "status"]
    if missing:
        raise KeyError(f"address register lacks {missing}: {header}")
    return cols


def build_addresses() -> dict:
    """One point per (street, number), with a street table.

    Swiss house numbers carry letters and sub-numbers ('12a', '4.1'); the
    index keeps the leading integer, which is what anyone types first, and
    the point of the lowest-suffixed entrance under it.
    """
    from pyproj import Transformer

    t = pd.read_csv(ADDRESSES_CSV, sep=None, engine="python", dtype=str)
    c = _address_columns(list(t.columns))
    if c["status"]:
        # planned and withdrawn addresses are not places anyone is going
        t = t[~t[c["status"]].fillna("").str.lower().isin({"planned", "outdated",
                                                           "geplant", "aufgehoben"})]
    t = t[t[c["street"]].notna() & t[c["number"]].notna()].copy()
    num = pd.to_numeric(t[c["number"]].str.extract(r"^(\d+)")[0], errors="coerce")
    ok = num.notna() & (num < 65536)
    t = t[ok].copy(); t["num"] = num[ok].astype(int)
    tr = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(pd.to_numeric(t[c["e"]]).to_numpy(),
                            pd.to_numeric(t[c["n"]]).to_numpy())
    t["lon"], t["lat"] = lon, lat
    t = t[t["lon"].between(STUDY_BBOX[0], STUDY_BBOX[1])
          & t["lat"].between(STUDY_BBOX[2], STUDY_BBOX[3])]
    t["street_t"] = t[c["street"]].str.strip()
    t = (t.sort_values(["street_t", "num", c["number"]])
           .drop_duplicates(["street_t", "num"]).reset_index(drop=True))
    streets = sorted(t["street_t"].unique())
    sidx = {s: i for i, s in enumerate(streets)}
    log.info("addresses: %d unique street numbers on %d streets", len(t), len(streets))
    lon0, lat0 = STUDY_BBOX[0], STUDY_BBOX[2]
    return {
        "streets": streets,
        "street": t["street_t"].map(sidx).to_numpy().astype("<u2"),
        "number": t["num"].to_numpy().astype("<u2"),
        # 1e-5 degree offsets from the bbox corner fit in uint16 (~1 m)
        "lon": np.clip(np.round((t["lon"].to_numpy() - lon0) / 1e-5), 0, 65535).astype("<u2"),
        "lat": np.clip(np.round((t["lat"].to_numpy() - lat0) / 1e-5), 0, 65535).astype("<u2"),
        "origin": [lon0, lat0],
    }


def build_hillshade(width_px: int = 1600) -> dict:
    """Quiet shaded relief in WGS84, as a palette PNG data URI with bounds."""
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    from .elevation import DEM_MOSAIC
    from .viz_static import hillshade

    with step("rendering the hillshade base", log):
        with rasterio.open(DEM_MOSAIC) as src:
            transform, w, h = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
            scale = width_px / w
            w2, h2 = int(w * scale), int(h * scale)
            transform = transform * transform.scale(w / w2, h / h2)
            dem = np.full((h2, w2), np.nan, dtype="float32")
            reproject(rasterio.band(src, 1), dem, dst_transform=transform,
                      dst_crs="EPSG:4326", dst_nodata=np.nan,
                      resampling=Resampling.average)
            nod = src.nodata
        dem = np.where(np.isfinite(dem) & (dem != nod) & (dem > -50), dem, np.nan)
        valid = np.isfinite(dem)
        filled = np.where(valid, dem, np.nanmedian(dem))
        mid_lat = (STUDY_BBOX[2] + STUDY_BBOX[3]) / 2
        px_m = abs(transform.a) * 111320 * np.cos(np.radians(mid_lat))
        hs = hillshade(filled, res=px_m, z_factor=1.8)
        shade = (0.72 + 0.28 * hs)[..., None]
        # tint by height above the Limmat (about 392 m where it leaves the
        # city) up to the Zürichberg's crest; the Uetliberg saturates
        tint = np.clip((filled - HILLSHADE_BASE_M) / HILLSHADE_SPAN_M, 0, 1)[..., None]
        base = np.array([243, 242, 238], float); dark = np.array([196, 194, 186], float)
        rgb = (base * (1 - tint * 0.45) + dark * (tint * 0.45)) * shade
        img = np.zeros((h2, w2, 4), np.uint8)
        img[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        img[..., 3] = np.where(valid, 255, 0)
        # the lake and rivers: swissALTI3D carries them as flat ground
        from .viz_static import water_geometry
        water = water_geometry("EPSG:4326")
        if water is not None:
            from rasterio.features import rasterize
            wet = rasterize([water], out_shape=(h2, w2), transform=transform,
                            fill=0, default_value=1, dtype="uint8").astype(bool)
            img[wet, :3] = WATER_RGB
            img[wet, 3] = 255
        im = Image.fromarray(img, "RGBA").quantize(colors=96, method=Image.Quantize.FASTOCTREE)
        buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
        im.save(HILLSHADE_PNG)
        west, north = transform.c, transform.f
        east, south = west + transform.a * w2, north + transform.e * h2
        log.info("  hillshade %dx%d, %.0f KB", w2, h2, len(buf.getvalue()) / 1024)
    return {"bounds": [[south, west], [north, east]], "png": buf.getvalue(),
            "data_uri": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
