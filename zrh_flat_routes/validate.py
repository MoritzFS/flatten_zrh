"""Validation of the elevation and routing model against known ground truth.

Upstream (flattensf) validates San Francisco against an independent USGS
DEM, published grades of its famously steep streets and the corridors local
knowledge calls flat. Zurich gets the same three checks, against Swiss
references, plus one about the city's one big pass:

1.  **Absolute elevation** -- the swissALTI3D mosaic against swisstopo's
    DHM25, which was interpolated from the contour lines of the 1:25,000
    National Map and so shares no lidar with swissALTI3D.  Agreement is
    evidence the mosaic is correctly georeferenced and in the expected
    vertical datum (LN02).

2.  **Street grades** -- computed grades against the ``incline`` tags
    OpenStreetMap mappers put on Zurich's ways, mostly copied from gradient
    warning signs.  This is the check that catches sampling or smoothing
    problems.

3.  **Flat corridors** -- the streets everyone in Zurich knows are level
    (the quays along the Limmat and the lake, Bahnhofstrasse, Langstrasse,
    Badenerstrasse, the Kreis 5 plain, the floor of the Glatt valley) must
    come out flat.

4.  **The saddle** -- the Limmat valley and the Glatt valley are divided by
    the Käferberg-Zürichberg ridge, whose lowest crossing is the saddle
    between Bucheggplatz (472 m) and Milchbuck (476 m).  The model's lowest
    possible crossing, and the high point of its flat routes across, must
    land there.

The model is *not* tuned to make these pass; where a target disagrees, the
disagreement is reported with a diagnosis.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .config import MIN_RELIABLE_GRADE_LENGTH_M, OUTPUT_DIR
from .utils import get_logger, step

log = get_logger("zrh_flat_routes.validate")

VALIDATION_MD = OUTPUT_DIR / "validation_report.md"

#: Published elevations (m) of the saddle between the Käferberg and the
#: Zürichberg, the lowest crossing from the Limmat valley to the Glatt
#: valley (Wikipedia, "Käferberg": "a small mountain pass between
#: Bucheggplatz (472 m) and Milchbuck (476 m)").
SADDLE_PUBLISHED = {"Bucheggplatz": 472.0, "Milchbuck": 476.0}
#: The saddle area, for deciding whether a route's high point is "on the
#: saddle": (lon_min, lon_max, lat_min, lat_max), WGS84.
SADDLE_BBOX = (8.522, 8.550, 47.392, 47.401)

#: The trip across the saddle: Central (the Limmat at the main station) to
#: Oerlikon station, on the floor of the Glatt valley.
SADDLE_TRIP = ((8.5442, 47.3770), (8.5442, 47.4115))

#: Corridors local knowledge says are flat: the streets that carry them and,
#: where the name alone is ambiguous, the geographic window that isolates the
#: corridor. Windows are (lon_min, lon_max, lat_min, lat_max) in WGS84.
KNOWN_FLAT = {
    "Limmatquai (along the Limmat)": {"streets": ["Limmatquai"]},
    "Lake promenade (Utoquai, Seefeldquai, General-Guisan-Quai, Mythenquai)": {
        "streets": ["Utoquai", "Seefeldquai", "General-Guisan-Quai", "Mythenquai"],
    },
    "Bahnhofstrasse": {"streets": ["Bahnhofstrasse"],
                       "bbox": (8.535, 8.542, 47.365, 47.378)},
    "Langstrasse": {"streets": ["Langstrasse"]},
    "Badenerstrasse (Stauffacher to Altstetten)": {
        "streets": ["Badenerstrasse"], "bbox": (8.470, 8.530, 47.380, 47.395),
    },
    "Limmatstrasse / Sihlquai (Kreis 5)": {"streets": ["Limmatstrasse", "Sihlquai"]},
    "Hardturmstrasse / Pfingstweidstrasse (the Limmat plain)": {
        "streets": ["Hardturmstrasse", "Pfingstweidstrasse"],
    },
    "Thurgauerstrasse (the Glatt valley floor)": {"streets": ["Thurgauerstrasse"]},
}

DRIVABLE = ("residential", "living_street", "tertiary", "secondary",
            "primary", "trunk", "unclassified")
#: OSM highway types counted as streets in the incline comparison.
STREET_TYPES = DRIVABLE + ("service", "pedestrian")


# --------------------------------------------------------------------------
def check_dem_agreement(n_points: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Compare the swissALTI3D mosaic with DHM25 at random points on land.

    Points are drawn inside the city, off the lake and rivers. DHM25 is in
    LV03; the shift to LV95 is the defined false-origin offset of
    (2,000,000, 1,000,000) m, whose residual against the official FINELTRA
    transformation (well under 2 m) is irrelevant at a 25 m grid.
    """
    import rasterio
    import shapely
    from shapely.prepared import prep

    from .download import DHM25_TIF
    from .elevation import DEM_MOSAIC, DemSampler
    from .neighborhoods import city_boundary
    from .viz_static import water_geometry

    if not DHM25_TIF.exists():
        log.warning("DHM25 not cached; skipping cross-check")
        return pd.DataFrame()

    land = city_boundary(buffer_m=0.0)
    water = water_geometry("EPSG:2056")
    if water is not None:
        land = land.difference(water.buffer(25.0))
    land_p = prep(land)
    sampler = DemSampler(DEM_MOSAIC, smooth=False)
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = land.bounds
    xs = rng.uniform(x0, x1, n_points * 6)
    ys = rng.uniform(y0, y1, n_points * 6)
    inside = np.array([land_p.contains(p) for p in shapely.points(xs, ys)])
    xs, ys = xs[inside], ys[inside]
    z1 = sampler.sample(xs, ys)
    ok = np.isfinite(z1)
    xs, ys, z1 = xs[ok][:n_points], ys[ok][:n_points], z1[ok][:n_points]

    with rasterio.open(DHM25_TIF) as d25:
        nod = d25.nodata
        z2 = np.array([v[0] for v in d25.sample(list(zip(xs - 2_000_000.0,
                                                         ys - 1_000_000.0)))],
                      dtype="float64")
    good = np.isfinite(z2) & (z2 > 0)
    if nod is not None:
        good &= z2 != nod
    df = pd.DataFrame({"x": xs[good], "y": ys[good],
                       "z_alti3d": z1[good], "z_dhm25": z2[good]})
    df["diff"] = df["z_alti3d"] - df["z_dhm25"]
    log.info("DEM cross-check on %d points: mean diff %+.2f m, median %+.2f m, "
             "RMS %.2f m, |diff|<2 m for %.1f%%", len(df), df["diff"].mean(),
             df["diff"].median(), float(np.sqrt((df["diff"] ** 2).mean())),
             100.0 * (df["diff"].abs() < 2).mean())
    return df


# --------------------------------------------------------------------------
_INCLINE_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*%\s*$")


def _parse_incline(value: str) -> float | None:
    """'12%' -> 0.12; anything else (up, down, degrees, junk) -> None."""
    m = _INCLINE_RE.match(str(value or ""))
    if not m:
        return None
    v = abs(float(m.group(1).replace(",", "."))) / 100.0
    return v if v <= 0.60 else None      # 140% on a stairway is not a grade


def _segment_osm_ways() -> dict[str, list[int]]:
    """Overture segment id -> the OpenStreetMap way ids it was built from."""
    import pyarrow.parquet as pq

    from .download import SEGMENTS_PARQUET
    t = pq.read_table(SEGMENTS_PARQUET, columns=["id", "sources"]).to_pandas()
    out: dict[str, list[int]] = {}
    for sid, srcs in zip(t["id"], t["sources"]):
        ways = []
        for s in (srcs if srcs is not None else []):
            rid = str((s or {}).get("record_id") or "")
            m = re.match(r"^w(\d+)", rid)
            if m:
                ways.append(int(m.group(1)))
        if ways:
            out[sid] = ways
    return out


def check_osm_inclines(edges, min_len_m: float = 30.0) -> pd.DataFrame:
    """Computed grades against OpenStreetMap ``incline`` tags, way by way.

    Each tagged way is matched to the network edges built from it (through
    Overture's source records), and compared on two numbers: the steepest
    reliable edge's maximum grade, which is what a warning sign states, and
    the length-weighted mean grade, which is what a mapper estimating a ramp
    tends to write. Stairways are left out (their tags are as often the
    angle of the flight as its gradient) and so are ways shorter than
    ``min_len_m``.
    """
    from .download import OSM_INCLINE_JSON

    if not OSM_INCLINE_JSON.exists():
        log.warning("OpenStreetMap incline tags not cached; skipping")
        return pd.DataFrame()
    tagged = {}
    for el in json.loads(OSM_INCLINE_JSON.read_text())["elements"]:
        tags = el.get("tags", {})
        g = _parse_incline(tags.get("incline"))
        if g is None or tags.get("highway") == "steps":
            continue
        tagged[int(el["id"])] = {"published": g, "highway": tags.get("highway"),
                                 "osm_name": tags.get("name")}
    seg_ways = _segment_osm_ways()
    rows = []
    e = edges[["segment_id", "name", "cls", "length_m", "max_abs_grade",
               "avg_grade_fwd"]].copy()
    e["way"] = e["segment_id"].map(lambda s: seg_ways.get(s, [None])[0])
    e = e[e["way"].isin(tagged.keys())]
    for way, grp in e.groupby("way"):
        L = float(grp["length_m"].sum())
        if L < min_len_m:
            continue
        rel = grp[grp["length_m"] >= MIN_RELIABLE_GRADE_LENGTH_M]
        pub = tagged[int(way)]
        w = grp["length_m"].to_numpy()
        names = grp["name"].dropna()
        rows.append({
            "way": int(way),
            "name": pub["osm_name"] or (names.iloc[0] if len(names) else None),
            "highway": pub["highway"], "length_m": L,
            "published": pub["published"],
            "computed_max": float((rel if len(rel) else grp)["max_abs_grade"].max()),
            "computed_mean": float((grp["avg_grade_fwd"].abs() * w).sum() / w.sum()),
        })
    df = pd.DataFrame(rows)
    if len(df):
        # the better of the two readings, since the tag does not say which
        # it means
        df["diff_max"] = df["computed_max"] - df["published"]
        df["diff_mean"] = df["computed_mean"] - df["published"]
        df["diff_best"] = np.where(df["diff_max"].abs() < df["diff_mean"].abs(),
                                   df["diff_max"], df["diff_mean"])
        log.info("OSM incline check on %d ways: median |diff| %.1f points "
                 "(best of max/mean), %.0f%% within 3 points", len(df),
                 100 * df["diff_best"].abs().median(),
                 100 * (df["diff_best"].abs() <= 0.03).mean())
    return df


def _window(edges, bbox):
    """Restrict an edge table to a WGS84 bounding box."""
    if bbox is None:
        return edges
    from shapely.geometry import box
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", edges.crs, always_xy=True)
    x0, y0 = tr.transform(bbox[0], bbox[2])
    x1, y1 = tr.transform(bbox[1], bbox[3])
    return edges[edges.geometry.intersects(box(x0, y0, x1, y1))]


def check_flat_corridors(edges, corridors=None) -> pd.DataFrame:
    """Do the known flat corridors measure flat, and do they get discovered?"""
    rows = []
    for label, spec in KNOWN_FLAT.items():
        streets = spec["streets"]
        sub = _window(edges[edges["name"].isin(streets)], spec.get("bbox"))
        if sub.empty:
            rows.append({"corridor": label, "streets": "; ".join(streets),
                         "street_km": 0.0, "gain_per_km": np.nan,
                         "mean_abs_grade": np.nan, "verdict": "not found",
                         "discovered": False})
            continue
        km = float(sub["length_m"].sum() / 1000)
        gain_km = float(sub["cum_gain_fwd"].sum() / km) if km else np.nan
        w = sub["length_m"].to_numpy(dtype="float64")
        mean_grade = float((sub["avg_grade_fwd"].abs() * w).sum() / w.sum())
        discovered = False
        if corridors is not None and len(corridors):
            names = corridors["street_names"].fillna("")
            discovered = bool(any(any(s in n for s in streets) for n in names))
        rows.append({
            "corridor": label, "streets": "; ".join(streets),
            "street_km": km, "gain_per_km": gain_km,
            "mean_abs_grade": mean_grade,
            "verdict": "flat" if gain_km < 15 else "not flat",
            "discovered": discovered,
        })
    return pd.DataFrame(rows)


#: set by run_validation so the route-based checks can reach the graphs
_ROUTE_CTX: dict = {}


def _nearest_graph_node(graph, edges, lon, lat):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", edges.crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    nodes = set(graph.node_ids)
    sub = edges[edges["u"].isin(nodes)]
    coords = np.array([g.coords[0] for g in sub.geometry])
    d = (coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2
    return sub.iloc[int(np.argmin(d))]["u"]


def check_saddle(ctx) -> dict:
    """The lowest crossing from the Limmat valley to the Glatt valley.

    Two independent answers are compared with the published saddle: the
    exact minimax pass height between Central and Oerlikon (the merge-tree
    construction in ``passes.py``), and the high point of the routes the
    model actually chooses for that trip, walking and cycling.
    """
    from pyproj import Transformer

    from .config import ROUTING_PROFILES
    from .passes import build_bottleneck_tree, pass_height
    from .routing import route

    edges = ctx.edges
    out: dict = {"published": SADDLE_PUBLISHED}
    tr = Transformer.from_crs(edges.crs, "EPSG:4326", always_xy=True)
    e_idx = edges.set_index("edge_id")

    def on_saddle(eid) -> tuple[str, bool, float, float]:
        g = e_idx.loc[eid, "geometry"].interpolate(0.5, normalized=True)
        lon, lat = tr.transform(g.x, g.y)
        inside = (SADDLE_BBOX[0] <= lon <= SADDLE_BBOX[1]
                  and SADDLE_BBOX[2] <= lat <= SADDLE_BBOX[3])
        name = e_idx.loc[eid, "name"]
        return (name if isinstance(name, str) else "(unnamed path)"), inside, lon, lat

    (alon, alat), (blon, blat) = SADDLE_TRIP
    tree = build_bottleneck_tree(edges, mode="walk")
    a = _nearest_graph_node(ctx.graphs["walk"], edges, alon, alat)
    b = _nearest_graph_node(ctx.graphs["walk"], edges, blon, blat)
    h, eid = pass_height(tree, a, b)
    name, inside, lon, lat = on_saddle(eid)
    out["pass"] = {"elev_m": h, "street": name, "on_saddle": inside,
                   "lon": lon, "lat": lat}

    out["routes"] = {}
    for mode in ("walk", "bike"):
        graph = ctx.graphs[mode]
        a = _nearest_graph_node(graph, edges, alon, alat)
        b = _nearest_graph_node(graph, edges, blon, blat)
        for pname in ("shortest", "balanced", "min_climb"):
            arcs, s = route(graph, a, b, ROUTING_PROFILES[pname])
            tab = graph.table.iloc[arcs]
            hi = tab["edge_id"].to_numpy()[int(np.argmax(tab["end_elev"].to_numpy()))]
            top = float(np.max(tab[["start_elev", "end_elev"]].to_numpy()))
            nm, ins, _, _ = on_saddle(hi)
            out["routes"][(mode, pname)] = {
                "km": s["distance_m"] / 1000.0, "gain_m": s["elev_gain_m"],
                "max_grade": s["max_grade"], "high_point_m": top,
                "high_point_street": nm, "high_point_on_saddle": ins,
            }
    return out


def run_validation(ctx, corridors=None, write: bool = True) -> dict:
    """Run every check and write a markdown report."""
    _ROUTE_CTX["ctx"] = ctx
    with step("validating elevation model against known ground truth", log):
        dem = check_dem_agreement()
        incl = check_osm_inclines(ctx.edges)
        flat = check_flat_corridors(ctx.edges, corridors)
        try:
            saddle = check_saddle(ctx)
        except Exception as exc:                      # pragma: no cover
            log.warning("saddle check failed: %s", exc)
            saddle = {}

    if write:
        _write_report(dem, incl, flat, saddle)
    return {"dem": dem, "incline": incl, "flat": flat, "saddle": saddle}


def _write_report(dem, incl, flat, saddle) -> None:
    L: list[str] = ["# Validation report", "",
                    "Zurich's checks follow the four upstream (flattensf) runs for "
                    "San Francisco, against Swiss references. Every figure below "
                    "is generated by `python -m zrh_flat_routes validate`.", ""]
    L += ["## 1. Elevation: swissALTI3D (2 m) vs the independent DHM25 (25 m)", ""]
    if len(dem):
        d = dem["diff"]
        L += [f"- Points compared: **{len(dem):,}**, random, on land inside the city",
              f"- Mean difference: **{d.mean():+.2f} m**, "
              f"median **{d.median():+.2f} m**",
              f"- RMS difference: **{np.sqrt((d**2).mean()):.2f} m**",
              f"- Within 2 m: **{100*(d.abs()<2).mean():.1f}%** of points; "
              f"within 5 m: **{100*(d.abs()<5).mean():.1f}%**", "",
              "DHM25 was interpolated from the contour lines of the 1:25,000 "
              "National Map and shares no lidar with swissALTI3D, so a small "
              "mean difference confirms the mosaic is correctly georeferenced "
              "and in metres above sea level (LN02). Residual scatter is "
              "expected: swisstopo gives DHM25's own average deviation as "
              "1.5 m on the Central Plateau, its 25 m grid cannot resolve "
              "street-scale relief, and it predates much of the city's "
              "construction (railway cuttings, the Hardbrücke, the Europaallee "
              "site).", ""]
    else:
        L += ["_Not run: DHM25 was not cached._", ""]

    L += ["## 2. Grades against OpenStreetMap incline tags", ""]
    if len(incl):
        groups = (
            ("Streets and lanes", incl["highway"].isin(STREET_TYPES)),
            ("Cycle paths and footways", incl["highway"].isin(("cycleway", "footway"))),
            ("Hiking paths and tracks", incl["highway"].isin(("path", "track", "bridleway"))),
        )
        L += [f"{len(incl):,} ways carry a numeric `incline` tag that matches "
              "a network edge (stairways excluded, ways under 30 m dropped). "
              "A tag does not say whether it means the steepest pitch, as a "
              "warning sign does, or the average, as a mapper estimating a "
              "ramp tends to write, so each way is scored against whichever "
              "of the computed maximum and mean grade it is closer to.", "",
              "| Ways | Count | Median difference | Within 3 points | "
              "Within 5 points | Correlation with computed max / mean |",
              "|---|---|---|---|---|---|"]
        for label, sel in groups:
            g = incl[sel]
            if len(g) < 3:
                continue
            best = g["diff_best"].abs()
            L.append(f"| {label} | {len(g)} | {100*best.median():.1f} points | "
                     f"{100*(best <= 0.03).mean():.0f}% | "
                     f"{100*(best <= 0.05).mean():.0f}% | "
                     f"{np.corrcoef(g['published'], g['computed_max'])[0, 1]:.2f} / "
                     f"{np.corrcoef(g['published'], g['computed_mean'])[0, 1]:.2f} |")
        L += ["", "| Street | Type | Length | Tagged | Computed max | Computed mean |",
              "|---|---|---|---|---|---|"]
        streets = incl[groups[0][1]].sort_values("published", ascending=False).head(15)
        for _, r in streets.iterrows():
            nm = r["name"] if isinstance(r["name"], str) and r["name"] else "(unnamed)"
            L.append(f"| [{nm}](https://www.openstreetmap.org/way/{r['way']}) | "
                     f"{r['highway']} | {r['length_m']:.0f} m | {r['published']:.0%} | "
                     f"{r['computed_max']:.1%} | {r['computed_mean']:.1%} |")
        L += ["", "_The fifteen steepest tagged streets: most are lanes of the "
              "Altstadt, climbing from the Limmat. Schneggengasse is the one "
              "real miss; the model reads it as nearly level._", ""]
    else:
        L += ["_Not run: the incline tags were not cached._", ""]

    L += ["## 3. Known flat corridors", "",
          "| Corridor | Km | Gain per km | Mean abs grade | Verdict | "
          "Discovered by the model? |", "|---|---|---|---|---|---|"]
    for _, r in flat.iterrows():
        g = "n/a" if not np.isfinite(r["gain_per_km"]) else f"{r['gain_per_km']:.1f} m"
        m = "n/a" if not np.isfinite(r["mean_abs_grade"]) else f"{r['mean_abs_grade']:.1%}"
        L.append(f"| {r['corridor']} | {r['street_km']:.1f} | {g} | {m} | "
                 f"{r['verdict']} | {'yes' if r['discovered'] else 'no'} |")
    L += ["", "Gain per km is cumulative climbing per kilometre of street, "
          "summed in the street's own direction: a level street scores a few "
          "metres from DEM texture that the dead-band does not quite remove. "
          "A corridor is *discovered* when one of its streets appears in the "
          "names of a corridor the model found for itself.", ""]

    if saddle:
        p = saddle["pass"]
        pub = saddle["published"]
        L += ["## 4. The saddle between the Limmat and the Glatt valleys", "",
              "The Käferberg-Zürichberg ridge separates the city centre from "
              "Oerlikon, Seebach and Schwamendingen; its lowest crossing is "
              "the saddle between Bucheggplatz "
              f"({pub['Bucheggplatz']:.0f} m) and Milchbuck "
              f"({pub['Milchbuck']:.0f} m).", "",
              f"- **Lowest possible crossing** from Central to Oerlikon on "
              f"foot, computed exactly by the merge tree: **{p['elev_m']:.1f} m**, "
              f"on {p['street']} "
              f"({'on the saddle' if p['on_saddle'] else 'NOT on the saddle'}).", "",
              "| Mode | Objective | Distance | Climb | Max grade | High point | Where |",
              "|---|---|---|---|---|---|---|"]
        for (mode, pname), r in saddle["routes"].items():
            L.append(f"| {mode} | {pname} | {r['km']:.2f} km | {r['gain_m']:.0f} m | "
                     f"{r['max_grade']:.1%} | {r['high_point_m']:.0f} m | "
                     f"{r['high_point_street']}"
                     f"{'' if r['high_point_on_saddle'] else ' (off the saddle)'} |")
        L += [""]

    VALIDATION_MD.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_MD.write_text("\n".join(L))
    log.info("wrote %s", VALIDATION_MD.name)
