"""Registry of every external dataset used by the project.

Each entry records the URL, the access date, resolution/vintage, licence and
the limitations that matter for this analysis.  ``python -m zrh_flat_routes
sources`` prints this table, and it is the single source of truth for the
data-provenance section of the README.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Date on which every URL below was last fetched and verified.
ACCESS_DATE = "2026-10-06"

#: Overture Maps release used for the street network.
OVERTURE_RELEASE = "2026-09-23.1"
OVERTURE_BUCKET = "https://overturemaps-us-west-2.s3.amazonaws.com"
OVERTURE_PREFIX = f"release/{OVERTURE_RELEASE}/theme=transportation"
#: Places, addresses and base themes of the same release, used only for the
#: route page's offline place search.
OVERTURE_PLACES_PREFIX = f"release/{OVERTURE_RELEASE}/theme=places"
OVERTURE_ADDRESSES_PREFIX = f"release/{OVERTURE_RELEASE}/theme=addresses"
OVERTURE_BASE_PREFIX = f"release/{OVERTURE_RELEASE}/theme=base"

#: swisstopo's STAC catalogue, which lists one item per 1 km swissALTI3D tile.
STAC_ROOT = "https://data.geo.admin.ch/api/stac/v0.9"
SWISSALTI3D_COLLECTION = "ch.swisstopo.swissalti3d"
#: Ground sample distance (m) of the swissALTI3D product used.
SWISSALTI3D_GSD = "2"
#: The DHM25 matrix model, one national ASCII grid in LV03; validation only.
DHM25_URL = "https://cms.geo.admin.ch/ogd/topography/DHM25_MM_ASCII_GRID.zip"

#: City of Zurich open-data WFS for the statistical quarters.
QUARTERS_WFS = "https://www.ogd.stadt-zuerich.ch/wfs/geoportal/Statistische_Quartiere"
QUARTERS_TYPENAME = "adm_statistische_quartiere_map"
QUARTERS_PAGE = "https://data.stadt-zuerich.ch/dataset/geo_statistische_quartiere"

#: swisstopo's official directory of building addresses (national file).
ADDRESSES_COLLECTION = "ch.swisstopo.amtliches-gebaeudeadressverzeichnis"
#: The study area in LV95 (xmin, ymin, xmax, ymax), for cutting national files.
LV95_WINDOW = (2673000, 1238000, 2692000, 1257000)

#: OpenStreetMap ways carrying an ``incline`` tag, via the Overpass API;
#: validation only.
OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter")

#: Some build sandboxes cannot reach data.geo.admin.ch, cms.geo.admin.ch,
#: ogd.stadt-zuerich.ch or the Overpass API. The workflow
#: ``.github/workflows/mirror-data.yml`` fetches those files in GitHub
#: Actions and attaches them, unmodified (DHM25 cut to a window around
#: Zurich), to this release; ``download`` falls back to it.
MIRROR_RELEASE = "https://github.com/MoritzFS/flatten_zrh/releases/download/source-data"


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    publisher: str
    url: str
    accessed: str
    resolution: str
    licence: str
    limitations: str
    role: str
    local: str = ""
    notes: str = ""
    optional: bool = False
    substituted: bool = False
    substitution_reason: str = ""


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        key="overture_segments",
        title=f"Overture Maps transportation segments (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_PREFIX}/type=segment/",
        accessed=ACCESS_DATE,
        resolution="Vector linestrings; OSM-equivalent positional accuracy (~1-5 m)",
        licence="ODbL 1.0 (OpenStreetMap contributors); Overture schema CDLA-Permissive 2.0",
        role="Routable street network: geometry, road class, per-mode access "
             "restrictions, bridge/tunnel flags and connector topology.",
        local="data/raw/overture_segments_zrh.parquet",
        limitations=(
            "OSM-derived, so completeness and tagging quality vary by area, "
            "though Zurich is among the most thoroughly mapped cities in "
            "Europe. Separately mapped sidewalks and crossings are present "
            "but deliberately not used: travel is modelled on street "
            "centrelines. Swiss mappers tag limited-access 'Autostrassen' as "
            "trunk, with walking and cycling denied by access rules, which "
            "the access parser honours."
        ),
        notes="Read with Parquet row-group bbox pruning, so the whole "
              "extract costs a few seconds and a few MB instead of the "
              "theme's tens of gigabytes.",
    ),
    Dataset(
        key="overture_connectors",
        title=f"Overture Maps transportation connectors (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_PREFIX}/type=connector/",
        accessed=ACCESS_DATE,
        resolution="Vector points",
        licence="ODbL 1.0; Overture schema CDLA-Permissive 2.0",
        role="Authoritative intersection nodes. Using connector IDs for graph "
             "topology avoids geometric snapping tolerances entirely.",
        local="data/raw/overture_connectors_zrh.parquet",
        limitations="Connectors exist only where OSM ways share a node; "
                    "grade-separated crossings correctly do not connect.",
    ),
    Dataset(
        key="dem_swissalti3d",
        title="swissALTI3D digital terrain model, 2 m (2026 edition)",
        publisher="Federal Office of Topography swisstopo",
        url=f"{STAC_ROOT}/collections/{SWISSALTI3D_COLLECTION}",
        accessed=ACCESS_DATE,
        resolution="2 m grid, 1 km tiles; CH1903+/LV95 (EPSG:2056); float32 "
                   "metres above sea level (LN02)",
        licence="swisstopo Open Government Data: free use, commercial use "
                "included; source reference mandatory "
                "('Federal Office of Topography swisstopo' or '©swisstopo')",
        role="Primary elevation source for all grade and climbing metrics.",
        local="data/raw/dem/*.tif",
        limitations=(
            "A bare-earth model (buildings and vegetation removed), produced "
            "from airborne lidar and published at 0.5 m and 2 m; the 2 m "
            "product is used, which is far finer than the 50 m smoothing "
            "window the analysis applies. Bridges, tunnels and elevated "
            "structures are removed, so the model describes the ground or "
            "water under a deck -- handled explicitly by interpolating "
            "elevation across segments flagged is_bridge or is_tunnel. "
            "swisstopo gives its accuracy (1 sigma) as 0.3 m where it is "
            "built from new-generation lidar and 0.5 m from the previous "
            "generation below 2,000 m; it is updated on a six-year cycle. "
            "316 tiles cover the study area."
        ),
    ),
    Dataset(
        key="dem_dhm25",
        title="DHM25 matrix model (25 m), window around Zurich",
        publisher="Federal Office of Topography swisstopo",
        url=DHM25_URL,
        accessed=ACCESS_DATE,
        resolution="25 m grid; CH1903/LV03 (EPSG:21781); metres above sea level (LN02)",
        licence="swisstopo Open Government Data; source reference mandatory",
        role="Independent cross-check on swissALTI3D (validation only -- too "
             "coarse for street grades).",
        local="data/raw/dhm25_zurich_lv03.tif",
        limitations="Interpolated from the contour lines and spot heights of "
                    "the Swiss National Map, so it shares no lidar with "
                    "swissALTI3D, but its 25 m posting smooths away "
                    "street-scale relief and it predates much of the city's "
                    "recent construction.",
        optional=True,
    ),
    Dataset(
        key="neighborhoods",
        title="Statistische Quartiere (statistical quarters) of the City of Zurich",
        publisher="Stadt Zürich: Statistik Stadt Zürich and GIS-Zentrum "
                  "(Geomatik + Vermessung), via Open Data Zürich",
        url=QUARTERS_PAGE,
        accessed=ACCESS_DATE,
        resolution="Vector polygons, 34 quarters in 12 districts (Kreise); "
                   "dataset last updated 2026-10-02",
        licence="CC0 1.0 (Creative Commons Zero), as published on "
                "data.stadt-zuerich.ch; opendata.swiss lists it as 'Open use'",
        role="Quarter boundaries for origin/destination selection, corridor "
             "attribution and the city boundary that clips the network.",
        local="data/raw/zrh_quarters.geojson",
        limitations=(
            "Statistical units, not the 23 historical quarters (Quartiere) "
            "alone: several large historical quarters are split (Wiedikon, "
            "Aussersihl, Schwamendingen...). Quarters bordering the lake "
            "include water, which affects the geometric centroid but not "
            "the street-weighted access point the analysis uses."
        ),
    ),
    Dataset(
        key="osm_incline",
        title="OpenStreetMap ways with an incline tag (via the Overpass API)",
        publisher="OpenStreetMap contributors",
        url=OVERPASS_URLS[0],
        accessed=ACCESS_DATE,
        resolution="Tag values on OSM ways, mostly copied from gradient "
                   "warning signs",
        licence="ODbL 1.0 (OpenStreetMap contributors)",
        role="Independent reference gradients for validating computed street "
             "grades (validation only).",
        local="data/raw/osm_incline_zurich.json",
        limitations="Coverage is sparse and the values are what mappers "
                    "entered: some are signed maxima, some estimates, some "
                    "just 'up'/'down'. Only numeric percentages are used.",
        optional=True,
    ),
    Dataset(
        key="overture_places",
        title=f"Overture Maps places (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (Meta, Microsoft and other POI sources)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_PLACES_PREFIX}/type=place/",
        accessed=ACCESS_DATE,
        resolution="Point features with names, categories and a confidence score",
        licence="CDLA Permissive 2.0",
        role="Offline place search in the route page (parks, landmarks, "
             "transit, schools, shops, cafes).",
        local="data/raw/overture_places_zrh.parquet",
        limitations="Point-of-interest coverage and naming are uneven; only "
                    "records with confidence >= 0.6 in routable categories "
                    "are kept. Not used by the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="overture_base",
        title=f"Overture Maps base theme: land use, infrastructure, land (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_BASE_PREFIX}/",
        accessed=ACCESS_DATE,
        resolution="Mapped outlines and points with names and OSM-derived classes",
        licence="ODbL 1.0 (OpenStreetMap contributors)",
        role="Mapped parks, schools, hospitals, plazas, stations, piers, "
             "bridges, viewpoints, peaks and beaches for the route page's "
             "offline search; these outrank the POI feed, which places the "
             "same names unreliably.",
        local="data/raw/overture_{land_use,infrastructure,land}_zrh.parquet",
        limitations="Only named features in a fixed class list are used. "
                    "Not used by the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="addresses",
        title="Official directory of building addresses (Amtliches Verzeichnis "
              "der Gebäudeadressen), rows inside the study area",
        publisher="Federal Office of Topography swisstopo",
        url=f"{STAC_ROOT}/collections/{ADDRESSES_COLLECTION}",
        accessed=ACCESS_DATE,
        resolution="One point per building entrance, with street name and "
                   "house number; LV95",
        licence="swisstopo Open Government Data; source reference mandatory",
        role="Offline street-address search in the route page.",
        local="data/raw/swisstopo_addresses_zurich.csv",
        limitations="One point per (street, leading house number) is kept; "
                    "letter and sub-number suffixes ('12a', '4.1') fold into "
                    "the number. Not used by the analysis itself. Overture "
                    "also carries Swiss addresses (OpenAddresses source "
                    "'ch/countrywide'), but labels their licence only as "
                    "proprietary, so the official register is used instead.",
        optional=True,
    ),
)

DATASETS_BY_KEY = {d.key: d for d in DATASETS}


def format_table() -> str:
    """Human-readable provenance report."""
    lines = [f"Data sources (all URLs verified {ACCESS_DATE})", "=" * 78]
    for d in DATASETS:
        flag = " [OPTIONAL]" if d.optional else ""
        flag += " [SUBSTITUTED]" if d.substituted else ""
        lines += [
            f"\n{d.key}{flag}",
            f"  title       : {d.title}",
            f"  publisher   : {d.publisher}",
            f"  url         : {d.url}",
            f"  accessed    : {d.accessed}",
            f"  resolution  : {d.resolution}",
            f"  licence     : {d.licence}",
            f"  local cache : {d.local}",
            f"  role        : {d.role}",
            f"  limitations : {d.limitations}",
        ]
        if d.notes:
            lines.append(f"  notes       : {d.notes}")
        if d.substitution_reason:
            lines.append(f"  substitution: {d.substitution_reason}")
    return "\n".join(lines)


def markdown_table() -> str:
    """Compact markdown table for the README."""
    rows = ["| Dataset | Publisher | Resolution / vintage | Licence | Role |",
            "|---|---|---|---|---|"]
    for d in DATASETS:
        rows.append(
            f"| {d.title} | {d.publisher} | {d.resolution} | {d.licence} | {d.role} |"
        )
    return "\n".join(rows)
