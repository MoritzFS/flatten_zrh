"""The route page's offline place index."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zrh_flat_routes import places
from zrh_flat_routes.config import STUDY_BBOX
from zrh_flat_routes.download import ADDRESSES_CSV, BASE_PARQUETS, PLACES_PARQUET


def test_core_strips_a_trailing_city_name_and_folds_umlauts():
    assert places._core("Kunsthaus Zürich") == "kunsthaus"
    assert places._core("Lindenhof, Zürich") == "lindenhof"
    assert places._core("Uetliberg - Zurich ZH") == places._core("Uetliberg")
    assert places._core("Zürichhorn") == "zurichhorn"
    # the POI feed's spelling and the mapped one are the same place
    assert places._core("Zurich HB") == places._core("Zürich HB")
    # a suffix that is part of the name is not a city suffix
    assert places._core("Café Schweiz") == "cafe schweiz"


def test_support_counts_nearby_records_that_mention_the_name():
    names = pd.Series(["Lindenhof", "Lindenhof Bar", "Lindenhof Keller",
                       "Lindenhof", "Nowhere"])
    lon = np.array([8.5409, 8.5412, 8.5405, 8.5100, 8.4500])
    lat = np.array([47.3730, 47.3733, 47.3727, 47.4200, 47.3300])
    sup = places._support(names, lon, lat, names, lon, lat)
    assert sup[0] == 2          # the real square: two neighbours mention it
    assert sup[3] == 0          # the stray copy across town: none


def test_variant_pruning_keeps_different_kinds_and_distant_namesakes():
    df = pd.DataFrame({
        "name": ["Platzspitz", "Platzspitz, Zürich", "Platzspitz Café",
                 "Rieterpark", "Rieterpark - Süd", "Rieterpark Villa"],
        "group": ["park", "park", "food", "park", "park", "landmark"],
        "lon": [8.5408, 8.5300, 8.5410, 8.5300, 8.5320, 8.5320],
        "lat": [47.3815, 47.3700, 47.3817, 47.3590, 47.3580, 47.3580],
        "conf": [0.97, 0.72, 0.99, 0.98, 0.9, 0.9],
        "support": [3, 0, 0, 5, 0, 0],
    })
    out = places._prune_variants(df)
    kept = set(out["name"])
    assert "Platzspitz" in kept
    assert "Platzspitz, Zürich" not in kept      # city suffix, any distance
    assert "Platzspitz Café" in kept             # a different kind of place
    assert "Rieterpark" in kept
    assert "Rieterpark Villa" in kept            # different kind
    assert "Rieterpark - Süd" not in kept        # same kind, unsupported variant


needs_places = pytest.mark.skipif(not PLACES_PARQUET.exists(),
                                  reason="Overture places not downloaded")
needs_base = pytest.mark.skipif(not all(p.exists() for p in BASE_PARQUETS.values()),
                                reason="Overture base theme not downloaded")
needs_addresses = pytest.mark.skipif(not ADDRESSES_CSV.exists(),
                                     reason="swisstopo addresses not downloaded")


@pytest.fixture(scope="module")
def index():
    return places.build_places()


@needs_places
@needs_base
def test_place_index_is_compact_and_inside_the_city(index):
    n = len(index["names"])
    assert 5000 < n < 20000
    assert len(index["group"]) == n == len(index["lon"]) == len(index["lat"])
    assert max(index["group"]) < len(index["groups"])
    assert min(index["lon"]) >= STUDY_BBOX[0] and max(index["lon"]) <= STUDY_BBOX[1]
    assert min(index["lat"]) >= STUDY_BBOX[2] and max(index["lat"]) <= STUDY_BBOX[3]
    assert len(set(index["names"])) == n or len(set(zip(index["names"], index["group"]))) == n


@needs_places
@needs_base
def test_stops_and_mapped_features_win_over_the_poi_feed(index):
    """The POI feed's 'Bellevue' is a restaurant in Affoltern; the tram stop
    at Bellevueplatz is the only record left under that name."""
    hits = {}
    for n, g, lo, la in zip(index["names"], index["group"], index["lon"], index["lat"]):
        hits.setdefault(n, []).append((index["groups"][g], lo, la))
    for name, lon, lat in [("Bellevue", 8.5453, 47.3670),
                           ("Central", 8.5437, 47.3768),
                           ("Paradeplatz", 8.5390, 47.3697),
                           ("Zürich Oerlikon", 8.5442, 47.4115),
                           ("Lindenhof", 8.5409, 47.3730)]:
        assert name in hits, name
        for kind, lo, la in hits[name]:
            assert abs(lo - lon) < 0.004 and abs(la - lat) < 0.004, (name, kind, lo, la)


@needs_addresses
def test_addresses_pack_into_sorted_uint16_offsets():
    a = places.build_addresses()
    n = len(a["number"])
    assert n > 50_000
    assert a["street"].dtype == np.dtype("<u2") and a["number"].dtype == np.dtype("<u2")
    assert a["lon"].dtype == np.dtype("<u2") and a["lat"].dtype == np.dtype("<u2")
    # sorted by street then number, so the browser can binary-search
    key = a["street"].astype(np.int64) * 100_000 + a["number"]
    assert np.all(np.diff(key) > 0)
    lon = a["origin"][0] + a["lon"] * 1e-5
    lat = a["origin"][1] + a["lat"] * 1e-5
    assert lon.min() >= STUDY_BBOX[0] and lon.max() <= STUDY_BBOX[1] + 1e-4
    assert lat.min() >= STUDY_BBOX[2] and lat.max() <= STUDY_BBOX[3] + 1e-4
    assert "Bahnhofstrasse" in a["streets"]
    assert "Langstrasse" in a["streets"]
