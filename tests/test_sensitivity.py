"""Tests for the parameter-override plumbing and the sensitivity harness."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _config_in_subprocess(env_extra: dict) -> dict:
    """Import the config in a fresh interpreter under the given environment."""
    code = (
        "import json; from zrh_flat_routes import config as c, elevation as e\n"
        "print(json.dumps({'spacing': c.ELEVATION.sample_spacing_m,"
        " 'sigma': c.ELEVATION.dem_sigma_m, 'sigma_mod': e.DEM_SMOOTH_SIGMA_M,"
        " 'rank': c.ANALYSIS.point_rank, 'processed': str(c.PROCESSED_DIR),"
        " 'outputs': str(c.OUTPUT_DIR), 'raw': str(c.RAW_DIR)}))"
    )
    env = dict(os.environ)
    env.pop("ZFR_OVERRIDES", None); env.pop("ZFR_RUN_DIR", None)
    env.update(env_extra)
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_defaults_without_overrides():
    c = _config_in_subprocess({})
    assert c["spacing"] == 5.0 and c["sigma"] == 3.0 and c["rank"] == 0
    assert c["sigma_mod"] == c["sigma"]
    assert c["processed"].endswith("data/processed")


def test_overrides_reach_every_module():
    c = _config_in_subprocess({"ZFR_OVERRIDES": json.dumps(
        {"sample_spacing_m": 10, "dem_sigma_m": 0, "point_rank": 2})})
    assert c["spacing"] == 10 and c["rank"] == 2
    # the elevation module reads sigma from the config, not a local constant
    assert c["sigma"] == 0 and c["sigma_mod"] == 0


def test_unknown_override_keys_are_ignored():
    c = _config_in_subprocess({"ZFR_OVERRIDES": json.dumps({"nonsense": 1})})
    assert c["spacing"] == 5.0


def test_malformed_overrides_fail_loudly():
    with pytest.raises(subprocess.CalledProcessError):
        _config_in_subprocess({"ZFR_OVERRIDES": "{not json"})


def test_run_dir_redirects_processed_and_outputs_but_not_raw(tmp_path):
    c = _config_in_subprocess({"ZFR_RUN_DIR": str(tmp_path)})
    assert c["processed"] == str(tmp_path / "processed")
    assert c["outputs"] == str(tmp_path / "outputs")
    # raw data is shared: an experiment must never re-download the sources
    assert c["raw"].endswith("data/raw")


# ------------------------------------------------------------- the writer
def _fake_row(tag, **over):
    r = {
        "tag": tag, "params": {},
        "shortest_detour_pct": 0.0, "shortest_gain_saved_pct": 0.0,
        "shortest_gain_m": 77.0, "shortest_max_grade_pct": 21.8,
        "shortest_dist_km": 4.9,
        "min_climb_detour_pct": 5.0, "min_climb_gain_saved_pct": 27.0,
        "min_climb_gain_m": 58.0, "min_climb_max_grade_pct": 15.0,
        "min_climb_dist_km": 5.2,
        "grade_averse_detour_pct": 29.0, "grade_averse_gain_saved_pct": 16.0,
        "grade_averse_gain_m": 66.0, "grade_averse_max_grade_pct": 8.3,
        "grade_averse_dist_km": 6.3,
        "balanced_detour_pct": 10.0, "balanced_gain_saved_pct": 20.0,
        "balanced_gain_m": 61.0, "balanced_max_grade_pct": 11.2,
        "balanced_dist_km": 5.4,
        "network_gain_per_km": 21.0,
        "grade_Stüssihofstatt": 15.7, "grade_Trittligasse": 22.7,
        "grade_Kirchgasse": 14.4, "grade_Kantonsschulstrasse": 12.1,
        "gainkm_Limmatquai": 2.9, "gainkm_Bahnhofstrasse": 2.0,
        "gainkm_Mythenquai": 1.5,
        "top_corridors": ["Schaffhauserstrasse - X", "Limmatquai - Y", "Manessestrasse"],
        "corridor_count": 34, "top_corridor_km": 8.1,
        "top_pass_m": 472.6, "top_pass_pairs": 114, "top_pass_nbhd": "Unterstrass",
        "n_passes": 33, "saddle_pass_m": 472.6, "saddle_pass_street": "Bucheggstrasse",
        "saddle_on_saddle": True, "saddle_bike_flat_high_m": 472.0,
        "dem_rms_m": 1.97,
    }
    r.update(over)
    return r


def test_writer_produces_csv_and_markdown(tmp_path, monkeypatch):
    from zrh_flat_routes import sensitivity as S
    monkeypatch.setattr(S, "SENS_CSV", tmp_path / "s.csv")
    monkeypatch.setattr(S, "SENS_MD", tmp_path / "s.md")
    monkeypatch.setattr(S, "RUNS_DIR", tmp_path / "no-runs")   # no run dirs
    df = pd.DataFrame([
        _fake_row("baseline"),
        _fake_row("spacing_10m", min_climb_gain_saved_pct=36.0,
                  top_corridors=["Schaffhauserstrasse - X", "Other", "Manessestrasse"]),
    ]).set_index("tag")
    S._write(df)
    md = (tmp_path / "s.md").read_text()
    assert "| baseline |" in md and "| spacing_10m |" in md
    assert "coarser DEM sampling" in md          # the reason column
    assert "| 2/12 |" in md                      # two lead streets kept
    assert "Other" in md                         # the street that entered
    csv = pd.read_csv(tmp_path / "s.csv", index_col="tag")
    assert "top_corridors" not in csv.columns    # list column dropped from CSV
    assert csv.loc["spacing_10m", "min_climb_gain_saved_pct"] == 36.0
    assert csv.loc["spacing_10m", "lead_streets_shared"] == 2
    assert pd.isna(csv.loc["spacing_10m", "edge_overlap_pct"])  # no run dirs


def test_jaccard():
    from zrh_flat_routes.sensitivity import _jaccard
    assert _jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert _jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert _jaccard([], []) == 1.0


def test_edge_overlap_is_length_weighted():
    from zrh_flat_routes.sensitivity import _edge_overlap, _lead_streets
    a = {1: 100.0, 2: 300.0}
    b = {2: 300.0, 3: 100.0}
    # shared 300 of a 500 m union
    assert _edge_overlap(a, b) == pytest.approx(60.0)
    assert _edge_overlap(a, a) == 100.0
    assert _edge_overlap({}, {}) == 100.0
    import math
    assert math.isnan(_edge_overlap(None, b))
    assert _lead_streets(["Limmatquai - Bahnhofquai", "Manessestrasse", ""]) == {
        "Limmatquai", "Manessestrasse"}


def test_grid_is_one_at_a_time_around_the_baseline():
    from zrh_flat_routes.sensitivity import BASELINE, GRID
    tags = [t for t, _, _ in GRID]
    assert tags[0] == "baseline" and len(set(tags)) == len(tags)
    for tag, over, why in GRID[1:]:
        assert len(over) == 1, f"{tag} perturbs more than one parameter"
        (k, v), = over.items()
        assert k in BASELINE and v != BASELINE[k]
        assert why
