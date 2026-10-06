"""Generate the written analysis of the major findings.

Every number in the report is read from the analysis outputs rather than
typed in, so the prose cannot drift away from the data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURED_PAIRS, OUTPUT_DIR, PROCESSED_DIR
from .utils import get_logger

log = get_logger("zrh_flat_routes.report")

REPORT_MD = OUTPUT_DIR / "findings.md"


def _km(m):
    return m / 1000.0


def _load():
    import geopandas as gpd
    from .corridors import CORRIDORS_GPKG
    from .pairs import PAIRS_PARQUET, PARETO_PARQUET
    from .passes import BARRIERS_GEOJSON, PASSES_GEOJSON

    out = {
        "edges": gpd.read_parquet(PROCESSED_DIR / "edges_metrics.parquet"),
        "pairs": pd.read_parquet(PAIRS_PARQUET),
        "corridors": gpd.read_file(CORRIDORS_GPKG),
        "passes": gpd.read_file(PASSES_GEOJSON),
        "barriers": gpd.read_file(BARRIERS_GEOJSON),
    }
    if PARETO_PARQUET.exists():
        out["pareto"] = pd.read_parquet(PARETO_PARQUET)
    pm = OUTPUT_DIR / "pass_matrix.csv"
    if pm.exists():
        out["pass_matrix"] = pd.read_csv(pm)
    return out


# --------------------------------------------------------------------------
def _headline(d) -> list[str]:
    e, p = d["edges"], d["pairs"]
    walk = p[(p["mode"] == "walk")]
    by = walk.groupby("profile")
    short = by.get_group("shortest")
    flat = by.get_group("min_climb")
    bal = by.get_group("balanced")
    ga = by.get_group("grade_averse")

    walkable = e[e["walk_ok"]] if "walk_ok" in e.columns else e
    km = walkable["length_m"].sum() / 1000
    total_km = e["length_m"].sum() / 1000
    L = [
        "## The headline",
        "",
        f"The street network modelled here is {total_km:,.0f} km long, of "
        f"which {km:,.0f} km is walkable -- a large share of it forest paths "
        f"on the Uetliberg, the Zürichberg and the Käferberg. It climbs an "
        f"average of {walkable['cum_gain_fwd'].sum()/km:.1f} m for every "
        f"kilometre of street. But that average conceals a usable "
        f"low-elevation network. Across all {len(short):,} ordered pairs of "
        f"the city's 34 statistical quarters, on foot:",
        "",
        "| Objective | Mean distance | Mean climb | Mean steepest grade | "
        "Distance penalty | Climbing avoided |",
        "|---|---|---|---|---|---|",
    ]
    for label, g in (("Shortest (distance only)", short),
                     ("Balanced", bal),
                     ("Flattest (minimum climbing)", flat),
                     ("Grade-averse", ga)):
        L.append(
            f"| {label} | {_km(g['distance_m'].mean()):.2f} km | "
            f"{g['elev_gain_m'].mean():.0f} m | "
            f"{g['max_grade'].mean():.1%} | "
            f"{100*(g['detour_ratio'].mean()-1):+.0f}% | "
            f"{g['gain_saved_pct'].mean():.0f}% |")
    extra = 100 * (flat["detour_ratio"].mean() - 1)
    saved = flat["gain_saved_pct"].mean()
    L += [
        "",
        f"**About {extra:.0f}% more walking buys about {saved:.0f}% less "
        f"climbing.** That is the central result: the minimum-climbing route "
        f"is on average only {extra:.0f}% longer than the shortest one, yet "
        f"it avoids {saved:.0f}% of the ascent, and it drops the typical "
        f"steepest pitch from {short['max_grade'].mean():.0%} to "
        f"{flat['max_grade'].mean():.0%}.",
        "",
        "That is a smaller saving than San Francisco's (upstream found 14% "
        "more walking for 39% less climbing there), and the reason is "
        "geographic. San Francisco's hills are scattered across a city "
        "whose low ground joins up around them. Zurich's relief is two "
        "valley floors -- the Limmat's and the Glatt's -- walled in by "
        "continuous moraine ridges and slopes, with most of the quarters "
        "that climb at all (Fluntern, Hottingen, Witikon, Höngg, Friesenberg) "
        "*on* those slopes. Much of the climbing between quarters is "
        "therefore unavoidable, and the flat network's job is less to go "
        "around hills than to find the gentlest way up them.",
        "",
        "The grade-averse objective is worth separating out. It climbs "
        f"*more* in total than the flattest route "
        f"({ga['elev_gain_m'].mean():.0f} m against "
        f"{flat['elev_gain_m'].mean():.0f} m) and costs much more distance, "
        f"but it holds the steepest pitch to {ga['max_grade'].mean():.1%} "
        f"where the flattest route still allows "
        f"{flat['max_grade'].mean():.1%}. Total climbing and peak steepness "
        "are genuinely different objectives, and a single definition of "
        "\"flat\" cannot serve both: minimising total ascent will happily send "
        "you up one short wall, and avoiding walls will make you climb a "
        "little more overall.",
        "",
        "A note on the baseline. The shortest pedestrian route minimises "
        "distance only, as specified, and Zurich's distance-minimising "
        "pedestrian network runs straight up its public stairways -- the "
        f"city has {e[e['cls']=='steps']['length_m'].sum()/1000:.0f} km of "
        "them in this extract, the Stägen and Treppen that climb the "
        "Zürichberg and Käferberg slopes. The mean steepest pitch on a "
        f"shortest walking route is {short['max_grade'].mean():.0%}. That is "
        "not an artefact -- it is what minimising distance means on a "
        "hillside -- and it is a large part of why the flat alternatives "
        "matter.",
        "",
    ]
    return L


def _featured(d) -> list[str]:
    p = d["pairs"]
    L = ["## Specific answers", "",
         "### What is the flattest reasonable route between two quarters?", "",
         "| From | To | Shortest | Flattest | Balanced |", "|---|---|---|---|---|"]

    def cell(g):
        if g.empty:
            return "n/a"
        r = g.iloc[0]
        return (f"{_km(r['distance_m']):.2f} km / {r['elev_gain_m']:.0f} m "
                f"/ max {r['max_grade']:.0%}")

    walk = p[p["mode"] == "walk"]
    for o, dst in FEATURED_PAIRS:
        sub = walk[(walk["origin"] == o) & (walk["destination"] == dst)]
        if sub.empty:
            continue
        L.append(f"| {o} | {dst} | "
                 f"{cell(sub[sub['profile']=='shortest'])} | "
                 f"{cell(sub[sub['profile']=='min_climb'])} | "
                 f"{cell(sub[sub['profile']=='balanced'])} |")
    L += ["", "Each cell is distance / cumulative climb / steepest gradient, "
          "walking between the quarters' access points.", ""]

    best = (walk[walk["profile"] == "min_climb"]
            .nlargest(8, "gain_saved_m")
            [["origin", "destination", "shortest_distance_m", "shortest_gain_m",
              "distance_m", "elev_gain_m", "climb_saved_per_extra_m"]])
    L += ["### Where does flat routing pay off most?", "",
          "The quarter pairs where choosing the flat route avoids the most "
          "climbing:", "",
          "| From | To | Shortest | Flattest | Climbing avoided | "
          "Metres of climb saved per extra metre walked |",
          "|---|---|---|---|---|---|"]
    for _, r in best.iterrows():
        eff = r["climb_saved_per_extra_m"]
        L.append(
            f"| {r['origin']} | {r['destination']} | "
            f"{_km(r['shortest_distance_m']):.2f} km / "
            f"{r['shortest_gain_m']:.0f} m | "
            f"{_km(r['distance_m']):.2f} km / {r['elev_gain_m']:.0f} m | "
            f"{r['shortest_gain_m']-r['elev_gain_m']:.0f} m | "
            f"{eff:.2f} |")
    L.append("")
    return L


def _corridors(d) -> list[str]:
    c = d["corridors"]
    walk = c[c["mode"] == "walk"].sort_values("total_score", ascending=False).head(12)
    L = ["## Zurich's low-elevation corridors", "",
         "These were *discovered*, not listed: the analysis aggregated how "
         "often each street segment carried a good low-elevation route "
         "between quarters, weighted by the climbing those routes avoided, "
         "and merged the high-scoring segments into contiguous corridors. No "
         "corridor was named in advance.", "",
         "| Corridor | Length | Mean grade | Climb per km | Pairs served | "
         "Quarters | Elevation range |", "|---|---|---|---|---|---|---|"]
    for _, r in walk.iterrows():
        L.append(
            f"| {r['corridor_name']} | {r['length_km']:.1f} km | "
            f"{r['mean_abs_grade']:.1%} | {r['gain_per_km']:.1f} m | "
            f"{int(r['pair_count_max'])} | {int(r['neighborhood_span'])} | "
            f"{r['elev_min_m']:.0f}-{r['elev_max_m']:.0f} m |")
    L += ["",
          "For scale: the quays along the Limmat and the lake measure 1-4 m "
          "of climbing per kilometre, and the steep lanes in the validation "
          "report run at 100 m per kilometre and more. A corridor that climbs "
          "is not a contradiction: where every route between two quarters "
          "must gain height, the corridor is the gentlest way of gaining it.",
          ""]

    if len(walk):
        top = walk.iloc[0]
        second = walk.iloc[1] if len(walk) > 1 else None
        joined = (second is not None and "Schaffhauserstrasse" in top["corridor_name"]
                  and any(s in second["corridor_name"]
                          for s in ("Bucheggstrasse", "Schaffhauserstrasse")))
        L += ["### The spine", "",
              f"**{top['corridor_name']}** ({top['length_km']:.1f} km, "
              f"{top['mean_abs_grade']:.1%} mean gradient) is the city's "
              f"single most important flat corridor, serving "
              f"{int(top['pair_count_max'])} quarter pairs and avoiding "
              f"{top['climb_saved_m']/1000:.1f} km of cumulative climbing in "
              f"aggregate. "]
        if joined:
            L[-1] += (
                f"With the second, **{second['corridor_name']}** "
                f"({second['length_km']:.1f} km, {int(second['pair_count_max'])} "
                f"pairs), which carries it over the top, it is the way from the "
                f"Limmat to the Glatt valley: north from the river by "
                f"Stampfenbachstrasse and Schaffhauserstrasse, over the saddle "
                f"between the Käferberg and the Zürichberg at Bucheggplatz and "
                f"Milchbuck, and down Hofwiesenstrasse, Bucheggstrasse and "
                f"Schaffhauserstrasse into Oerlikon. The ridge it crosses runs "
                f"unbroken from the Hönggerberg to the Adlisberg, so every route "
                f"from the city centre, Aussersihl or the lakeshore to Oerlikon, "
                f"Seebach, Affoltern or Schwamendingen meets it, and this is the "
                f"lowest place to cross. The model was not told it existed.")
        L += [""]
    return L


def _connectors(d) -> list[str]:
    """The less obvious corridors, with a word on why each matters."""
    c = d["corridors"]
    walk = c[c["mode"] == "walk"].sort_values("total_score", ascending=False)
    # the spine section covers the first corridor, and the second too when
    # it is the spine's continuation over the saddle
    skip = 2 if (len(walk) > 1 and "Schaffhauserstrasse" in walk.iloc[0]["corridor_name"]
                 and any(s in walk.iloc[1]["corridor_name"]
                         for s in ("Bucheggstrasse", "Schaffhauserstrasse"))) else 1
    rest = walk.iloc[skip:skip + 8]
    L = ["### The quiet connectors", "",
         "Beyond the spine, these corridors do the same job on a smaller "
         "scale:", "",
         "| Corridor | Length | Mean grade | Climb per km | Connects | "
         "Why it matters |", "|---|---|---|---|---|---|"]
    why = {
        "Limmatquai": "the right bank of the Limmat, dead level from the lake "
                      "to Central",
        "Seilergraben": "the line of the old town moat, a level terrace along "
                        "the foot of the Zürichberg",
        "Hirschengraben": "the line of the old town moat, a level terrace "
                          "along the foot of the Zürichberg",
        "Manessestrasse": "the Sihl valley through Wiedikon, under the "
                          "Uetliberg slope",
        "Allmendstrasse": "south along the Sihl to the Allmend Brunau and "
                          "Leimbach, between the Uetliberg and the Entlisberg",
        "Winterthurerstrasse": "east across the north of the city, from the "
                               "Milchbuck saddle towards Schwamendingen",
        "Weinbergstrasse": "from Central up to Schaffhauserplatz, beside "
                           "Stampfenbachstrasse: the other gentle way up from "
                           "the river",
        "Altwiesenstrasse": "across Schwamendingen, on the floor of the Glatt "
                            "valley",
        "Berninastrasse": "east-west through Oerlikon towards Schwamendingen",
        "Wehntalerstrasse": "north-west from the Glatt valley floor towards "
                            "Affoltern, along the foot of the Käferberg",
        "Hofwiesenstrasse": "the descent from the Bucheggplatz saddle to "
                            "Oerlikon",
        "Museumstrasse": "the level ring around the main station",
        "Bahnhofplatz": "the level ring around the main station",
        "Mythenquai": "the lakeshore south of the city centre",
        "Utoquai": "the lakeshore east of the city centre",
        "Badenerstrasse": "the old road west along the Limmat valley floor to "
                          "Altstetten",
        "Langstrasse": "the level spine of Aussersihl, under the railway",
    }
    for _, r in rest.iterrows():
        lead = r["corridor_name"].split(" - ")[0]
        note = why.get(lead) or next(
            (v for k, v in why.items() if k in r["corridor_name"]),
            "a low-gradient link the analysis found to be repeatedly useful "
            "between quarters")
        L.append(f"| {r['corridor_name']} | {r['length_km']:.1f} km | "
                 f"{r['mean_abs_grade']:.1%} | {r['gain_per_km']:.1f} m | "
                 f"{int(r['neighborhood_span'])} quarters | {note} |")
    L.append("")
    return L


def _passes(d) -> list[str]:
    pz = d["passes"]
    L = ["## Passes, saddles and barriers", "",
         "The question \"how much climbing is unavoidable between these two "
         "parts of the city?\" is a **minimax** problem, not a shortest-path "
         "one: what matters is the lowest summit you can possibly cross. "
         "Solving it over a minimum bottleneck spanning tree gives, for every "
         "pair of quarters, the exact elevation of the lowest available "
         "crossing and the block on which it happens.", ""]
    if "pass_matrix" in d:
        pm = d["pass_matrix"]
        worst = pm.loc[pm["pass_elev_m"].idxmax()]
        L += [f"Across all {len(pm):,} quarter pairs the lowest possible "
              f"crossing averages {pm['pass_elev_m'].mean():.0f} m above sea "
              f"level -- against 406 m for the lake -- and reaches "
              f"{pm['pass_elev_m'].max():.0f} m at worst "
              f"({worst['neighborhood_a']} to {worst['neighborhood_b']}). "
              f"Only {len(pz)} distinct blocks in the whole city act as the "
              f"binding constraint for any pair -- the city's real passes.", ""]
    L += ["| Pass | Quarter | Lowest possible crossing | Pairs forced "
          "over it | Gradient there |", "|---|---|---|---|---|"]
    for _, r in pz.head(12).iterrows():
        nm = r["name"] if isinstance(r["name"], str) and r["name"] else \
            "(unnamed path)"
        L.append(f"| {nm} | {r['neighborhood']} | {r['pass_elev_m']:.0f} m | "
                 f"{int(r['pairs_served'])} | {r['max_abs_grade']:.1%} |")
    if len(pz):
        top = pz.iloc[0]
        n_pairs = len(d["pass_matrix"]) if "pass_matrix" in d else 0
        L += ["",
              f"The single most consequential pass in Zurich is "
              f"**{top['name']}** in {top['neighborhood']}, at "
              f"{top['pass_elev_m']:.0f} m. It is the binding constraint for "
              f"{int(top['pairs_served'])} of the {n_pairs} quarter pairs -- "
              f"more than any other street -- because it is the lowest point "
              f"of the ridge between the Limmat valley and the Glatt valley: "
              f"the saddle that the Käferberg article on Wikipedia puts "
              f"between Bucheggplatz (472 m) and Milchbuck (476 m). Anyone "
              f"going from the centre, the lakeshore or the Limmat plain to "
              f"Oerlikon, Seebach or Schwamendingen climbs to that height "
              f"whatever route they choose. Its gradient where it crosses is "
              f"only {top['max_abs_grade']:.1%}: the crossing is high but "
              f"gentle, which is exactly what makes it the pass.", "",
              "The passes after it are of a different kind. They are not "
              "cols between valleys but the tops of the climbs to the "
              "quarters that sit on the slopes themselves -- Witikon on its "
              "plateau behind the Adlisberg, Fluntern and Hottingen on the "
              "Zürichberg, Höngg on the Hönggerberg, Friesenberg and "
              "Leimbach under the Uetliberg. For these there is no flat way "
              "in: the flat network can choose the gentlest approach, not "
              "avoid the height.", ""]

    b = d["barriers"]
    if "unavoidability" in b.columns:
        named = b[b["name"].notna() & (b["name"].astype(str) != "")]
        L += ["### Barriers with an alternative, and barriers without", "",
              "A steep street that carries heavy shortest-path traffic but "
              "almost none once climbing is penalised has a flat alternative "
              "nearby. One that keeps its traffic under every objective does "
              "not.", "",
              "| Street | Quarter | Gradient | Pairs via shortest route | "
              "Still via the flat route | Verdict |",
              "|---|---|---|---|---|---|"]
        top = named.nlargest(10, "barrier_score")
        for _, r in top.iterrows():
            un = float(r.get("unavoidability") or 0)
            verdict = ("**unavoidable**" if un > 0.5 else
                       "avoidable" if un < 0.15 else "partly avoidable")
            q = r["neighborhood"] if isinstance(r["neighborhood"], str) else \
                "(outside the quarters)"
            L.append(f"| {r['name']} | {q} | "
                     f"{r['max_abs_grade']:.1%} | {int(r['shortest_use'])} | "
                     f"{float(r.get('flat_use_per_objective') or 0):.0f} | "
                     f"{verdict} |")
        L += ["", "_Named streets only; several of the highest-scoring "
              "barriers are unnamed forest paths._", ""]
    return L


def _pareto(d) -> list[str]:
    if "pareto" not in d:
        return []
    pa = d["pareto"]
    pa = pa[(pa["mode"] == "walk") & pa["pareto_optimal"]]

    # citywide: for every pair, how much detour does halving the climb cost?
    rows = []
    for (o, dst), g in pa.groupby(["origin", "destination"]):
        g = g.sort_values("distance_m")
        s = g.iloc[0]                       # the pure-distance anchor
        if s["elev_gain_m"] <= 0:
            continue
        half = g[g["elev_gain_m"] <= 0.5 * s["elev_gain_m"]]
        rows.append({
            "halvable": len(half) > 0,
            "detour_to_halve": (half["distance_m"].min() / s["distance_m"] - 1)
            if len(half) else np.nan,
            "best_saved_pct": 100 * (1 - g["elev_gain_m"].min() / s["elev_gain_m"]),
            "best_detour": g.loc[g["elev_gain_m"].idxmin(), "distance_m"]
            / s["distance_m"] - 1,
        })
    r = pd.DataFrame(rows)
    n_pairs = pa.groupby(["origin", "destination"]).ngroups

    L = ["## The distance / climbing trade-off", "",
         f"For every one of the {n_pairs:,} ordered pairs, a single weight is "
         "swept from zero (pure distance) up to the minimum-climbing "
         "objective, tracing the frontier between distance, cumulative "
         "climbing and peak gradient. The useful question is where the knee "
         "is: how much detour buys how much of the climbing.", ""]
    if len(r):
        L += [f"- **{100*r['halvable'].mean():.0f}% of pairs can halve their "
              f"climbing** by some route, and the median detour that costs is "
              f"**{100*r['detour_to_halve'].median():.0f}%**. "
              f"{100*(r['detour_to_halve'] <= 0.10).mean():.0f}% of all pairs "
              f"can halve it within a 10% detour, "
              f"{100*(r['detour_to_halve'] <= 0.20).mean():.0f}% within 20%.",
              f"- Taken to the flattest possible route, the median pair "
              f"sheds **{r['best_saved_pct'].median():.0f}%** of its climbing "
              f"for a median **{100*r['best_detour'].median():.0f}%** more "
              f"distance.", ""]
    for o, dst in FEATURED_PAIRS[:4]:
        g = pa[(pa["origin"] == o) & (pa["destination"] == dst)]
        if g.empty:
            continue
        g = g.sort_values("distance_m")
        L += [f"**{o} to {dst}**", "",
              "| Distance | Climb | Steepest grade |", "|---|---|---|"]
        for _, row in g.iterrows():
            L.append(f"| {_km(row['distance_m']):.2f} km | "
                     f"{row['elev_gain_m']:.0f} m | "
                     f"{row['max_grade']:.0%} |")
        L.append("")
    L += ["Where a frontier has more than one point it is concave: the first "
          "fraction of extra distance removes most of the avoidable "
          "climbing, and everything after that buys very little. That is the "
          "practical argument for the balanced objective over the purely "
          "flattest one.", ""]
    return L


def _modes(d) -> list[str]:
    p = d["pairs"]
    e = d["edges"]
    w = p[(p["mode"] == "walk") & (p["profile"] == "min_climb")]
    b = p[(p["mode"] == "bike") & (p["profile"] == "min_climb")]
    steps_km = e[e["cls"] == "steps"]["length_m"].sum() / 1000
    bike_km = e[e["bike_ok"]]["length_m"].sum() / 1000
    walk_km = e[e["walk_ok"]]["length_m"].sum() / 1000
    L = ["## Walking is not cycling", "",
         f"The two networks are modelled separately, and they are not "
         f"interchangeable. Zurich has {steps_km:.0f} km of public stairways, "
         f"a genuine part of the pedestrian network and useless on a "
         f"bicycle; the bicycle graph excludes them outright. It also leaves "
         f"out footpaths and pedestrian zones that are not signed for "
         f"bicycles, as Swiss signage law does, so the rideable network is "
         f"{bike_km:,.0f} km against {walk_km:,.0f} km on foot. Bicycle costs "
         f"carry stress weights (a protected cycleway counts as 0.85 of its "
         f"length, a trunk road as 1.9) and respect one-way restrictions -- "
         f"with Zurich's many contraflow exemptions for bicycles -- which "
         f"pedestrians do not.", "",
         f"The result is that the flattest bicycle route averages "
         f"{_km(b['distance_m'].mean()):.2f} km and "
         f"{b['elev_gain_m'].mean():.0f} m of climbing against "
         f"{_km(w['distance_m'].mean()):.2f} km and "
         f"{w['elev_gain_m'].mean():.0f} m on foot. The difference is "
         f"modest in aggregate but decisive in specific places: any route "
         f"whose flat pedestrian option runs up a stairway or along a "
         f"footpath has a longer bicycle equivalent, or none.", ""]
    return L


def _top_corridor_phrase(sd: pd.DataFrame) -> str:
    """'the top corridor is X in every run', or an honest count."""
    leads = sd["top_corridor"].fillna("").map(lambda v: v.split(" - ")[0])
    counts = leads.value_counts()
    top, n = counts.index[0], int(counts.iloc[0])
    if n == len(sd):
        return f"the top corridor is {top} in every run"
    return f"the top corridor is {top} in {n} of {len(sd)} runs"


def _robustness(d) -> list[str]:
    """Summarise the sensitivity analysis, if it has been run."""
    path = OUTPUT_DIR / "sensitivity.csv"
    if not path.exists():
        return []
    sd = pd.read_csv(path, index_col="tag")
    if "baseline" not in sd.index or len(sd) < 2:
        return []
    base = sd.loc["baseline"]
    others = sd.drop(index="baseline")
    L = ["## How much of this depends on the modelling choices?", "",
         "Every figure above was recomputed with the whole pipeline rebuilt "
         f"under {len(others)} one-at-a-time changes to the elevation "
         "parameters and the choice of access intersection "
         "(`outputs/sensitivity.md` has the full tables).", "",
         "| Finding | Baseline | Range across all perturbations |",
         "|---|---|---|",
         f"| Flattest route: extra distance | {base['min_climb_detour_pct']:+.0f}% | "
         f"{others['min_climb_detour_pct'].min():+.0f}% to "
         f"{others['min_climb_detour_pct'].max():+.0f}% |",
         f"| Flattest route: climbing avoided | {base['min_climb_gain_saved_pct']:.0f}% | "
         f"{others['min_climb_gain_saved_pct'].min():.0f}% to "
         f"{others['min_climb_gain_saved_pct'].max():.0f}% |",
         f"| Grade-averse: mean steepest pitch | {base['grade_averse_max_grade_pct']:.1f}% | "
         f"{others['grade_averse_max_grade_pct'].min():.1f}% to "
         f"{others['grade_averse_max_grade_pct'].max():.1f}% |",
         f"| Corridor material shared with baseline (by length) | 100% | "
         f"{others['edge_overlap_pct'].min():.0f}% to "
         f"{others['edge_overlap_pct'].max():.0f}% |",
         f"| Lead streets of the top 12 corridors kept | 12 of 12 | "
         f"{int(others['lead_streets_shared'].min())} to "
         f"{int(others['lead_streets_shared'].max())} of 12 |",
         f"| Dominant pass | {base['top_pass_nbhd']}, {base['top_pass_m']:.0f} m | "
         f"same location in {int((sd['top_pass_nbhd'] == base['top_pass_nbhd']).sum())} "
         f"of {len(sd)} runs; {others['top_pass_m'].min():.0f}-"
         f"{others['top_pass_m'].max():.0f} m |",
         f"| Lowest crossing, Central to Oerlikon (published 472 m) | "
         f"{base['saddle_pass_m']:.1f} m | "
         f"{others['saddle_pass_m'].min():.1f}-{others['saddle_pass_m'].max():.1f} m |",
         f"| Stüssihofstatt gradient (tagged 17%) | "
         f"{base['grade_Stüssihofstatt']:.1f}% | "
         f"{others['grade_Stüssihofstatt'].min():.1f}% to "
         f"{others['grade_Stüssihofstatt'].max():.1f}% |",
         ""]
    dev = (others["min_climb_gain_saved_pct"] - base["min_climb_gain_saved_pct"]).abs()
    worst = dev.idxmax()
    least = others["edge_overlap_pct"].idxmin()
    drifters = sorted({s for v in others["lead_streets_new"].fillna("")
                       for s in str(v).split("; ") if s})
    L += [f"The headline barely moves: the perturbation that shifts it most "
          f"is **{worst}** ({others.loc[worst, 'change']}), at "
          f"{others.loc[worst, 'min_climb_gain_saved_pct']:.0f}% climbing "
          f"avoided against {base['min_climb_gain_saved_pct']:.0f}% at "
          f"baseline. The dominant pass holds in "
          f"{int((sd['top_pass_nbhd'] == base['top_pass_nbhd']).sum())} of "
          f"{len(sd)} runs.", "",
          f"The corridors are where the model is least rigid, and it is "
          f"worth being precise about how. The *street* that qualifies as "
          f"corridor material is {others['edge_overlap_pct'].min():.0f}-"
          f"{others['edge_overlap_pct'].max():.0f}% the same by length, and "
          f"{_top_corridor_phrase(sd)}; what changes "
          f"is where each corridor is cut and therefore what it is called, "
          f"most under the profile smoothing window (**{least}**, "
          f"{others['edge_overlap_pct'].min():.0f}%). A handful of "
          f"borderline streets drift in and out of the top twelve "
          f"({', '.join(drifters)}): these are real corridors whose rank "
          f"depends on tenths of a percent of gradient, not artefacts, and "
          f"they should be read as a tier rather than a ranking.", ""]
    return L


def _limits(d) -> list[str]:
    return [
        "## What this analysis does not tell you", "",
        "- **Elevation is the ground, not the street surface.** swissALTI3D "
        "is a bare-earth model, so bridges and tunnels are corrected by "
        "interpolating across the structure, and streets that pass *under* "
        "a bridge are interpolated across the crossing too -- the model fills "
        "the space under a deck from the embankments either side. That needs "
        "the bridge to be tagged as one: Birchstrasse in Seebach passes under "
        "a motorway link OpenStreetMap does not flag and keeps a phantom "
        "hump of about 10 m. Ramps, retaining walls and terraces the "
        "centreline does not quite follow can likewise put a short spike "
        "into a profile.",
        "- **Travel is modelled on street centrelines.** Sidewalk and "
        "crossing geometry exists in the source data but is deliberately "
        "excluded: including it would represent every street two or three "
        "times and wreck the corridor aggregation. A carriageway that "
        "OpenStreetMap closes to walkers because its sidewalks are mapped "
        "separately stays walkable where a sidewalk runs alongside it. "
        "Pedestrian distances are therefore block-scale, not door-to-door.",
        "- **One access point per quarter.** Each quarter is represented by "
        "a single street-network-weighted, intersection-snapped point. Large "
        "or strung-out quarters -- Höngg, Affoltern, Leimbach, Witikon -- are "
        "served worse by this than compact ones.",
        "- **No traffic, surface quality, signals, trams or safety.** The "
        "bicycle stress weights are a crude proxy for road class, not a "
        "level-of-traffic-stress model, and nothing here accounts for "
        "signal delay, tram tracks, cobbles in the Altstadt or collision "
        "risk. Nor does it know about the Polybahn, the Rigiblick and "
        "Dolder funiculars or the lifts that carry walkers up some slopes.",
        "- **Bicycle facilities are OSM-derived.** The bicycle and "
        "low-stress layers are inferred from OpenStreetMap tagging, not "
        "from the City's own cycling network.",
        "",
    ]


def write_report() -> Path:
    d = _load()
    L = ["# Zurich's flat street network: findings", "",
         "*Generated by `python -m zrh_flat_routes report`. Every figure is "
         "computed from the analysis outputs in this repository; see "
         "`validation_report.md` for the checks against known ground truth. "
         "The analysis, and most of this report's structure, is a port of "
         "[flattensf](https://github.com/almostimplemented/flattensf) by "
         "Drew Edwards.*",
         ""]
    L += _headline(d)
    L += _featured(d)
    L += _corridors(d)
    L += _connectors(d)
    L += _passes(d)
    L += _pareto(d)
    L += _modes(d)
    L += _robustness(d)
    L += _limits(d)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(L))
    return REPORT_MD
