#!/usr/bin/env python3
"""Build one offline HTML report for every collected I2CE experiment.

Python 3.9+, pandas and plotly. Optional new experiments use the CSV schemas
in templates/. No simulator, notebook server, or network is used by this script.

From any directory:
  python /path/to/repo/transformer_profiling/report.py
  bash /path/to/repo/tools/exp/exp.sh report --output /tmp/profiling.html
"""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "final"
DEFAULT_OUTPUT = SCRIPT_DIR / "reports" / "profiling_report.html"
OPTIONAL_RESULTS = {"tile_results": "tile_results.csv", "scaling_results": "scaling_results.csv",
                    "phase_results": "phase_results.csv"}
# Loaded after argparse, so --help works before the dependencies are installed.
pd = None


def load_dependencies():
    global pd
    import pandas as pd
    from plotly.offline import get_plotlyjs
    return get_plotlyjs

STAGES = ["MHA", "Projection", "non_GEMM_after_projection", "FF1", "FF2", "non_GEMM_after_ff2"]
SHORT = ["MHA", "Projection", "Non-GEMM 1", "FF1", "FF2", "Non-GEMM 2"]
COLORS = ["#2762ad", "#3d9db5", "#b8c9d9", "#ed9e38", "#9865b0", "#85958a"]
WARM = [[0, "#fff7ec"], [0.25, "#fee8c8"], [0.5, "#fdbb84"], [0.75, "#ef6548"], [1, "#b30000"]]
BLUE = [[0, "#edf6fa"], [0.35, "#9dccdf"], [0.7, "#3b88b6"], [1, "#123c70"]]
METADATA = ["model", "d_q", "d_seq", "d_model", "num_head", "d_ff", "implementation", "n_learners", "codebook_size", "sve_bits"]


def clean(obj):
    """JSON-safe data, retaining missing observations as null rather than zero."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "item"):
        return clean(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def num(x):
    return "—" if pd.isna(x) else "{:g}".format(x)


def input_files(path):
    """Find result files only; ignore manifests, metric maps and generated figures."""
    if not path.is_dir():
        return [path]
    combined = path / "final_all_experiments.tsv"
    experiments = sorted(p for p in path.iterdir() if p.is_file()
                         and re.fullmatch(r"e\d+(?:_.*)?\.tsv", p.name, re.IGNORECASE))
    files = ([combined] if combined.is_file() else []) + experiments
    if not files:
        raise ValueError(str(path) + ": no experiment TSVs found.")
    return files


def read_results(path):
    """Individual experiment files override their rows in a stale combined table."""
    files = input_files(path)
    if not path.is_dir():
        return pd.read_csv(files[0], sep="\t")
    combined, parts, seen = None, [], set()
    for source in files:
        data = pd.read_csv(source, sep="\t")
        if "exp_id" not in data or data.empty or data.exp_id.isna().any():
            raise ValueError(str(source) + ": missing experiment identifiers.")
        if source.name == "final_all_experiments.tsv":
            combined = data
            continue
        ids = set(data.exp_id)
        expected = re.match(r"e(\d+)", source.name, re.IGNORECASE).group(1)
        if len(ids) != 1 or ids != {"E" + expected}:
            raise ValueError(str(source) + ": filename and experiment ID must match.")
        if ids & seen:
            raise ValueError("Duplicate per-experiment files for " + ", ".join(sorted(ids & seen)))
        seen.update(ids)
        parts.append(data)
    if combined is not None:
        parts.append(combined.loc[~combined.exp_id.isin(seen)])
    return pd.concat(parts, ignore_index=True)


def load_data(path):
    frame = read_results(Path(path))
    required = {"exp_id", "row_kind", "interval", "sim_seconds", "instructions", "cpu_cycles",
                "dcache_demand_accesses", "dcache_demand_misses", "l2_demand_accesses", "l2_demand_misses"} | set(METADATA)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Missing TSV columns: " + ", ".join(sorted(missing)))
    if not frame.row_kind.isin(["interval_delta", "final_total"]).all():
        raise ValueError("Unexpected row_kind; expected interval_delta or final_total.")
    if frame.empty or frame[["exp_id", "interval"]].isna().any().any():
        raise ValueError("Missing experiment or interval identifiers.")
    if frame.duplicated(["exp_id", "row_kind", "interval"]).any():
        raise ValueError("Duplicate experiment/interval rows; do not combine two runs under one exp_id.")
    numeric = ["sim_seconds", "instructions", "cpu_cycles", "dcache_demand_accesses", "dcache_demand_misses",
               "l2_demand_accesses", "l2_demand_misses", "n_learners", "codebook_size", "sve_bits"]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame[["sim_seconds", "instructions", "cpu_cycles"]].isna().any().any():
        raise ValueError("Missing time, instruction, or cycle counters.")
    if (frame[numeric].drop(columns=["codebook_size", "sim_seconds"]) < 0).any().any():
        raise ValueError("Negative counters: check cumulative dump subtraction/reset boundaries.")
    for col in numeric:
        if not frame[col].dropna().map(math.isfinite).all():
            raise ValueError("Non-finite numeric value in " + col)
    if frame[METADATA].drop(columns="codebook_size").isna().any().any():
        raise ValueError("Missing experiment configuration fields.")
    if not frame.exp_id.str.fullmatch(r"E\d+").all():
        raise ValueError("Expected experiment IDs such as E01 or E37.")
    for level, prefix in [("l1d", "dcache"), ("l2", "l2")]:
        misses = frame[prefix + "_demand_misses"]
        accesses = frame[prefix + "_demand_accesses"]
        if (misses > accesses).any():
            raise ValueError(prefix + ": demand misses exceed demand accesses.")
        frame[level + "_miss_pct"] = 100 * misses / accesses.where(accesses > 0)
        frame[level + "_mpki"] = 1000 * misses / frame.instructions.where(frame.instructions > 0)
    frame["ipc_calc"] = frame.instructions / frame.cpu_cycles.where(frame.cpu_cycles > 0)
    frame["exp_order"] = pd.to_numeric(frame.exp_id.str.extract(r"(\d+)")[0], errors="raise")
    if frame[["exp_id", "exp_order"]].drop_duplicates().exp_order.duplicated().any():
        raise ValueError("Ambiguous IDs such as E01 and E1 refer to the same experiment number.")
    # This is a documented property of the historical dataset, not a new estimate.
    frame["estimated"] = (frame.exp_id.isin(["E05", "E06"]) & (frame.model == "BERT-base")
                          & (frame.implementation == "Dense naive/default")
                          & frame.get("gem5_timestamp", pd.Series("", index=frame.index)).isin(
                              ["run_20260516_021425", "run_20260516_021427", "run_20260516_021428"]))
    frame["label"] = frame.exp_id + frame.estimated.map({True: " (est.)", False: ""})
    frame["detail"] = [html.escape("{} · {} · learners={} · CB={} · SVE={} bit · dims(q,seq,model,heads,ff)=({},{},{},{},{}){}".format(
        r.exp_id, r.implementation, num(r.n_learners), num(r.codebook_size), num(r.sve_bits),
        num(r.d_q), num(r.d_seq), num(r.d_model), num(r.num_head), num(r.d_ff),
        " · estimated total" if r.estimated else "")) for r in frame.itertuples()]
    totals = frame.loc[frame.row_kind == "final_total"].sort_values("exp_order").copy()
    stages = frame.loc[frame.row_kind == "interval_delta"].sort_values("exp_order").copy()
    if totals.exp_id.duplicated().any() or set(totals.exp_id) != set(stages.exp_id):
        raise ValueError("Each experiment needs exactly one total and its corresponding stage rows.")
    for exp_id, group in frame.groupby("exp_id"):
        if (group[METADATA].nunique(dropna=False) > 1).any():
            raise ValueError(exp_id + ": configuration differs between stage and total rows.")
    for exp_id, group in stages.groupby("exp_id"):
        if set(group.interval) != set(STAGES):
            raise ValueError(exp_id + ": expected the six existing profiling regions.")
    sums = stages.groupby("exp_id").sim_seconds.sum()
    reference = totals.set_index("exp_id").sim_seconds
    if not reference.map(math.isfinite).all() or (reference <= 0).any():
        raise ValueError("Invalid final_total simulated time.")
    invalid_times = set(stages.loc[(stages.sim_seconds < 0) | ~stages.sim_seconds.map(math.isfinite), "exp_id"])
    invalid_times.update(stages.loc[stages.sim_seconds > stages.exp_id.map(reference) + 1e-5, "exp_id"])
    invalid_times.update(sums.index[(sums - reference).abs() > 1e-5 + 1e-8 * reference.abs()])
    frame["phase_timing_valid"] = ~frame.exp_id.isin(invalid_times)
    frame.loc[~frame.phase_timing_valid, "detail"] += " · invalid source stage timings; source total retained"
    if invalid_times:
        print("Source warning: invalid stage timings excluded from phase bars:", ", ".join(sorted(invalid_times)), file=sys.stderr)
    totals = frame.loc[frame.row_kind == "final_total"].sort_values("exp_order").copy()
    stages = frame.loc[frame.row_kind == "interval_delta"].sort_values("exp_order").copy()
    return frame, totals, stages


def layout(title, height=500, **kwargs):
    base = dict(title=dict(text=title, x=0.02, xanchor="left", y=0.98, yanchor="top", font=dict(size=18)),
                height=height, margin=dict(l=80, r=60, t=110 if kwargs.get("updatemenus") else 65, b=90),
                paper_bgcolor="white", plot_bgcolor="white",
                font=dict(family="Arial, sans-serif", color="#25364a", size=12),
                xaxis=dict(zeroline=False, gridcolor="#edf1f5", automargin=True),
                yaxis=dict(zeroline=False, gridcolor="#edf1f5", automargin=True),
                legend=dict(orientation="h", y=-0.20, x=0))
    base.update(kwargs)
    return base


def log_ticks(values):
    low, high = math.floor(math.log10(values.min())), math.ceil(math.log10(values.max()))
    ticks = [multiple * 10 ** power for power in range(low, high + 1) for multiple in [1, 2, 5]]
    return dict(type="log", tickmode="array", tickvals=ticks, ticktext=[num(v) for v in ticks])


def switch_menu(buttons):
    return [dict(type="dropdown", x=0, y=1.015, xanchor="left", yanchor="bottom", buttons=buttons)]


def phase_chart(total, stage, model):
    total = total.loc[total.phase_timing_valid]
    ids = total.exp_id.tolist()
    labels = total.label.tolist()
    pivot = stage.pivot(index="exp_id", columns="interval", values="sim_seconds").reindex(ids)[STAGES]
    pivot.loc[total.loc[~total.phase_timing_valid, "exp_id"], :] = float("nan")
    seconds = [pivot[s].tolist() for s in STAGES]
    percent = [(100 * pivot[s] / pivot.sum(axis=1).where(pivot.sum(axis=1) > 0)).tolist() for s in STAGES]
    traces = []
    for i, name in enumerate(SHORT):
        custom = [[a, b, d] for a, b, d in zip(seconds[i], percent[i], total.detail)]
        traces.append(dict(type="bar", orientation="h", name=name, x=seconds[i], y=labels,
                           marker=dict(color=COLORS[i]), customdata=custom,
                           hovertemplate="%{customdata[2]}<br>" + name + ": %{customdata[0]:.4f} s (%{customdata[1]:.1f}%)<extra></extra>"))
    buttons = [dict(label="Simulated seconds", method="update", args=[dict(x=seconds), {"xaxis.title.text": "Simulated time (s)", "xaxis.range": None, "xaxis.autorange": True}]),
               dict(label="Share of total (%)", method="update", args=[dict(x=percent), {"xaxis.title.text": "Share of measured regions (%)", "xaxis.range": [0, 100], "xaxis.autorange": False}])]
    lay = layout(model + " · phase breakdown", max(600, 27 * len(ids) + 180), barmode="stack", updatemenus=switch_menu(buttons))
    lay["xaxis"].update(title=dict(text="Simulated time (s)"))
    lay["yaxis"].update(autorange="reversed", categoryorder="array", categoryarray=labels)
    return dict(data=traces, layout=lay)


def cache_chart(total, stage, model):
    ids = total.exp_id.tolist()
    def values(column):
        return stage.pivot(index="interval", columns="exp_id", values=column).reindex(index=STAGES, columns=ids).values.tolist()
    options = [("l2_miss_pct", "L2 demand miss rate (%)", 0, 100), ("l1d_miss_pct", "L1D demand miss rate (%)", 0, 100),
               ("l2_mpki", "L2 demand MPKI", None, None), ("l1d_mpki", "L1D demand MPKI", None, None),
               ("ipc_calc", "IPC", None, None), ("l2_demand_misses", "L2 demand misses", None, None)]
    traces = []
    for k, (col, name, zmin, zmax) in enumerate(options):
        traces.append(dict(type="heatmap", x=total.label.tolist(), y=SHORT, z=values(col),
                           visible=k == 0, zmin=zmin, zmax=zmax, zauto=zmin is None, colorscale=WARM,
                           xgap=1, ygap=1, hoverongaps=False, colorbar=dict(title=dict(text=name), thickness=14),
                           hovertemplate="%{x}<br>%{y}<br>" + name + ": %{z:.2f}<extra></extra>"))
    buttons = [dict(label=name, method="update", args=[dict(visible=[i == j for i in range(len(options))])])
               for j, (_, name, _, _) in enumerate(options)]
    lay = layout(model + " · cache and instruction metrics by phase", 440, updatemenus=switch_menu(buttons))
    lay["yaxis"].update(autorange="reversed")
    lay["xaxis"].update(title=dict(text="Experiment"), tickangle=-45, type="category")
    return dict(data=traces, layout=lay)


def sve_chart(total, model):
    data = total.loc[(total.implementation == "Codebook SIMD int8") & (total.codebook_size == 8)]
    traces, seconds, speedups = [], [], []
    for i, (learners, group) in enumerate(data.groupby("n_learners")):
        group = group.sort_values("sve_bits")
        baseline = group.loc[group.sve_bits == 128, "sim_seconds"]
        comparable = (not group.sve_bits.duplicated().any() and
                      (group[["d_q", "d_seq", "d_model", "num_head", "d_ff"]].nunique(dropna=False) == 1).all())
        speeds = (float(baseline.iloc[0]) / group.sim_seconds).tolist() if comparable and len(baseline) == 1 else [None] * len(group)
        seconds.append(group.sim_seconds.tolist())
        speedups.append(speeds)
        traces.append(dict(type="scatter", mode="lines+markers" if comparable else "markers", x=group.sve_bits.tolist(), y=seconds[-1],
                           name="{} learner{}".format(num(learners), "s" if learners != 1 else ""),
                           line=dict(color=[COLORS[0], COLORS[1], COLORS[4]][i % 3], width=3), marker=dict(size=9),
                           customdata=[[d, t, s] for d, t, s in zip(group.detail, group.sim_seconds, speeds)],
                           hovertemplate="%{customdata[0]}<br>%{customdata[1]:.4f} s<br>%{customdata[2]:.3f}× vs same learner count at 128 bit<extra></extra>"))
    buttons = [dict(label="Simulated seconds", method="update", args=[dict(y=seconds), {"yaxis.title.text": "Simulated time (s)"}]),
               dict(label="Speedup vs 128-bit SVE", method="update", args=[dict(y=speedups), {"yaxis.title.text": "Speedup vs same workload at 128 bit"}])]
    lay = layout(model + " · SVE width sensitivity · codebook size 8", 470, updatemenus=switch_menu(buttons))
    lay["xaxis"].update(title=dict(text="SVE width (bits)"), tickvals=[128, 256, 512])
    lay["yaxis"].update(title=dict(text="Simulated time (s)"), rangemode="tozero")
    if not traces:
        lay["annotations"] = [dict(text="No codebook-size-8 observations in this selection", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)]
    return dict(data=traces, layout=lay)


def codebook_chart(total, model):
    data = total.loc[(total.implementation == "Codebook SIMD int8") & (total.n_learners == 4)]
    repeated = data.duplicated(["codebook_size", "sve_bits"], keep=False)
    unambiguous = data.loc[~repeated]
    same_dims = not (data[["d_q", "d_seq", "d_model", "num_head", "d_ff"]].nunique(dropna=False) > 1).any()
    if not same_dims:
        unambiguous = data.iloc[0:0]
    xs, ys = sorted(data.sve_bits.unique()), sorted(data.codebook_size.dropna().unique())
    metrics = [("sim_seconds", "Simulated time (s)"), ("l1d_miss_pct", "L1D demand miss rate (%)"), ("l2_miss_pct", "L2 demand miss rate (%)")]
    traces = []
    for i, (col, name) in enumerate(metrics):
        table = unambiguous.pivot(index="codebook_size", columns="sve_bits", values=col).reindex(index=ys, columns=xs)
        text = [["" if pd.isna(v) else "{:.2f}".format(v) for v in row] for row in table.values.tolist()]
        traces.append(dict(type="heatmap", x=[num(v) for v in xs], y=[num(v) for v in ys], z=table.values.tolist(),
                           text=text, texttemplate="%{text}", visible=i == 0, colorscale=BLUE,
                           zmin=0 if i else None, zmax=100 if i else None, zauto=i == 0,
                           colorbar=dict(title=dict(text=name), thickness=14), hoverongaps=False, xgap=3, ygap=3,
                           hovertemplate="SVE=%{x} bit · CB=%{y}<br>" + name + ": %{z:.4f}<extra></extra>"))
    ambiguous = data.loc[repeated] if same_dims else data
    cells = list(ambiguous.groupby(["codebook_size", "sve_bits"]))
    traces.append(dict(type="scatter", mode="markers+text", name="Multiple / incompatible observations",
                       x=[num(key[1]) for key, _ in cells], y=[num(key[0]) for key, _ in cells],
                       text=[str(len(group)) + " run(s)" for _, group in cells], textposition="middle center",
                       marker=dict(symbol="square-open", size=42, color="#72859a"), showlegend=False,
                       customdata=["<br>".join(html.escape(str(r.exp_id)) + ": {:.4f} s".format(r.sim_seconds) for r in group.itertuples()) for _, group in cells],
                       hovertemplate="SVE=%{x}, CB=%{y}<br>%{customdata}<extra>No automatic averaging</extra>"))
    buttons = [dict(label=name, method="update", args=[dict(visible=[i == j for i in range(len(metrics))] + [True])]) for j, (_, name) in enumerate(metrics)]
    lay = layout(model + " · codebook × SVE grid · 4 learners", 440, updatemenus=switch_menu(buttons))
    lay["xaxis"].update(title=dict(text="SVE width (bits)"), type="category")
    lay["yaxis"].update(title=dict(text="Codebook size"), type="category")
    if data.empty:
        lay["annotations"] = [dict(text="No four-learner codebook observations in this selection", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)]
    return dict(data=traces, layout=lay)


def miss_time_chart(total, model):
    traces = []
    for i, (impl, group) in enumerate(total.groupby("implementation")):
        traces.append(dict(type="scatter", mode="markers+text", name=impl,
                           x=(group.l2_demand_misses / 1e6).tolist(), y=group.sim_seconds.tolist(),
                           text=group.label.tolist(), textposition="top center", textfont=dict(size=10),
                           marker=dict(size=10, color=COLORS[i % len(COLORS)], symbol=["diamond-open" if e else "circle" for e in group.estimated]),
                           customdata=[[d, r, m] for d, r, m in zip(group.detail, group.l2_miss_pct, group.l2_mpki)],
                           hovertemplate="%{customdata[0]}<br>%{y:.4f} s<br>L2 misses=%{x:.3f} million<br>L2 miss rate=%{customdata[1]:.2f}%<br>L2 MPKI=%{customdata[2]:.2f}<extra></extra>"))
    lay = layout(model + " · execution time and L2 demand misses", 500)
    lay["xaxis"].update(title=dict(text="L2 demand misses (millions)"), rangemode="tozero")
    lay["yaxis"].update(title=dict(text="Simulated time (s)"), rangemode="tozero")
    return dict(data=traces, layout=lay)


def runtime_chart(total, model):
    traces = []
    for i, (implementation, group) in enumerate(total.groupby("implementation")):
        traces.append(dict(type="scatter", mode="markers", name=implementation,
                           x=group.label.tolist(), y=group.sim_seconds.tolist(),
                           marker=dict(size=10, color=COLORS[i % len(COLORS)],
                                       symbol=["diamond-open" if e else "circle" for e in group.estimated]),
                           customdata=group.detail.tolist(),
                           hovertemplate="%{customdata}<br>%{y:.6f} simulated seconds<extra></extra>"))
    lay = layout(model + " · experiment runtime", 430)
    lay["margin"]["b"] = 110
    lay["xaxis"].update(title=dict(text="Experiment"), type="category", categoryorder="array", categoryarray=total.label.tolist(), tickangle=-45)
    lay["yaxis"].update(title=dict(text="Simulated time (s)"), **log_ticks(total.sim_seconds))
    return dict(data=traces, layout=lay)


def load_optional(path, required, numbers, unique, positive_time=True):
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(str(path) + ": missing columns " + ", ".join(sorted(missing)))
    if frame.empty:
        return frame
    if frame[list(required)].isna().any().any():
        raise ValueError(str(path) + ": fill all required fields; do not use a header-only template as results.")
    for col in numbers:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not frame[col].map(math.isfinite).all() or (frame[col] < 0).any():
            raise ValueError(str(path) + ": invalid " + col)
    if frame.duplicated(unique).any():
        raise ValueError(str(path) + ": duplicate experiment/replicate key.")
    if positive_time and (frame.sim_seconds <= 0).any():
        raise ValueError(str(path) + ": simulated times must be positive.")
    for col in set(numbers) & {"tile_m", "tile_n", "tile_k", "num_cores"}:
        if (frame[col] <= 0).any() or (frame[col] % 1 != 0).any():
            raise ValueError(str(path) + ": " + col + " must contain positive integers.")
    return frame


def optional_numbers(frame, columns):
    present = set()
    for col in columns:
        if col not in frame or frame[col].isna().all():
            continue
        values = pd.to_numeric(frame[col], errors="raise")
        if values.isna().any() or not values.map(math.isfinite).all() or (values < 0).any():
            raise ValueError(col + ": supply finite, nonnegative measurements for every row, or leave the column empty.")
        frame[col] = values
        present.add(col)
    return present


def footprint_chart(group, config, tk, has_capacity):
    if (group.tile_footprint_bytes <= 0).any():
        raise ValueError("tile_footprint_bytes must be positive.")
    tiles = group.groupby(["tile_m", "tile_n"])
    if (tiles.tile_footprint_bytes.nunique() != 1).any():
        raise ValueError("Tile footprint differs between repetitions of the same tile.")
    median = tiles[["tile_footprint_bytes", "sim_seconds", "l2_miss_pct"]].median().reset_index()
    trace = dict(type="scatter", mode="markers", x=(median.tile_footprint_bytes / 1024).tolist(),
                 y=median.sim_seconds.tolist(), marker=dict(size=10, color=COLORS[0]),
                 customdata=median[["tile_m", "tile_n", "l2_miss_pct"]].values.tolist(),
                 hovertemplate="tile_m=%{customdata[0]}, tile_n=%{customdata[1]}<br>Footprint=%{x:.2f} KiB<br>Median time=%{y:.5f} s<br>L2 miss rate=%{customdata[2]:.2f}%<extra></extra>")
    lay = layout("Tile footprint · {} · tile_k={}".format(html.escape(str(config)), num(tk)), 470)
    lay["xaxis"].update(title=dict(text="Accounted tile footprint (KiB)"), rangemode="tozero")
    lay["yaxis"].update(title=dict(text="Median simulated time (s)"), rangemode="tozero")
    if has_capacity:
        if group.l2_capacity_bytes.nunique() != 1 or group.l2_capacity_bytes.iloc[0] <= 0:
            raise ValueError("L2 capacity must be positive and constant within a tile sweep.")
        capacity = group.l2_capacity_bytes.iloc[0] / 1024
        lay["shapes"] = [dict(type="line", x0=capacity, x1=capacity, y0=0, y1=1,
                              yref="paper", line=dict(dash="dash", color="#8b97a8"))]
        lay["annotations"] = [dict(x=capacity, y=1, yref="paper", text="L2 capacity", showarrow=False, yshift=12)]
    return dict(data=[trace], layout=lay)


def tile_charts(path):
    # config_id identifies a FIXED workload, layer, kernel, dtype, core count,
    # architecture, SVE width, cache setup, and input. tile_k is separately fixed.
    req = ["config_id", "tile_m", "tile_n", "tile_k", "replicate", "sim_seconds", "l1d_accesses", "l1d_misses", "l2_accesses", "l2_misses"]
    frame = load_optional(path, req, [c for c in req if c not in {"config_id", "replicate"}], ["config_id", "tile_m", "tile_n", "tile_k", "replicate"])
    extra = optional_numbers(frame, ["tile_footprint_bytes", "l2_capacity_bytes"])
    if "l2_capacity_bytes" in extra and ((frame.l2_capacity_bytes <= 0).any()
            or (frame.groupby("config_id").l2_capacity_bytes.nunique() != 1).any()):
        raise ValueError("L2 capacity must be positive and constant within a tile sweep.")
    for level in ["l1d", "l2"]:
        if (frame[level + "_misses"] > frame[level + "_accesses"]).any():
            raise ValueError(level + ": misses exceed accesses in tile results.")
        frame[level + "_miss_pct"] = 100 * frame[level + "_misses"] / frame[level + "_accesses"].where(frame[level + "_accesses"] > 0)
    output = []
    for (config, tk), group in frame.groupby(["config_id", "tile_k"]):
        aggregate = group.groupby(["tile_m", "tile_n"])[["sim_seconds", "l1d_miss_pct", "l2_miss_pct"]].median().reset_index()
        xs, ys = sorted(group.tile_n.unique()), sorted(group.tile_m.unique())
        counts = group.groupby(["tile_m", "tile_n"]).size().unstack().reindex(index=ys, columns=xs).values.tolist()
        traces = []
        for i, (col, name) in enumerate([("sim_seconds", "Median simulated time (s)"), ("l1d_miss_pct", "Median L1D demand miss rate (%)"), ("l2_miss_pct", "Median L2 demand miss rate (%)")]):
            table = aggregate.pivot(index="tile_m", columns="tile_n", values=col).reindex(index=ys, columns=xs)
            traces.append(dict(type="heatmap", x=[num(x) for x in xs], y=[num(y) for y in ys], z=table.values.tolist(),
                               customdata=counts, visible=i == 0, zmin=0 if i else None, zmax=100 if i else None, zauto=i == 0,
                               colorscale=BLUE, hoverongaps=False, xgap=2, ygap=2, colorbar=dict(title=dict(text=name)),
                               hovertemplate="tile_n=%{x}, tile_m=%{y}<br>" + name + "=%{z:.4f}<br>repeats=%{customdata}<extra></extra>"))
        buttons = [dict(label=n, method="update", args=[dict(visible=[i == j for i in range(3)])]) for j, n in enumerate(["Runtime", "L1D miss rate", "L2 miss rate"])]
        lay = layout("Tile sweep · {} · tile_k={}".format(html.escape(str(config)), num(tk)), 470, updatemenus=switch_menu(buttons))
        lay["xaxis"].update(title=dict(text="tile_n"), type="category")
        lay["yaxis"].update(title=dict(text="tile_m"), type="category")
        output.append(dict(data=traces, layout=lay))
        if "tile_footprint_bytes" in extra:
            output.append(footprint_chart(group, config, tk, "l2_capacity_bytes" in extra))
    return output


def scaling_charts(path):
    req = ["config_id", "num_cores", "replicate", "sim_seconds"]
    frame = load_optional(path, req, ["num_cores", "sim_seconds"], ["config_id", "num_cores", "replicate"])
    extra = optional_numbers(frame, ["dram_bytes"])
    output = []
    for config, group in frame.groupby("config_id"):
        med = group.groupby("num_cores").sim_seconds.agg(["median", "min", "max", "count"]).sort_index()
        if 1 not in med.index:
            raise ValueError(str(config) + ": scaling requires a measured 1-core baseline of the SAME algorithm/workload.")
        baseline = med.loc[1, "median"]
        cores = med.index.tolist()
        speed = (baseline / med["median"]).tolist()
        traces = [dict(type="scatter", mode="lines+markers", name="Measured median speedup", x=cores, y=speed,
                       customdata=[[t, n, e, lo, hi] for t, n, e, lo, hi in zip(med["median"], med["count"], [s / c for s, c in zip(speed, cores)], med["min"], med["max"])],
                       hovertemplate="%{x} cores<br>speedup=%{y:.3f}×<br>median time=%{customdata[0]:.4f} s<br>repeats=%{customdata[1]}<br>efficiency=%{customdata[2]:.1%}<br>time range=%{customdata[3]:.4f}–%{customdata[4]:.4f} s<extra></extra>", line=dict(color=COLORS[0], width=3)),
                  dict(type="scatter", mode="lines", x=cores, y=cores, name="Ideal linear", line=dict(dash="dash", color="#9aa9b6"))]
        lay = layout("Multicore scaling · " + html.escape(str(config)), 470)
        lay["xaxis"].update(title=dict(text="Active CPU cores"), tickvals=cores)
        lay["yaxis"].update(title=dict(text="Speedup = T(1) / T(p)"), rangemode="tozero")
        output.append(dict(data=traces, layout=lay))
        efficiency = [100 * s / p for s, p in zip(speed, cores)]
        lay = layout("Parallel efficiency · " + html.escape(str(config)), 430)
        lay["xaxis"].update(title=dict(text="Active CPU cores"), tickvals=cores)
        lay["yaxis"].update(title=dict(text="Efficiency = T(1) / (p × T(p)) (%)"), rangemode="tozero")
        output.append(dict(data=[
            dict(type="scatter", mode="lines+markers", x=cores, y=efficiency, name="Parallel efficiency",
                 line=dict(color=COLORS[0], width=3), hovertemplate="%{x} cores<br>Efficiency=%{y:.2f}%<extra></extra>"),
            dict(type="scatter", mode="lines", x=cores, y=[100] * len(cores), name="Linear scaling",
                 line=dict(dash="dash", color="#9aa9b6"))], layout=lay))
        if "dram_bytes" in extra:
            rates = (group.dram_bytes / group.sim_seconds / 1e9).groupby(group.num_cores).median().reindex(cores)
            lay = layout("DRAM bandwidth · " + html.escape(str(config)), 430)
            lay["xaxis"].update(title=dict(text="Active CPU cores"), tickvals=cores)
            lay["yaxis"].update(title=dict(text="Median measured DRAM bandwidth (GB/s)"), rangemode="tozero")
            output.append(dict(data=[dict(type="scatter", mode="lines+markers", x=cores, y=rates.tolist(),
                                          line=dict(color=COLORS[1], width=3),
                                          hovertemplate="%{x} cores<br>DRAM bandwidth=%{y:.4f} GB/s<extra></extra>")], layout=lay))
    return output


def implementation_phase_charts(path):
    keys = ["config_id", "variant", "replicate", "scope", "worker_id"]
    required = keys + ["phase", "sim_seconds", "roi_seconds"]
    frame = load_optional(path, required, ["sim_seconds", "roi_seconds"], keys + ["phase"], positive_time=False)
    if frame.empty:
        return []
    frame["worker_id"] = frame.worker_id.astype(str)
    if not frame.scope.isin(["wall", "worker"]).all():
        raise ValueError("Phase scope must be wall or worker.")
    if ((frame.scope == "wall") != (frame.worker_id == "all")).any():
        raise ValueError("Use worker_id=all only for wall phases; give worker phases their actual worker ID.")
    if not frame.phase.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Phase names must be nonempty strings.")
    if (frame.roi_seconds <= 0).any() or frame.phase.str.lower().isin(["unattributed", "__roi"]).any():
        raise ValueError("ROI time must be positive; Unattributed and __roi are reserved phase names.")
    grouped = frame.groupby(keys, sort=False)
    if (grouped.roi_seconds.nunique() != 1).any():
        raise ValueError("All phases in a run must use the same ROI time.")
    # Wall and worker rows are separate time bases; never sum concurrent workers.
    roi_per_scope = frame.groupby(["config_id", "variant", "replicate"]).roi_seconds.nunique()
    if (roi_per_scope != 1).any():
        raise ValueError("Workers must share the same ROI duration within a run.")
    for _, group in frame.groupby(["config_id", "scope"]):
        phase_sets = group.groupby(keys).phase.agg(lambda v: tuple(sorted(v)))
        if len(set(phase_sets)) != 1:
            raise ValueError("Phase sets must match within a configuration/scope; record measured zero durations explicitly.")
    for _, group in frame.groupby(["config_id", "variant", "scope"]):
        workers = group.groupby("replicate").worker_id.agg(lambda v: tuple(sorted(set(v))))
        if len(set(workers)) != 1:
            raise ValueError("Worker IDs must be complete and consistent across repetitions.")
    roi = grouped.roi_seconds.first()
    seconds = grouped.sim_seconds.sum()
    if (seconds > roi + 1e-9 + 1e-8 * roi).any():
        raise ValueError("Phase durations exceed ROI time; use exclusive phases and do not sum overlapping workers.")
    matrix = frame.pivot(index=keys, columns="phase", values="sim_seconds")
    matrix["Unattributed"] = (roi - seconds).clip(lower=0)
    matrix["__roi"] = roi
    output = []
    for (config, scope), group in frame.groupby(["config_id", "scope"], sort=False):
        phases = list(dict.fromkeys(group.phase))
        runs = matrix.xs((config, scope), level=("config_id", "scope"))
        means = runs.groupby(level=["variant", "worker_id"])[phases + ["Unattributed", "__roi"]].mean()
        repeats = runs.groupby(level=["variant", "worker_id"]).size()
        if means.Unattributed.max() > 1e-9:
            phases.append("Unattributed")
        labels = [str(v) if scope == "wall" else "{} / worker {}".format(v, w) for v, w in means.index]
        values = [means[phase].tolist() for phase in phases]
        percents = [(100 * means[phase] / means["__roi"]).tolist() for phase in phases]
        phase_colors = [COLORS[i] for i in [0, 1, 3, 4, 5]]
        traces = [dict(type="bar", name=html.escape(str(phase)), x=labels, y=values[i],
                       marker=dict(color="#cad2dc" if phase == "Unattributed" else phase_colors[i % len(phase_colors)]),
                       customdata=[[s, p, n] for s, p, n in zip(values[i], percents[i], repeats)],
                       hovertemplate="%{x}<br>" + html.escape(str(phase)) + ": %{customdata[0]:.5f} s<br>Share of mean ROI=%{customdata[1]:.2f}%<br>Repeats=%{customdata[2]}<extra></extra>")
                  for i, phase in enumerate(phases)]
        axis = "Mean wall time (s)" if scope == "wall" else "Mean worker time (s)"
        buttons = [dict(label="Mean time (s)", method="update", args=[dict(y=values), {"yaxis.title.text": axis}]),
                   dict(label="Share of ROI (%)", method="update", args=[dict(y=percents), {"yaxis.title.text": "Share of mean ROI (%)"}])]
        lay = layout("Implementation phases · {} · {}".format(html.escape(str(config)), scope), 520,
                     barmode="stack", updatemenus=switch_menu(buttons))
        lay["xaxis"].update(title=dict(text="Implementation" if scope == "wall" else "Implementation / worker"), type="category")
        lay["yaxis"].update(title=dict(text=axis), rangemode="tozero")
        output.append(dict(data=traces, layout=lay))
    return output


CSS = """
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#203247;font:14px/1.5 Arial,Helvetica,sans-serif}
main{max-width:1360px;margin:auto;padding:24px 28px 40px}h1{margin:0;font-size:23px;font-weight:600;letter-spacing:-.4px}
.report-header{padding:0 2px 16px}.toolbar{display:flex;gap:24px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:5;background:#f5f7faf5;padding:12px 0 16px}
.toolbar label{display:flex;align-items:center;gap:10px;font-weight:600}select,button{font:inherit;border:1px solid #cbd5e1;border-radius:5px;background:white;color:#203247;padding:7px 12px}button{cursor:pointer}.tabs{display:flex;gap:6px;flex-wrap:wrap}.tabs button.active{background:#234e75;border-color:#234e75;color:white}
.plot-card{background:white;border:1px solid #e1e7ef;border-radius:6px;margin:0 0 20px;padding:12px 8px}.graph{width:100%;min-height:420px}.hidden{display:none!important}
@media(max-width:700px){main{padding:18px 10px}.toolbar{position:static;gap:12px}.plot-card{padding:4px}.graph{min-width:600px}.chartwrap{overflow-x:auto}h1{font-size:21px}}
"""


def build_report(frame, totals, stages, bundle, source_name, tile_path=None, scaling_path=None, provenance=None, phase_path=None):
    specs, sections = {}, []

    def graph(figure, tab):
        ident = "chart_{}".format(len(specs))
        specs[ident] = figure
        return ('<section class="plot-card" data-tab="{}"><div class="chartwrap">'
                '<div class="graph" id="{}"></div></div></section>').format(tab, ident)

    models = sorted(totals.model.unique())
    for model in models:
        total, stage = totals[totals.model == model], stages[stages.model == model]
        sections.append('<div class="model-view" data-model="{}">'.format(html.escape(model, quote=True)))
        sections.append(graph(runtime_chart(total, model), "stages"))
        if total.phase_timing_valid.any():
            sections.append(graph(phase_chart(total, stage, model), "stages"))
        sections.append(graph(cache_chart(total, stage, model), "stages"))
        codebook = total.loc[total.implementation == "Codebook SIMD int8"]
        if (codebook.codebook_size == 8).any():
            sections.append(graph(sve_chart(total, model), "curves"))
        if (codebook.n_learners == 4).any():
            sections.append(graph(codebook_chart(total, model), "curves"))
        sections.append(graph(miss_time_chart(total, model), "curves"))
        sections.append('</div>')

    tabs = [("stages", "Overview"), ("curves", "Parameters")]
    for key, label, figures in [
        ("tiles", "Tile sweeps", tile_charts(tile_path) if tile_path else []),
        ("scaling", "Multicore", scaling_charts(scaling_path) if scaling_path else []),
        ("phases", "Implementation phases", implementation_phase_charts(phase_path) if phase_path else []),
    ]:
        if figures:
            tabs.append((key, label))
            sections.extend(graph(figure, key) for figure in figures)

    options = ''.join('<option value="{0}">{0}</option>'.format(html.escape(m, quote=True)) for m in models)
    buttons = ''.join('<button type="button" data-view="{}">{}</button>'.format(key, label) for key, label in tabs)
    payload = json.dumps(clean(specs), ensure_ascii=True, allow_nan=False).replace('<', '\\u003c')
    metadata = json.dumps(provenance or {}, ensure_ascii=True).replace('<', '\\u003c')
    model_default = "BERT-base" if "BERT-base" in models else models[0]
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Transformer Inference Performance</title><style>' + CSS + '</style><script>' + bundle +
            '</script></head><body><main><header class="report-header"><h1>Transformer Inference Performance</h1></header>'
            '<div class="toolbar"><label>Model <select id="model">' + options + '</select></label>'
            '<nav class="tabs" aria-label="Charts">' + buttons + '</nav></div>' + ''.join(sections) + '</main>'
            '<script type="application/json" id="report-provenance">' + metadata + '</script>'
            '<script type="application/json" id="chart-specs">' + payload + '</script><script>'
            "const specs=JSON.parse(document.getElementById('chart-specs').textContent);"
            "let activeTab='stages';const model=document.getElementById('model');model.value=" + json.dumps(model_default) + ';' + """
function refresh(){
 document.querySelectorAll('.model-view').forEach(e=>e.classList.toggle('hidden',e.dataset.model!==model.value));
 document.querySelectorAll('[data-tab]').forEach(e=>e.classList.toggle('hidden',e.dataset.tab!==activeTab));
 document.querySelectorAll('[data-view]').forEach(e=>{e.classList.toggle('active',e.dataset.view===activeTab);e.setAttribute('aria-pressed',e.dataset.view===activeTab);});
 model.disabled=['tiles','scaling','phases'].includes(activeTab);
 document.querySelectorAll('.graph').forEach(e=>{if(e.offsetParent!==null){if(!e.dataset.drawn){e.dataset.drawn='1';const s=specs[e.id];Plotly.newPlot(e,s.data,s.layout,{responsive:true,displaylogo:false,locale:'en',toImageButtonOptions:{format:'svg',filename:e.id}});}else{Plotly.Plots.resize(e);}}});
}
model.addEventListener('change',refresh);document.querySelectorAll('[data-view]').forEach(e=>e.addEventListener('click',()=>{activeTab=e.dataset.view;refresh();}));refresh();
</script></body></html>""")


def atomic_write(path, text):
    """Do not leave a truncated HTML/TSV if generation or writing fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(text)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def source_provenance(path):
    info = {"input": str(path),
            "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in input_files(path)],
            "pandas": pd.__version__, "python": sys.version.split()[0]}
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["report_repo_commit"] = result.stdout.strip()
            status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=5)
            info["report_repo_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


def optional_paths(args):
    folder = args.input if args.input.is_dir() else args.input.parent
    return {key: getattr(args, key) or (folder / name if (folder / name).is_file() else None)
            for key, name in OPTIONAL_RESULTS.items()}


def generate(args, get_plotlyjs):
    optional = optional_paths(args)
    inputs = {p.resolve() for p in input_files(args.input)} | {p.resolve() for p in optional.values() if p is not None}
    outputs = [p for p in [args.output, args.metrics_output] if p is not None]
    if any(p in inputs or (args.input.is_dir() and p.parent == args.input
                          and (re.fullmatch(r"e\d+(?:_.*)?\.tsv", p.name, re.IGNORECASE)
                               or p.name in {"final_all_experiments.tsv", *OPTIONAL_RESULTS.values()}))
           for p in outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("Output paths must be distinct and must not overwrite any input.")
    frame, totals, stages = load_data(args.input)
    if args.experiments:
        unknown = set(args.experiments) - set(totals.exp_id)
        if unknown:
            raise ValueError("Unknown experiment IDs: " + ", ".join(sorted(unknown)))
        frame = frame[frame.exp_id.isin(args.experiments)]
        totals = totals[totals.exp_id.isin(args.experiments)]
        stages = stages[stages.exp_id.isin(args.experiments)]
    provenance = source_provenance(args.input)
    provenance["selected_experiments"] = totals.exp_id.tolist()
    for key, path in optional.items():
        if path:
            provenance[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    import plotly
    provenance["plotly"] = plotly.__version__
    report = build_report(frame, totals, stages, get_plotlyjs(), args.input.name,
                          tile_path=optional["tile_results"], scaling_path=optional["scaling_results"],
                          phase_path=optional["phase_results"], provenance=provenance)
    metrics = frame.to_csv(sep="\t", index=False) if args.metrics_output else None
    atomic_write(args.output, report)
    if args.metrics_output:
        atomic_write(args.metrics_output, metrics)
    print("Created:", args.output.resolve(), flush=True)
    print("Experiments: {} | stage intervals: {}".format(len(totals), len(stages)), flush=True)


def watched_state(args):
    """Poll result files only, including optional CSVs added after watching starts."""
    folder = args.input if args.input.is_dir() else args.input.parent
    paths = set(input_files(args.input))
    paths.update(getattr(args, key) or folder / name for key, name in OPTIONAL_RESULTS.items())
    state = []
    for path in sorted(paths):
        try:
            stat = path.stat()
            state.append((str(path), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            state.append((str(path), None, None))
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Result directory or a single combined TSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Self-contained HTML destination")
    parser.add_argument("--metrics-output", type=Path, help="Optional TSV with recomputed per-stage ratios")
    parser.add_argument("--tile-results", type=Path)
    parser.add_argument("--scaling-results", type=Path)
    parser.add_argument("--phase-results", type=Path, help="Measured exclusive wall/worker phase durations")
    parser.add_argument("--experiments", nargs="+", help="Optional experiment IDs; by default include every collected experiment")
    parser.add_argument("--watch", action="store_true", help="Regenerate when local result files change; stop with Ctrl-C")
    parser.add_argument("--watch-interval", type=float, default=2.0, help="Polling interval in seconds")
    args = parser.parse_args(argv)
    for key in ["input", "output", "metrics_output", *OPTIONAL_RESULTS]:
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    if not math.isfinite(args.watch_interval) or args.watch_interval <= 0:
        parser.error("--watch-interval must be a positive finite number.")
    if sys.version_info < (3, 9):
        parser.error("Activate gem5_env: Python 3.9 or newer is required.")
    try:
        get_plotlyjs = load_dependencies()
    except ImportError as exc:
        parser.exit(1, "Missing visualization dependency: {}\nActivate gem5_env and run:\n  python -m pip install -r {}\n".format(exc, SCRIPT_DIR / "requirements-visualization.txt"))
    try:
        state = watched_state(args) if args.watch else None
        generate(args, get_plotlyjs)
        if args.watch:
            print("Watching:", args.input, "(Ctrl-C to stop)", flush=True)
            while True:
                time.sleep(args.watch_interval)
                try:
                    current = watched_state(args)
                    if current != state:
                        generate(args, get_plotlyjs)
                        state = current
                except (OSError, ValueError) as exc:
                    print("Refresh skipped; previous report retained:", exc, file=sys.stderr, flush=True)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, "Error: {}\n".format(exc))
    except KeyboardInterrupt:
        print("\nStopped watching.")


if __name__ == "__main__":
    main()
