"""
NexusGrid Gradio Dashboard — Pro Command Center

Professional SCADA-style UI: dark glass panels, live KPIs, spacious charts.
Seven panels: topology, frequency, Sankey, threat feed, action trace,
reward breakdown, task radar.
"""

from __future__ import annotations

import gradio as gr
import json
import math
from typing import Dict, Any, List

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]
    PLOTLY_AVAILABLE = False

# ── Pro command-center palette ────────────────────────────────────
C: Dict[str, str] = {
    "bg":          "#070b12",
    "bg_grad":     "#0c1220",
    "surface":     "#111827",
    "surface_hi":  "#1a2332",
    "glass":       "rgba(17,24,39,0.72)",
    "border":      "#243044",
    "border_mid":  "#334155",
    "border_hi":   "#3d4f66",
    "text":        "#f1f5f9",
    "text_mid":    "#94a3b8",
    "text_soft":   "#64748b",
    "accent":      "#22d3ee",
    "accent_soft": "rgba(34,211,238,0.12)",
    "indigo":      "#818cf8",
    "blue":        "#3b82f6",
    "purple":      "#a78bfa",
    "green":       "#34d399",
    "green_soft":  "rgba(52,211,153,0.15)",
    "amber":       "#fbbf24",
    "amber_soft":  "rgba(251,191,36,0.12)",
    "orange":      "#fb923c",
    "red":         "#f87171",
    "red_soft":    "rgba(248,113,113,0.12)",
    "crimson":     "#fb7185",
}

FS = {
    "title":  15,
    "axis":   12,
    "label":  11,
    "legend": 11,
    "value":  36,
    "zone":   14,
    "hover":  12,
}

TYPE_COL = {
    "hydro": "#0284c7", "nuclear": "#16a34a", "gas": "#ea580c",
    "wind": "#0891b2", "solar": "#ca8a04", "battery": "#9333ea", "load": "#64748b",
}
TYPE_RING = {
    "hydro": 0, "nuclear": 0, "gas": 0,
    "wind": 1, "solar": 1, "battery": 1, "load": 2,
}
RING_R = {0: 0.30, 1: 0.62, 2: 0.92}

TASK_META = [
    {"id": 0, "name": "Smoke Test",        "difficulty": "Trivial",    "max": 1.0},
    {"id": 1, "name": "Duck Curve",         "difficulty": "Easy",       "max": 1.0},
    {"id": 2, "name": "Cascade Overload",   "difficulty": "Medium",     "max": 1.0},
    {"id": 3, "name": "Phantom Injection",  "difficulty": "Hard",       "max": 1.0},
    {"id": 4, "name": "Stuxnet Resonance",  "difficulty": "Very Hard",  "max": 1.0},
    {"id": 5, "name": "Black Start",        "difficulty": "Expert",     "max": 1.0},
]

DIFF_COL = {
    "Trivial": "#94a3b8", "Easy": "#34d399", "Medium": "#22d3ee",
    "Hard": "#fbbf24", "Very Hard": "#fb923c", "Expert": "#fb7185",
}

PLOT_HEIGHT = 440
PLOT_HEIGHT_TALL = 500
PLOT_HEIGHT_WIDE = 540


def _pro_layout(**kw) -> dict:
    """Plotly layout defaults; merges extra font keys without duplicate kwargs."""
    font_extra = kw.pop("font", None)
    base_font = dict(family="DM Sans, Inter, system-ui, sans-serif", color=C["text"], size=FS["axis"])
    if font_extra:
        base_font.update(font_extra)
    base = dict(
        template="plotly_dark",
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface_hi"],
        font=base_font,
        margin=dict(l=52, r=52, t=72, b=52),
        autosize=True,
        hoverlabel=dict(
            bgcolor=C["surface_hi"],
            bordercolor=C["border_hi"],
            font=dict(family="JetBrains Mono, ui-monospace, monospace", size=FS["hover"], color=C["text"]),
        ),
    )
    base.update(kw)
    return base


def _apply_layout(fig: go.Figure, **kw) -> go.Figure:
    """Apply layout without duplicate keyword errors (e.g. font passed twice)."""
    fig.update_layout(**_pro_layout(**kw))
    return fig


def _safe_chart(builder, *args, **kwargs):
    try:
        return builder(*args, **kwargs)
    except Exception as exc:
        fig = go.Figure()
        fig.update_layout(**_pro_layout(height=360))
        fig.add_annotation(
            text=f"Chart unavailable: {exc}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=C["text_mid"]),
        )
        return fig


def _shell(title: str, dot: str = C["accent"]) -> str:
    return (
        f'<div class="chart-shell">'
        f'<div class="chart-head"><span>{title}</span>'
        f'<span class="dot" style="background:{dot}"></span></div></div>'
    )


def _section(num: str, title: str, subtitle: str) -> str:
    return (
        f'<div class="ops-section">'
        f'<span class="ops-num">{num}</span>'
        f'<span class="ops-title">{title}</span>'
        f'<span class="ops-sub">{subtitle}</span>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════
# PANEL 1 — GRID TOPOLOGY
# ═══════════════════════════════════════════════════════════════════
def create_topology_graph(obs_dict: Dict[str, Any]) -> go.Figure:
    topo = obs_dict.get("topology_graph", {})
    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.update_layout(
            **_pro_layout(height=PLOT_HEIGHT_TALL, margin=dict(l=40, r=40, t=56, b=40)),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        fig.add_annotation(
            text="System offline — reset to initialise the grid",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=C["text_soft"]),
        )
        return fig

    buckets: Dict[int, list] = {0: [], 1: [], 2: []}
    for nd in nodes:
        ring = TYPE_RING.get(nd.get("node_type", "load"), 2)
        buckets[ring].append(nd)

    pos: Dict[str, tuple] = {}
    for ring_id, bucket in buckets.items():
        n = len(bucket)
        r = RING_R[ring_id]
        angle_offset = 15 * ring_id
        for i, nd in enumerate(bucket):
            deg = angle_offset + (360 * i / max(n, 1))
            rad = math.radians(deg)
            pos[nd["id"]] = (r * math.cos(rad), r * math.sin(rad))

    ring_traces = []
    for r_val in (0.30, 0.62, 0.92):
        angles = list(range(0, 361, 4))
        xs = [r_val * math.cos(math.radians(a)) for a in angles]
        ys = [r_val * math.sin(math.radians(a)) for a in angles]
        ring_traces.append(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=C["border"], width=1, dash="dot"),
            showlegend=False, hoverinfo="none",
        ))

    edge_traces: List[go.Scatter] = []
    for edge in edges:
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if src not in pos or tgt not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        cap = max(edge.get("capacity_mw", 1), 1)
        load = edge.get("current_load_mw", 0)
        status = edge.get("status", "LIVE")
        pct = load / cap

        if status == "TRIPPED":
            col, w, dash = C["text_soft"], 1.5, "dot"
        elif pct >= 0.95:
            col, w, dash = C["crimson"], 3.5, "solid"
        elif pct >= 0.80:
            col, w, dash = C["amber"], 2.8, "solid"
        elif pct >= 0.50:
            col, w, dash = C["accent"], 2.0, "solid"
        else:
            col, w, dash = "#94a3b8", 1.2, "solid"

        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dist = math.sqrt(mx**2 + my**2)
        if dist > 0.01:
            bulge = 0.10
            cx = mx - bulge * mx / dist
            cy = my - bulge * my / dist
        else:
            cx, cy = mx, my

        curve_x, curve_y = [], []
        for k in range(9):
            t = k / 8
            curve_x.append((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1)
            curve_y.append((1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1)
        curve_x.append(None)
        curve_y.append(None)

        edge_traces.append(go.Scatter(
            x=curve_x, y=curve_y, mode="lines",
            line=dict(width=w, color=col, dash=dash),
            hoverinfo="none", showlegend=False,
        ))

    nx_l, ny_l, nc_l, nb_l, ns_l = [], [], [], [], []
    hover_l = []

    for nd in nodes:
        nid = nd["id"]
        t = nd.get("node_type", "load")
        crit = nd.get("critical", False)
        en = nd.get("energized", True)
        cap = nd.get("capacity_mw", 0)
        gen = nd.get("generation_mw", 0)
        spoof = nd.get("spoofed", False)
        x, y = pos.get(nid, (0, 0))

        nx_l.append(x)
        ny_l.append(y)
        base = TYPE_COL.get(t, C["text_soft"])
        nc_l.append(base if en else "#334155")
        nb_l.append(C["crimson"] if crit else (C["amber"] if spoof else "#e2e8f0"))
        ns_l.append(20 + min(int(cap / 300), 12))

        tags = []
        if crit:
            tags.append("CRITICAL")
        if spoof:
            tags.append("SPOOFED")
        tag_str = (" · " + " · ".join(tags)) if tags else ""
        hover_l.append(
            f"<b>{nid}</b>{tag_str}<br>"
            f"Type: {t.upper()}<br>"
            f"State: {'Live' if en else 'Dark'}<br>"
            f"Capacity: {cap:,.0f} MW<br>"
            f"Generation: {gen:,.0f} MW"
        )

    node_trace = go.Scatter(
        x=nx_l, y=ny_l, mode="markers",
        hovertext=hover_l,
        hoverinfo="text",
        marker=dict(size=ns_l, color=nc_l, line=dict(width=2, color=nb_l), opacity=0.96),
        showlegend=False,
    )

    seen: set = set()
    legend: List[go.Scatter] = []
    for nd in nodes:
        t = nd.get("node_type", "load")
        if t in seen:
            continue
        seen.add(t)
        legend.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=TYPE_COL.get(t, C["text_soft"]),
                        line=dict(width=1, color=C["text"])),
            name=t.capitalize(),
            showlegend=True,
        ))

    fig = go.Figure(data=ring_traces + edge_traces + [node_trace] + legend)
    layout = _pro_layout(height=PLOT_HEIGHT_TALL, margin=dict(l=24, r=24, t=72, b=88))
    layout["title"] = dict(
        text="Grid Topology — 20-node network",
        font=dict(size=FS["title"], color=C["text"]),
        x=0.5, xanchor="center",
    )
    layout["legend"] = dict(
        orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
        font=dict(size=FS["legend"], color=C["text_mid"]),
        bgcolor="rgba(17,24,39,0.92)", bordercolor=C["border_hi"], borderwidth=1,
    )
    layout["xaxis"] = dict(visible=False, range=[-1.25, 1.25], scaleanchor="y", scaleratio=1)
    layout["yaxis"] = dict(visible=False, range=[-1.25, 1.25])
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════
# PANEL 2 — FREQUENCY GAUGE
# ═══════════════════════════════════════════════════════════════════
def create_frequency_gauge(freq_hz: float,
                           history: List[float] | None = None) -> go.Figure:
    freq_hz = round(freq_hz, 3)

    if freq_hz < 59.0:
        col, zone = C["crimson"], "BLACKOUT"
    elif freq_hz < 59.2:
        col, zone = C["red"], "CRITICAL"
    elif freq_hz < 59.5:
        col, zone = C["orange"], "DANGER"
    elif freq_hz < 59.7:
        col, zone = C["amber"], "WARNING"
    elif freq_hz <= 60.3:
        col, zone = C["green"], "NOMINAL"
    else:
        col, zone = C["amber"], "ELEVATED"

    delta = round(freq_hz - 60.0, 3)
    delta_sign = "+" if delta >= 0 else ""

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.14,
        specs=[[{"type": "indicator"}], [{"type": "scatter"}]],
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=freq_hz,
        number=dict(
            suffix=" Hz",
            font=dict(size=FS["value"], color=col, family="Inter, sans-serif"),
            valueformat=".3f",
        ),
        title=dict(
            text=f"Grid Frequency · {zone}",
            font=dict(size=FS["zone"], color=col),
        ),
        gauge=dict(
            axis=dict(
                range=[58.5, 61.5],
                tickvals=[59.0, 59.5, 60.0, 60.5, 61.0],
                tickwidth=1,
                tickcolor=C["text_mid"],
                tickfont=dict(size=11, color=C["text_mid"]),
            ),
            bar=dict(color=col, thickness=0.30),
            bgcolor="#1e293b",
            borderwidth=1,
            bordercolor=C["border_hi"],
            steps=[
                dict(range=[58.5, 59.0], color="rgba(251,113,133,0.55)"),
                dict(range=[59.0, 59.2], color="rgba(248,113,113,0.50)"),
                dict(range=[59.2, 59.5], color="rgba(251,146,60,0.45)"),
                dict(range=[59.5, 59.7], color="rgba(251,191,36,0.40)"),
                dict(range=[59.7, 60.3], color="rgba(52,211,153,0.48)"),
                dict(range=[60.3, 61.5], color="rgba(251,191,36,0.40)"),
            ],
            threshold=dict(line=dict(color=C["text"], width=2), thickness=0.8, value=freq_hz),
        ),
    ), row=1, col=1)

    hist = list(history) if history else [60.0]
    xs = list(range(len(hist)))
    fig.add_trace(go.Scatter(
        x=xs, y=hist, mode="lines",
        line=dict(color=col, width=2.5),
        fill="tozeroy",
        fillcolor=C["accent_soft"] if zone == "NOMINAL" else C["amber_soft"],
        name="History",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0, max(len(hist) - 1, 1)], y=[60.0, 60.0], mode="lines",
        line=dict(color=C["text_soft"], width=1.5, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    fig.update_xaxes(title_text="Simulation tick", title_font=dict(size=11), row=2, col=1)
    fig.update_yaxes(
        title_text="Hz",
        range=[58.8, 61.2],
        gridcolor=C["border"],
        tickfont=dict(size=11),
        row=2, col=1,
    )
    fig.update_layout(
        **_pro_layout(height=PLOT_HEIGHT, margin=dict(l=40, r=40, t=48, b=40)),
        annotations=[dict(
            text=f"Δ {delta_sign}{delta:.3f} Hz from nominal",
            xref="paper", yref="paper", x=0.5, y=1.02,
            showarrow=False, font=dict(size=12, color=C["text_mid"]),
        )],
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# PANEL 3 — POWER FLOW SANKEY
# ═══════════════════════════════════════════════════════════════════
def _type_stats(nodes: List[Dict]) -> tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    """Installed gen capacity, peak load, and energized counts per asset class."""
    installed: Dict[str, float] = {}
    peak_load: Dict[str, float] = {}
    energized_n: Dict[str, int] = {}
    for nd in nodes:
        t = nd.get("node_type", "load")
        if t == "load":
            peak_load[t] = peak_load.get(t, 0.0) + float(nd.get("peak_load_mw", 0) or 0)
        else:
            installed[t] = installed.get(t, 0.0) + float(nd.get("capacity_mw", 0) or 0)
        if nd.get("energized"):
            energized_n[t] = energized_n.get(t, 0) + 1
    return installed, peak_load, energized_n


def _aggregate_flows(
    edges: List[Dict],
    id_to_type: Dict[str, str],
    *,
    use_capacity: bool,
) -> tuple[Dict[tuple, float], Dict[tuple, list]]:
    """Sum MW by (source_type, target_type); live flow or line rating when de-energized."""
    flow_map: Dict[tuple, float] = {}
    ratio_map: Dict[tuple, list] = {}
    for edge in edges:
        s, t = edge.get("source", ""), edge.get("target", "")
        if s not in id_to_type or t not in id_to_type:
            continue
        line_cap = max(float(edge.get("capacity_mw", 0) or 0), 1.0)
        if use_capacity:
            val = line_cap
        else:
            val = max(float(edge.get("current_load_mw", 0) or 0), 0.0)
        if val <= 0:
            continue
        st, tt = id_to_type[s], id_to_type[t]
        key = (st, tt)
        flow_map[key] = flow_map.get(key, 0.0) + val
        if not use_capacity:
            ratio_map.setdefault(key, []).append(val / line_cap)
    return flow_map, ratio_map


def _sankey_link_color(pct: float, *, capacity_mode: bool) -> str:
    if capacity_mode:
        return "rgba(100,116,139,0.45)"
    if pct >= 0.95:
        return "rgba(251,113,133,0.55)"
    if pct >= 0.80:
        return "rgba(251,191,36,0.48)"
    if pct >= 0.50:
        return "rgba(34,211,238,0.42)"
    return "rgba(148,163,184,0.28)"


def _create_capacity_bar_chart(
    installed: Dict[str, float],
    peak_load: Dict[str, float],
    energized_n: Dict[str, int],
    *,
    title: str,
    subtitle: str,
) -> go.Figure:
    """Horizontal bar chart for de-energized grid (Sankey cannot render cyclic graphs)."""
    order = ["hydro", "nuclear", "gas", "wind", "solar", "battery", "load"]
    cats: List[str] = []
    vals: List[float] = []
    cols: List[str] = []
    for t in order:
        if t == "load":
            v = peak_load.get("load", 0.0)
            if v <= 0:
                continue
            cats.append("Load · peak demand")
            vals.append(v)
            cols.append(TYPE_COL["load"])
        else:
            v = installed.get(t, 0.0)
            if v <= 0:
                continue
            en = energized_n.get(t, 0)
            cats.append(f"{t.replace('_', ' ').title()} ({en} live)")
            vals.append(v)
            cols.append(TYPE_COL.get(t, C["text_soft"]))

    fig = go.Figure(go.Bar(
        x=vals,
        y=cats,
        orientation="h",
        marker=dict(color=cols, line=dict(width=1, color=C["border_hi"])),
        text=[f"{v:,.0f} MW" for v in vals],
        textposition="outside",
        textfont=dict(size=12, color=C["text"]),
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} MW<extra></extra>",
    ))
    layout = _pro_layout(
        height=PLOT_HEIGHT_WIDE,
        margin=dict(l=140, r=80, t=88, b=48),
        font=dict(size=12),
    )
    layout["title"] = dict(
        text=title,
        font=dict(size=FS["title"], color=C["accent"]),
        x=0.5,
        xanchor="center",
    )
    layout["annotations"] = [dict(
        text=subtitle,
        xref="paper", yref="paper", x=0.5, y=1.02,
        showarrow=False,
        font=dict(size=11, color=C["text_mid"]),
    )]
    layout["xaxis"] = dict(
        title="MW",
        gridcolor=C["border"],
        zeroline=False,
        tickfont=dict(size=11),
    )
    layout["yaxis"] = dict(automargin=True, tickfont=dict(size=12))
    layout["bargap"] = 0.28
    fig.update_layout(**layout)
    return fig


def _hub_sankey_from_flow_map(
    flow_map: Dict[tuple, float],
    ratio_map: Dict[tuple, list],
    types: List[str],
    *,
    capacity_mode: bool,
) -> tuple[List[int], List[int], List[float], List[str], List[str], List[float], List[float]]:
    """Acyclic gen -> backbone -> load Sankey (Plotly cannot render cycles)."""
    gen = [t for t in types if t != "load"]
    labels = [t.replace("_", " ").title() for t in gen] + ["Grid Backbone", "Load"]
    idx = {t: i for i, t in enumerate(gen)}
    hub_i = len(gen)
    load_i = hub_i + 1

    x_pos = [0.07] * len(gen) + [0.50, 0.93]
    if len(gen) == 1:
        y_pos = [0.5, 0.5, 0.5]
    else:
        y_pos = [0.12 + 0.76 * i / (len(gen) - 1) for i in range(len(gen))] + [0.5, 0.5]

    hub_color = C["indigo"]
    ncolors = [TYPE_COL.get(t, C["text_soft"]) for t in gen] + [hub_color, TYPE_COL["load"]]

    src, tgt, val, col = [], [], [], []
    for (st, tt), v in flow_map.items():
        if st == tt or v <= 0:
            continue
        pct = max(ratio_map.get((st, tt), [0.0])) if not capacity_mode else 0.0
        link_col = _sankey_link_color(pct, capacity_mode=capacity_mode)
        if st != "load" and tt == "load" and st in idx:
            src.append(idx[st])
            tgt.append(load_i)
        elif st != "load" and tt != "load" and st in idx:
            src.append(idx[st])
            tgt.append(hub_i)
        elif st == "load" and tt != "load" and tt in idx:
            src.append(hub_i)
            tgt.append(idx[tt])
        else:
            continue
        val.append(v)
        col.append(link_col)

    return src, tgt, val, col, labels, ncolors, x_pos, y_pos


def _layout_power_panel(fig: go.Figure, *, title: str, subtitle: str) -> go.Figure:
    layout = _pro_layout(
        height=PLOT_HEIGHT_WIDE,
        margin=dict(l=40, r=40, t=96, b=40),
        font=dict(size=12),
    )
    layout["title"] = dict(
        text=title,
        font=dict(size=FS["title"], color=C["accent"]),
        x=0.5,
        xanchor="center",
    )
    layout["annotations"] = [dict(
        text=subtitle,
        xref="paper", yref="paper", x=0.5, y=1.02,
        showarrow=False,
        font=dict(size=11, color=C["text_mid"]),
    )]
    fig.update_layout(**layout)
    return fig


def create_power_flow_sankey(obs_dict: Dict[str, Any]) -> go.Figure:
    """Power flow Sankey (live MW) or capacity bar chart when grid is dark (Task 5)."""
    topo = obs_dict.get("topology_graph", {})
    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.update_layout(**_pro_layout(height=PLOT_HEIGHT_WIDE))
        fig.add_annotation(
            text="Awaiting grid data",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=C["text_soft"]),
        )
        return fig

    id_to_type: Dict[str, str] = {nd["id"]: nd.get("node_type", "load") for nd in nodes}
    live_map, live_ratio = _aggregate_flows(edges, id_to_type, use_capacity=False)
    live_total = sum(live_map.values())

    n_live = sum(1 for e in edges if e.get("status") == "LIVE")
    n_tripped = sum(1 for e in edges if e.get("status") == "TRIPPED")
    task_id = int(obs_dict.get("task_id", obs_dict.get("metadata", {}).get("task_id", -1)))

    capacity_mode = live_total < 1.0
    flow_map, ratio_map = (
        _aggregate_flows(edges, id_to_type, use_capacity=True)
        if capacity_mode else (live_map, live_ratio)
    )

    installed, peak_load, energized_n = _type_stats(nodes)
    types = sorted({id_to_type[n["id"]] for n in nodes})

    if capacity_mode:
        title = "Grid Assets & Peak Demand (MW)"
        if task_id == 5:
            subtitle = (
                f"Black start - grid de-energized - {n_tripped} lines TRIPPED - "
                "dispatch NODE_01 hydro to restore flow"
            )
        else:
            subtitle = f"No live MW flow ({n_tripped} tripped / {len(edges)} lines)"
        return _create_capacity_bar_chart(
            installed, peak_load, energized_n, title=title, subtitle=subtitle,
        )

    sankey_src, sankey_tgt, sankey_val, sankey_col, nlabels, ncolors, x_pos, y_pos = (
        _hub_sankey_from_flow_map(flow_map, ratio_map, types, capacity_mode=False)
    )

    if not sankey_val:
        fig = go.Figure()
        _layout_power_panel(fig, title="Power Flow by Asset Class", subtitle="No live MW on transmission lines")
        fig.add_annotation(
            text="No live power flow",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=C["text_soft"]),
        )
        return fig

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        valueformat=".0f",
        valuesuffix=" MW",
        node=dict(
            pad=28,
            thickness=20,
            line=dict(color=C["border_hi"], width=1),
            label=nlabels,
            color=ncolors,
            x=x_pos,
            y=y_pos,
        ),
        link=dict(
            source=sankey_src,
            target=sankey_tgt,
            value=sankey_val,
            color=sankey_col,
        ),
    ))
    title = "Power Flow by Asset Class (MW)"
    subtitle = f"{n_live} LIVE lines - {live_total:,.0f} MW total flow"
    return _layout_power_panel(fig, title=title, subtitle=subtitle)


# ═══════════════════════════════════════════════════════════════════
# PANEL 4 — TASK PERFORMANCE RADAR
# ═══════════════════════════════════════════════════════════════════
def create_task_radar(scores: Dict[int, float]) -> go.Figure:
    """Short axis labels (T0–T5) + hover for full mission names — avoids label overlap."""
    theta = [f"T{t['id']}" for t in TASK_META]
    values = [float(scores.get(t["id"], 0.0)) for t in TASK_META]
    labels = [f"{t['name']} · {t['difficulty']}" for t in TASK_META]
    theta_c = theta + [theta[0]]
    values_c = values + [values[0]]
    labels_c = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_c,
        theta=theta_c,
        fill="toself",
        fillcolor="rgba(167,139,250,0.18)",
        line=dict(color=C["purple"], width=3),
        marker=dict(size=10, color=C["purple"], line=dict(width=2, color="#f8fafc")),
        mode="lines+markers",
        name="Mission score",
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.5] * len(theta_c),
        theta=theta_c,
        line=dict(color=C["amber"], width=1.5, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
        mode="lines",
    ))

    layout = _pro_layout(
        height=PLOT_HEIGHT,
        margin=dict(l=100, r=100, t=64, b=100),
    )
    layout["title"] = dict(
        text="Mission Performance Radar",
        font=dict(size=FS["title"], color=C["purple"]),
        x=0.5,
        xanchor="center",
    )
    layout["showlegend"] = False
    layout["polar"] = dict(
        bgcolor=C["surface_hi"],
        radialaxis=dict(
            visible=True,
            range=[0, 1.08],
            tickvals=[0.25, 0.5, 0.75, 1.0],
            ticktext=["0.25", "0.50", "0.75", "1.00"],
            tickfont=dict(size=11, color=C["text_mid"], family="JetBrains Mono, monospace"),
            gridcolor="rgba(167,139,250,0.15)",
            linecolor=C["border_hi"],
        ),
        angularaxis=dict(
            tickfont=dict(size=13, color=C["text"], family="JetBrains Mono, monospace"),
            rotation=30,
            direction="clockwise",
            linecolor=C["border_hi"],
            gridcolor="rgba(167,139,250,0.12)",
        ),
    )
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════
# PANEL 5 — REWARD BREAKDOWN
# ═══════════════════════════════════════════════════════════════════
def create_reward_breakdown(reward_history: List[Dict[str, float]]) -> go.Figure:
    POS = [
        "fault_isolation", "cyber_detection", "frequency_stable",
        "proactive_dispatch", "reasoning_order", "stability_bonus",
    ]
    NEG = ["overload_penalty", "hallucination_penalty", "redundant_estimation"]

    acc: Dict[str, float] = {}
    for entry in reward_history:
        for k, v in entry.items():
            acc[k] = acc.get(k, 0.0) + v

    labels, values = [], []
    for sig in POS:
        labels.append(sig.replace("_", " ").title())
        values.append(acc.get(sig, 0.0))
    for sig in NEG:
        labels.append(sig.replace("_", " ").title())
        values.append(acc.get(sig, 0.0))

    if not any(values):
        labels = ["Awaiting signal data"]
        values = [0.0]

    pos_vals = [max(v, 0) for v in values]
    neg_vals = [min(v, 0) for v in values]
    max_abs = max(abs(v) for v in values) if values else 0.5
    pad = max_abs * 0.35 + 0.05

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pos_vals, y=labels, orientation="h",
        marker=dict(color=C["green"], opacity=0.85),
        text=[f"+{v:.3f}" if v > 0 else "" for v in values],
        textposition="outside",
        textfont=dict(color=C["green"], size=11),
        cliponaxis=False,
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>+%{x:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=neg_vals, y=labels, orientation="h",
        marker=dict(color=C["red"], opacity=0.85),
        text=[f"{v:.3f}" if v < 0 else "" for v in values],
        textposition="outside",
        textfont=dict(color=C["red"], size=11),
        cliponaxis=False,
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color=C["text_mid"], line_width=1)
    layout = _pro_layout(height=PLOT_HEIGHT, margin=dict(l=24, r=110, t=64, b=40))
    layout["barmode"] = "overlay"
    layout["title"] = dict(
        text="Reward signals — gains vs penalties",
        font=dict(size=FS["title"], color=C["text"]),
        x=0.5, xanchor="center",
    )
    layout["xaxis"] = dict(
        title="Accumulated reward",
        range=[-max_abs - pad, max_abs + pad],
        gridcolor=C["border"],
        zeroline=True,
        zerolinecolor=C["text_mid"],
        tickfont=dict(size=11),
    )
    layout["yaxis"] = dict(autorange="reversed", tickfont=dict(size=11), automargin=True)
    layout["bargap"] = 0.35
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════
# TEXT FORMATTERS
# ═══════════════════════════════════════════════════════════════════
def format_threat_feed(packets: List[Dict], obs: Dict) -> str:
    hdr = "SCADA THREAT MONITOR · Kirchhoff Oracle\n" + ("─" * 52) + "\n\n"
    if not packets:
        return hdr + "  System offline — reset to begin monitoring.\n"

    lines = [
        hdr,
        f"  {'TIME':>5}  {'STATUS':<10}  {'ROUTE':<18}  {'LAT':>5}\n",
        f"  {'─' * 5}  {'─' * 10}  {'─' * 18}  {'─' * 5}\n",
    ]
    for pkt in packets[-16:]:
        src = str(pkt.get("source_node", "?"))[:8]
        dst = str(pkt.get("dest_node", "?"))[:8]
        lat = pkt.get("latency_ms", 0)
        anomaly = pkt.get("anomaly_flag", False)
        ts = pkt.get("timestamp", 0)
        route = f"{src} -> {dst}"
        if anomaly:
            tag = "ANOMALY"
        elif lat > 50:
            tag = "ELEVATED"
        elif lat > 30:
            tag = "CAUTION"
        else:
            tag = "CLEAR"
        lines.append(f"  {ts:>5.1f}  {tag:<10}  {route:<18}  {lat:>4.0f}ms\n")

    est = obs.get("last_state_estimation")
    if est:
        lines.append(f"\n  Kirchhoff Oracle\n  {'─' * 36}\n")
        if not est.get("consistent", True):
            vn = est.get("violation_node", "?")
            est_mw = est.get("estimated_true_mw", 0)
            lines.append(f"  VIOLATION at {vn} · estimate {est_mw:.1f} MW\n")
        else:
            lines.append("  All nodes Kirchhoff-consistent\n")

    spoofs = obs.get("active_spoofs", [])
    if spoofs:
        lines.append(f"\n  Active spoofs: {', '.join(spoofs)}\n")
    return "".join(lines)


def format_action_trace(actions: List[Dict]) -> str:
    hdr = "AGENT ACTION TRACE · Decision log\n" + ("─" * 48) + "\n\n"
    if not actions:
        return hdr + "  No actions — reset and step the environment.\n"

    lines = [
        hdr,
        f"  {'TK':>3}  {'ACTION':<24}  {'RWD':>8}  {'SCORE':>7}\n",
        f"  {'─' * 3}  {'─' * 24}  {'─' * 8}  {'─' * 7}\n",
    ]
    for a in actions[-20:]:
        tick = a.get("tick", 0)
        act = a.get("action", "?")
        reward = a.get("reward", "0.000")
        score = a.get("score", "0.000")
        params = a.get("params", "{}")
        lines.append(
            f"  {tick:>3}  {act[:24]:<24}  {str(reward):>8}  {str(score):>7}\n"
        )
        if params and params not in ("{}", "null", ""):
            try:
                pd = json.loads(params)
                short = "  ".join(f"{k}={v}" for k, v in pd.items())[:50]
            except Exception:
                short = str(params)[:50]
            lines.append(f"       {short}\n")
    return "".join(lines)


def format_status_bar(state) -> str:
    done_str = "COMPLETE" if state.done else "RUNNING"
    return (
        f"TASK {state.task_id}  |  TICK {state.tick:03d}  |  "
        f"FREQ {state.frequency_hz:.3f} Hz  |  "
        f"SCORE {state.cumulative_score:+.4f}  |  {done_str}"
    )


def format_kpi_html(obs: Dict[str, Any], state) -> str:
    topo = obs.get("topology_graph", {}) if obs else {}
    node_list = topo.get("nodes", [])
    edge_list = topo.get("edges", [])
    n_nodes = len(node_list)
    n_edges = len(edge_list)
    n_energized = sum(1 for n in node_list if n.get("energized"))
    n_live_lines = sum(1 for e in edge_list if e.get("status") == "LIVE")
    packets = len(state.packet_log) if state.packet_log else 0
    actions = len(state.action_log) if state.action_log else 0
    freq = state.frequency_hz
    if freq < 59.5:
        freq_col = C["crimson"]
    elif freq < 59.7 or freq > 60.3:
        freq_col = C["amber"]
    else:
        freq_col = C["green"]
    score_col = C["green"] if state.cumulative_score >= 0 else C["red"]
    spoofs = len(obs.get("active_spoofs", [])) if obs else 0
    lines_label = "Live Lines" if n_live_lines else "Lines (TRIPPED)"
    lines_val = f"{n_live_lines}/{n_edges}" if n_edges else "—"
    nodes_val = f"{n_energized}/{n_nodes}" if n_nodes else "—"
    return f"""
    <div class="kpi-grid">
      <div class="kpi-pro"><span class="kpi-k">Energized Nodes</span><span class="kpi-v">{nodes_val}</span></div>
      <div class="kpi-pro"><span class="kpi-k">{lines_label}</span><span class="kpi-v">{lines_val}</span></div>
      <div class="kpi-pro"><span class="kpi-k">Grid Frequency</span><span class="kpi-v" style="color:{freq_col}">{freq:.3f} Hz</span></div>
      <div class="kpi-pro"><span class="kpi-k">Episode Score</span><span class="kpi-v" style="color:{score_col}">{state.cumulative_score:+.3f}</span></div>
      <div class="kpi-pro"><span class="kpi-k">SCADA Packets</span><span class="kpi-v">{packets}</span></div>
      <div class="kpi-pro"><span class="kpi-k">Active Spoofs</span><span class="kpi-v" style="color:{C['crimson'] if spoofs else C['green']}">{spoofs}</span></div>
    </div>
    """


def format_fleet_strip() -> str:
    """Static environment facts — always visible."""
    return f"""
    <div class="fleet-strip">
      <div class="fleet-item"><span class="fleet-k">Network</span><span class="fleet-v">20 nodes</span></div>
      <div class="fleet-item"><span class="fleet-k">Topology</span><span class="fleet-v">40 edges</span></div>
      <div class="fleet-item"><span class="fleet-k">Missions</span><span class="fleet-v">6 tasks</span></div>
      <div class="fleet-item"><span class="fleet-k">Threat model</span><span class="fleet-v" style="color:{C['crimson']}">3 vectors</span></div>
      <div class="fleet-item"><span class="fleet-k">Physics</span><span class="fleet-v" style="color:{C['accent']}">DC power flow</span></div>
      <div class="fleet-item"><span class="fleet-k">Oracle</span><span class="fleet-v" style="color:{C['indigo']}">Kirchhoff</span></div>
    </div>
    """


def format_mission_board(scores: Dict[int, float], active_id: int) -> str:
    cards = []
    for t in TASK_META:
        tid = t["id"]
        sc = scores.get(tid, 0.0)
        diff = t["difficulty"]
        dcol = DIFF_COL.get(diff, C["text_mid"])
        active = "mission-active" if tid == active_id else ""
        pct = int(sc * 100)
        cards.append(f"""
        <div class="mission-card {active}">
          <div class="mission-top">
            <span class="mission-id">T{tid}</span>
            <span class="mission-diff" style="color:{dcol}">{diff}</span>
          </div>
          <div class="mission-name">{t['name']}</div>
          <div class="mission-bar"><div class="mission-fill" style="width:{pct}%;background:{dcol}"></div></div>
          <div class="mission-score">{sc:.2f} / 1.00</div>
        </div>
        """)
    return f'<div class="mission-board">{"".join(cards)}</div>'


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD STATE
# ═══════════════════════════════════════════════════════════════════
class DashboardState:
    def __init__(self):
        self.task_scores: Dict[int, float] = {t["id"]: 0.0 for t in TASK_META}
        self.reset()

    def reset(self):
        self.current_obs: Dict[str, Any] = {}
        self.action_log: List[Dict] = []
        self.reward_history: List[Dict] = []
        self.packet_log: List[Dict] = []
        self.freq_history: List[float] = [60.0]
        self.frequency_hz: float = 60.0
        self.task_id: int = 0
        self.tick: int = 0
        self.cumulative_score: float = 0.0
        self.done: bool = False
        self.env = None

    def _push_freq(self, hz: float):
        self.freq_history.append(hz)
        if len(self.freq_history) > 100:
            self.freq_history = self.freq_history[-100:]

    def init_env(self) -> bool:
        try:
            import sys
            import os

            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root not in sys.path:
                sys.path.insert(0, root)
            try:
                from server.nexusgrid_environment import NexusgridEnvironment
            except ImportError:
                from nexusgrid_environment import NexusgridEnvironment
            self.env = NexusgridEnvironment()
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return False

    def do_reset(self, task_id: int, seed: int):
        if self.env is None:
            self.init_env()
        if self.env is None:
            return
        env = self.env
        self.action_log.clear()
        self.reward_history.clear()
        self.packet_log.clear()
        self.freq_history = [60.0]
        self.frequency_hz = 60.0
        self.tick = 0
        self.cumulative_score = 0.0
        self.done = False
        self.env = env
        self.task_id = task_id

        obs = self.env.reset(seed=seed, task_id=task_id)
        od = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
        self.current_obs = od
        self.frequency_hz = od.get("grid_frequency_hz", 60.0)
        self.tick = od.get("tick", 0)
        self.packet_log = od.get("network_packet_logs", [])
        self._push_freq(self.frequency_hz)

    def do_step(self, action_dict: Dict[str, Any]):
        if self.env is None:
            return
        import sys
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from models import GridAction
        try:
            action = GridAction(**action_dict)
        except Exception as e:
            print(f"[Dashboard] Bad action: {e}")
            return
        obs = self.env.step(action)
        od = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
        reward_val = od.get("reward", 0.0)
        self.current_obs = od
        self.frequency_hz = od.get("grid_frequency_hz", 60.0)
        self.tick = od.get("tick", 0)
        self.done = od.get("done", False)
        self.cumulative_score += reward_val if isinstance(reward_val, (int, float)) else 0
        self._push_freq(self.frequency_hz)

        self.action_log.append({
            "tick": self.tick,
            "action": action_dict.get("action_type", "?"),
            "params": json.dumps({k: v for k, v in action_dict.items() if k != "action_type"}),
            "reward": f"{reward_val:+.3f}" if isinstance(reward_val, (int, float)) else str(reward_val),
            "score": f"{self.cumulative_score:.3f}",
        })
        if pkts := od.get("network_packet_logs", []):
            self.packet_log = pkts
        meta = od.get("metadata", {})
        if isinstance(meta, dict):
            if bd := meta.get("reward_breakdown", {}):
                self.reward_history.append(bd)

        live_score = max(0.0, min(1.0, self.cumulative_score))
        self.task_scores[self.task_id] = max(
            self.task_scores.get(self.task_id, 0.0),
            live_score,
        )

        if self.done and self.env is not None:
            try:
                graded = self.env.get_score()
                if isinstance(graded, (int, float)):
                    self.task_scores[self.task_id] = max(
                        self.task_scores.get(self.task_id, 0.0),
                        max(0.0, min(1.0, graded)),
                    )
            except Exception:
                pass


dashboard_state = DashboardState()


def on_reset(task_id: int, seed: int):
    dashboard_state.do_reset(int(task_id), int(seed))
    return _refresh()


def on_step(action_type, node_id, edge_id, mw, status, hz_offset, duration, subgraph):
    ad: Dict[str, Any] = {"action_type": action_type}
    if action_type == "dispatch_generation":
        ad["node_id"] = node_id
        ad["mw"] = float(mw)
    elif action_type == "toggle_circuit_breaker":
        ad["edge_id"] = edge_id
        ad["status"] = status
    elif action_type == "run_state_estimation":
        try:
            ad["subgraph"] = json.loads(subgraph)
        except Exception:
            ad["subgraph"] = [s.strip() for s in subgraph.split(",") if s.strip()]
    elif action_type == "quarantine_scada_node":
        ad["node_id"] = node_id
    elif action_type == "inject_counter_signal":
        ad["node_id"] = node_id
        ad["hz_offset"] = float(hz_offset)
        ad["duration"] = int(duration)
    dashboard_state.do_step(ad)
    return _refresh()


def on_auto_run(task_id: int, seed: int, num_steps: int):
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dashboard_state.do_reset(int(task_id), int(seed))
    try:
        from inference import get_fallback_action
    except ImportError:
        def get_fallback_action(tid, t):
            return (
                {"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 100}
                if tid == 0
                else {"action_type": "advance_tick"}
            )

    for s in range(int(num_steps)):
        if dashboard_state.done:
            break
        dashboard_state.do_step(get_fallback_action(int(task_id), s))
    return _refresh()


def _refresh():
    obs = dashboard_state.current_obs
    st = dashboard_state
    return (
        _safe_chart(create_topology_graph, obs),
        _safe_chart(create_frequency_gauge, st.frequency_hz, st.freq_history),
        _safe_chart(create_power_flow_sankey, obs),
        _safe_chart(create_task_radar, st.task_scores),
        _safe_chart(create_reward_breakdown, st.reward_history),
        format_threat_feed(st.packet_log, obs),
        format_action_trace(st.action_log),
        format_status_bar(st),
        format_kpi_html(obs, st),
        format_mission_board(st.task_scores, st.task_id),
    )


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
  --bg: {C['bg']};
  --surface: {C['surface']};
  --surface-hi: {C['surface_hi']};
  --border: {C['border']};
  --border-hi: {C['border_hi']};
  --text: {C['text']};
  --text-mid: {C['text_mid']};
  --accent: {C['accent']};
  --indigo: {C['indigo']};
  --radius: 12px;
  --glow: 0 0 24px rgba(34,211,238,0.08);
}}

*, *::before, *::after {{ box-sizing: border-box; }}

body, .gradio-container, .main, .wrap, .app {{
  background: {C['bg']} !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(34,211,238,0.08), transparent),
    radial-gradient(ellipse 40% 30% at 100% 100%, rgba(129,140,248,0.06), transparent) !important;
  color: var(--text) !important;
  font-family: "DM Sans", system-ui, sans-serif !important;
}}

.gradio-container {{
  max-width: 1680px !important;
  margin: 0 auto !important;
  padding: 16px 20px 40px !important;
}}

.gradio-container footer {{ display: none !important; }}
.block, .form, .wrap, .contain, .gradio-group {{
  background: transparent !important;
  border-color: var(--border) !important;
}}

/* ── Header ── */
.pro-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  padding: 18px 22px;
  margin-bottom: 16px;
  background: linear-gradient(135deg, {C['surface']} 0%, {C['surface_hi']} 100%);
  border: 1px solid var(--border-hi);
  border-radius: var(--radius);
  box-shadow: var(--glow);
}}
.pro-header .brand {{
  display: flex;
  flex-direction: column;
  gap: 4px;
}}
.pro-header h1 {{
  margin: 0 !important;
  font-size: 1.45rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em;
  background: linear-gradient(90deg, {C['accent']}, {C['indigo']});
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.pro-header .sub {{
  font-size: 0.8rem;
  color: var(--text-mid);
  letter-spacing: 0.04em;
}}
.pro-badge {{
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(52,211,153,0.4);
  color: {C['green']};
  background: {C['green_soft']};
}}

/* ── Live KPI grid ── */
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}}
.kpi-pro {{
  background: {C['glass']};
  backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
.kpi-k {{
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-mid);
}}
.kpi-v {{
  font-family: "JetBrains Mono", monospace;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--accent);
}}

/* ── Status bar ── */
#status-bar textarea, #status-bar input {{
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: var(--accent) !important;
  background: {C['surface']} !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: var(--radius) !important;
  text-align: center !important;
  padding: 10px 14px !important;
  letter-spacing: 0.04em;
}}

/* ── Chart panels ── */
.chart-shell {{
  background: {C['surface']};
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 12px;
  box-shadow: var(--glow);
}}
.chart-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  background: rgba(0,0,0,0.2);
}}
.chart-head span {{
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-mid);
}}
.chart-head .dot {{
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 8px var(--accent);
}}

.plot-panel .block, .plot-panel > .wrap {{ padding: 4px 8px 12px !important; }}
.plot-panel .js-plotly-plot, .plot-panel .plotly-graph-div {{
  min-height: {PLOT_HEIGHT}px !important;
  width: 100% !important;
}}
.plot-panel-tall .js-plotly-plot, .plot-panel-tall .plotly-graph-div {{
  min-height: {PLOT_HEIGHT_TALL}px !important;
}}
.plot-panel-wide .js-plotly-plot, .plot-panel-wide .plotly-graph-div {{
  min-height: {PLOT_HEIGHT_WIDE}px !important;
}}

/* ── Sidebar ── */
.ctrl-box {{
  background: {C['surface']};
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  padding: 14px 16px;
  margin-bottom: 10px;
}}
.ctrl-box h3 {{
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--accent) !important;
  margin: 0 0 10px !important;
  padding: 0 !important;
  border: none !important;
}}

.gradio-container label span {{
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  color: var(--text-mid) !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}
.gradio-container input, .gradio-container select, .gradio-container textarea {{
  background: {C['bg']} !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.82rem !important;
}}
.gradio-container input:focus, .gradio-container select:focus, .gradio-container textarea:focus {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(34,211,238,0.2) !important;
}}

.mono-feed textarea {{
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.78rem !important;
  line-height: 1.55 !important;
  background: {C['bg']} !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: #cbd5e1 !important;
  padding: 14px !important;
  min-height: 300px !important;
}}
.log-threat textarea {{ border-color: rgba(248,113,113,0.35) !important; }}
.log-action textarea {{ border-color: rgba(34,211,238,0.28) !important; }}

.btn-reset, .btn-step, .btn-auto {{
  font-weight: 700 !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border-radius: 8px !important;
  padding: 12px !important;
  width: 100%;
  border: none !important;
  transition: transform 0.15s, box-shadow 0.15s !important;
}}
.btn-reset {{ background: linear-gradient(135deg, #2563eb, #3b82f6) !important; color: #fff !important; }}
.btn-step {{ background: linear-gradient(135deg, #0e7490, #22d3ee) !important; color: #0f172a !important; }}
.btn-auto {{ background: linear-gradient(135deg, #5b21b6, #a78bfa) !important; color: #fff !important; }}
.btn-reset:hover, .btn-step:hover, .btn-auto:hover {{
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(34,211,238,0.2) !important;
}}

.legend-block {{
  font-size: 0.75rem;
  color: var(--text-mid);
  line-height: 1.8;
  background: {C['surface']};
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
}}

/* Fleet + mission */
.fleet-strip {{
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 8px;
  margin-bottom: 14px;
}}
.fleet-item {{
  background: {C['surface']};
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px;
  text-align: center;
}}
.fleet-k {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-mid); display: block; }}
.fleet-v {{ font-family: "JetBrains Mono", monospace; font-size: 0.85rem; font-weight: 600; color: var(--accent); margin-top: 4px; display: block; }}

.ops-section {{
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin: 18px 0 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}}
.ops-num {{
  font-family: "JetBrains Mono", monospace;
  font-size: 0.7rem;
  font-weight: 700;
  color: {C['accent']};
  background: {C['accent_soft']};
  padding: 4px 8px;
  border-radius: 6px;
}}
.ops-title {{
  font-size: 0.85rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text);
}}
.ops-sub {{
  font-size: 0.75rem;
  color: var(--text-mid);
  margin-left: auto;
}}

.mission-board {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-bottom: 12px;
}}
.mission-card {{
  background: {C['surface']};
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
}}
.mission-active {{
  border-color: {C['accent']};
  box-shadow: 0 0 0 1px rgba(34,211,238,0.25);
}}
.mission-top {{ display: flex; justify-content: space-between; align-items: center; }}
.mission-id {{ font-family: "JetBrains Mono", monospace; font-weight: 700; color: {C['accent']}; }}
.mission-diff {{ font-size: 0.65rem; font-weight: 700; text-transform: uppercase; }}
.mission-name {{ font-size: 0.78rem; color: var(--text); margin: 6px 0; font-weight: 600; }}
.mission-bar {{ height: 4px; background: {C['border']}; border-radius: 2px; overflow: hidden; }}
.mission-fill {{ height: 100%; border-radius: 2px; }}
.mission-score {{ font-family: "JetBrains Mono", monospace; font-size: 0.68rem; color: var(--text-mid); margin-top: 6px; }}

.arch-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
  font-size: 0.72rem;
  color: var(--text-mid);
}}
.arch-pill {{
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: {C['surface']};
}}

@media (max-width: 1100px) {{
  .kpi-grid, .fleet-strip {{ grid-template-columns: repeat(3, 1fr); }}
  .mission-board {{ grid-template-columns: repeat(2, 1fr); }}
}}
@media (max-width: 640px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .gradio-container {{ padding: 10px !important; }}
}}
"""

IDLE_THREAT = (
    "SCADA THREAT MONITOR · Kirchhoff Oracle\n"
    + ("─" * 52)
    + "\n\n  System offline — reset to begin monitoring.\n"
)
IDLE_TRACE = (
    "AGENT ACTION TRACE · Decision log\n"
    + ("─" * 48)
    + "\n\n  No actions recorded — reset and step the environment.\n"
)


def create_dashboard() -> gr.Blocks:
    if not PLOTLY_AVAILABLE:
        with gr.Blocks(title="NexusGrid Dashboard") as demo:
            gr.Markdown(
                "# NexusGrid Dashboard\n\n"
                "Install `plotly` for the full chart UI. API endpoints remain available."
            )
        return demo

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        demo = gr.Blocks(title="NexusGrid — Grid Defense Dashboard")

    with demo:
        gr.HTML(f"<style>{CSS}</style>")

        gr.HTML(f"""
        <div class="pro-header">
          <div class="brand">
            <h1>NEXUSGRID</h1>
            <span class="sub">Defend the national grid · Physics + SCADA + agent reasoning</span>
          </div>
          <span class="pro-badge">LIVE OPS</span>
        </div>
        <div class="arch-strip">
          <span class="arch-pill">Layer 1 · DC Power Flow</span>
          <span class="arch-pill">Layer 2 · SCADA / Spoofs</span>
          <span class="arch-pill">Layer 3 · OpenEnv API</span>
          <span class="arch-pill">Layer 4 · LLM Agent</span>
        </div>
        """)

        gr.HTML(format_fleet_strip())
        kpi_panel = gr.HTML(format_kpi_html({}, dashboard_state))

        status_bar = gr.Textbox(
            value="TASK — | TICK 000 | FREQ 60.000 Hz | SCORE +0.0000 | IDLE",
            label="",
            interactive=False,
            elem_id="status-bar",
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=240):
                gr.HTML('<div class="ctrl-box"><h3>Episode Control</h3></div>')
                task_select = gr.Slider(0, 5, value=0, step=1, label="Mission / Task ID")
                seed_input = gr.Number(value=42, label="Random seed", precision=0)
                reset_btn = gr.Button("Initialise Grid", elem_classes=["btn-reset"])

                gr.HTML('<div class="ctrl-box"><h3>Manual Action</h3></div>')
                action_type = gr.Dropdown(
                    [
                        "dispatch_generation", "toggle_circuit_breaker",
                        "run_state_estimation", "quarantine_scada_node",
                        "inject_counter_signal", "advance_tick",
                    ],
                    value="dispatch_generation",
                    label="Action type",
                )
                node_id = gr.Textbox(value="NODE_01", label="Node ID")
                edge_id = gr.Textbox(value="LINE_01", label="Edge ID")
                mw_val = gr.Number(value=100, label="MW value")
                breaker_status = gr.Dropdown(["OPEN", "CLOSED"], value="CLOSED", label="Breaker status")
                hz_offset = gr.Number(value=-0.5, label="Hz offset")
                duration = gr.Number(value=5, precision=0, label="Duration")
                subgraph = gr.Textbox(value='["NODE_14","NODE_15"]', label="Subgraph JSON")
                step_btn = gr.Button("Execute Step", elem_classes=["btn-step"])

                gr.HTML('<div class="ctrl-box"><h3>Autonomous Agent</h3></div>')
                auto_steps = gr.Slider(1, 50, value=10, step=1, label="Steps")
                auto_btn = gr.Button("Run Agent", elem_classes=["btn-auto"])

                gr.HTML('<div class="ctrl-box"><h3>Mission Suite</h3></div>')
                mission_panel = gr.HTML(format_mission_board(dashboard_state.task_scores, 0))

                gr.HTML(f"""
                <div class="legend-block">
                  <b style="color:{C['accent']}">Transmission load</b><br>
                  <span style="color:#64748b">━━</span> &lt;50% &nbsp;
                  <span style="color:{C['accent']}">━━</span> 50–80% &nbsp;
                  <span style="color:{C['amber']}">━━</span> 80–95% &nbsp;
                  <span style="color:{C['crimson']}">━━</span> ≥95%<br><br>
                  <b style="color:{C['green']}">Frequency bands</b><br>
                  <span style="color:{C['green']}">●</span> 59.7–60.3 NOMINAL<br>
                  <span style="color:{C['amber']}">●</span> 59.5–59.7 WARNING<br>
                  <span style="color:{C['orange']}">●</span> 59.2–59.5 DANGER<br>
                  <span style="color:{C['red']}">●</span> 59.0–59.2 CRITICAL<br>
                  <span style="color:{C['crimson']}">●</span> &lt;59.0 BLACKOUT
                </div>
                """)

            with gr.Column(scale=4):
                gr.HTML(_section('01', 'Physical Grid', 'Topology · frequency · MW flow'))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        gr.HTML(_shell('Grid Topology · 20-node network', C['accent']))
                        topo_plot = gr.Plot(label='', elem_classes=['plot-panel', 'plot-panel-tall'])
                    with gr.Column(scale=2):
                        gr.HTML(_shell('Grid Frequency Monitor', C['green']))
                        freq_plot = gr.Plot(label='', elem_classes=['plot-panel'])

                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        gr.HTML(_shell('Power Flow / Transmission Capacity', C['accent']))
                        sankey_plot = gr.Plot(label='', elem_classes=['plot-panel', 'plot-panel-wide'])
                    with gr.Column(scale=2):
                        gr.HTML(_shell('Mission Performance Radar', C['purple']))
                        radar_plot = gr.Plot(label='', elem_classes=['plot-panel'])

                gr.HTML(_section('02', 'Agent Training', 'Reward rubric signals'))
                gr.HTML(_shell('Reward Signal Decomposition', C['green']))
                reward_plot = gr.Plot(label='', elem_classes=['plot-panel'])

                gr.HTML(_section('03', 'Cyber Layer', 'SCADA · Kirchhoff oracle · agent log'))
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.HTML(_shell('SCADA Threat Monitor', C['red']))
                        threat_feed = gr.Textbox(
                            value=IDLE_THREAT, label='', lines=14,
                            interactive=False, elem_classes=['mono-feed', 'log-threat'],
                        )
                    with gr.Column():
                        gr.HTML(_shell('Agent Decision Log', C['accent']))
                        action_trace = gr.Textbox(
                            value=IDLE_TRACE, label='', lines=14,
                            interactive=False, elem_classes=['mono-feed', 'log-action'],
                        )


        all_out = [
            topo_plot, freq_plot, sankey_plot, radar_plot, reward_plot,
            threat_feed, action_trace, status_bar, kpi_panel, mission_panel,
        ]
        reset_btn.click(fn=on_reset, inputs=[task_select, seed_input], outputs=all_out)
        step_btn.click(
            fn=on_step,
            inputs=[action_type, node_id, edge_id, mw_val, breaker_status, hz_offset, duration, subgraph],
            outputs=all_out,
        )
        auto_btn.click(fn=on_auto_run, inputs=[task_select, seed_input, auto_steps], outputs=all_out)

    return demo
