"""
NexusGrid Gradio Dashboard — APEX EDITION ⚡
════════════════════════════════════════════════════════════════════════
Aesthetic : Nuclear Command Bunker × Holographic War-Room
Fonts     : Orbitron (display) · Space Mono (terminal) · JetBrains Mono (data)
Palette   : Electric teal · Danger crimson · Neon amber · Deep void · Plasma purple

7 Live Panels
─────────────
1. Grid Topology Map       — Radial hub-spoke with Bézier edges & glow halos
2. Frequency Gauge         — Gauge + sparkline + zone badge
3. Power Flow Sankey       — Real-time energy flow between node types
4. SCADA Threat Feed       — Terminal log with severity heatmap
5. Agent Action Trace      — Decision log with reward arrows
6. Reward Breakdown        — Dual-bar waterfall accumulator
7. Task Progress Radar     — 6-axis completion radar

Responsive : 320 → 768 → 1280 → 1920 → ∞
"""

from __future__ import annotations

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import math
from typing import Dict, Any, List

# ═══════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════
C: Dict[str, str] = {
    "void":       "#030711",
    "deep":       "#060b16",
    "panel":      "#0a0f1e",
    "panel2":     "#0d1424",
    "panel3":     "#111a30",
    "border":     "#162035",
    "border_hi":  "#243a66",
    "teal":       "#00f5d4",
    "teal_dim":   "#00b89e",
    "teal_glow":  "rgba(0,245,212,0.12)",
    "cyan":       "#22d3ee",
    "blue":       "#0ea5e9",
    "blue_dim":   "#0369a1",
    "purple":     "#a855f7",
    "purple_dim": "#7c3aed",
    "green":      "#22d3a0",
    "green_glow": "rgba(34,211,160,0.18)",
    "amber":      "#fbbf24",
    "amber_glow": "rgba(251,191,36,0.18)",
    "orange":     "#f97316",
    "red":        "#ef4444",
    "red_glow":   "rgba(239,68,68,0.22)",
    "crimson":    "#ff2d55",
    "text":       "#d8e2ef",
    "text_dim":   "#4a5f7a",
    "text_hi":    "#f8fafc",
    "white":      "#ffffff",
}

# Node-type palette, icons, and ring assignments
TYPE_COL = {
    "hydro": "#38bdf8", "nuclear": "#4ade80", "gas": "#fb923c",
    "wind": "#22d3ee", "solar": "#fbbf24", "battery": "#c084fc", "load": "#64748b",
}
TYPE_ICON = {
    "hydro": "💧", "nuclear": "☢", "gas": "🔥", "wind": "🌬",
    "solar": "☀", "battery": "⚡", "load": "🏙",
}
TYPE_RING = {
    "hydro": 0, "nuclear": 0, "gas": 0,
    "wind": 1, "solar": 1, "battery": 1, "load": 2,
}
RING_R = {0: 0.28, 1: 0.58, 2: 0.90}

# Task metadata for radar chart
TASK_META = [
    {"id": 0, "name": "Smoke Test",       "max": 1.0, "difficulty": "Trivial"},
    {"id": 1, "name": "Duck Curve",        "max": 1.0, "difficulty": "Easy"},
    {"id": 2, "name": "Cascade Overload",  "max": 1.0, "difficulty": "Medium"},
    {"id": 3, "name": "Phantom Injection", "max": 1.0, "difficulty": "Hard"},
    {"id": 4, "name": "Stuxnet Resonance", "max": 1.0, "difficulty": "Very Hard"},
    {"id": 5, "name": "Black Start",       "max": 1.0, "difficulty": "Expert"},
]


# ═══════════════════════════════════════════════════════════════════
# PLOTLY DARK BASE LAYOUT
# ═══════════════════════════════════════════════════════════════════
def _dark_layout(**kw) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Space Mono', monospace", color=C["text"]),
        margin=dict(l=14, r=14, t=40, b=14),
        autosize=True,
    )
    base.update(kw)
    return base


# ═══════════════════════════════════════════════════════════════════
# PANEL 1 — GRID TOPOLOGY (radial hub-spoke with glow halos)
# ═══════════════════════════════════════════════════════════════════
def create_topology_graph(obs_dict: Dict[str, Any]) -> go.Figure:
    topo  = obs_dict.get("topology_graph", {})
    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.update_layout(
            **_dark_layout(margin=dict(l=20, r=20, t=50, b=20)),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        # decorative concentric rings
        for r_val, alpha in [(0.28, 0.08), (0.58, 0.05), (0.90, 0.03)]:
            angles = [i * 6 for i in range(61)]
            xs = [r_val * math.cos(math.radians(a)) for a in angles]
            ys = [r_val * math.sin(math.radians(a)) for a in angles]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=f"rgba(0,245,212,{alpha})", width=1, dash="dot"),
                showlegend=False, hoverinfo="none",
            ))
        # crosshair
        for angle in [0, 45, 90, 135]:
            rad = math.radians(angle)
            fig.add_trace(go.Scatter(
                x=[-math.cos(rad), math.cos(rad)],
                y=[-math.sin(rad), math.sin(rad)],
                mode="lines", line=dict(color="rgba(0,245,212,0.03)", width=1),
                showlegend=False, hoverinfo="none",
            ))
        fig.add_annotation(
            text="◌  SYSTEM OFFLINE", xref="paper", yref="paper", x=0.5, y=0.55,
            showarrow=False, font=dict(size=15, color=C["teal"], family="Orbitron"),
        )
        fig.add_annotation(
            text="Reset environment to initialise the grid",
            xref="paper", yref="paper", x=0.5, y=0.43, showarrow=False,
            font=dict(size=9, color=C["text_dim"], family="Space Mono"),
        )
        return fig

    # ── Bucket nodes by ring ──────────────────────────────────────
    buckets: Dict[int, list] = {0: [], 1: [], 2: []}
    for nd in nodes:
        ring = TYPE_RING.get(nd.get("node_type", "load"), 2)
        buckets[ring].append(nd)

    # ── Assign (x, y) via polar → cartesian ──────────────────────
    pos: Dict[str, tuple] = {}
    for ring_id, bucket in buckets.items():
        n = len(bucket)
        r = RING_R[ring_id]
        angle_offset = 12 * ring_id
        for i, nd in enumerate(bucket):
            deg = angle_offset + (360 * i / max(n, 1))
            rad = math.radians(deg)
            pos[nd["id"]] = (r * math.cos(rad), r * math.sin(rad))

    # ── Decorative background rings ──────────────────────────────
    ring_traces = []
    for r_val, alpha in [(0.28, 0.06), (0.58, 0.04), (0.90, 0.025)]:
        angles = list(range(0, 361, 3))
        xs = [r_val * math.cos(math.radians(a)) for a in angles]
        ys = [r_val * math.sin(math.radians(a)) for a in angles]
        ring_traces.append(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=f"rgba(0,245,212,{alpha})", width=0.8),
            showlegend=False, hoverinfo="none",
        ))

    # ── Edge traces with Bézier-like curves ──────────────────────
    edge_traces: List[go.Scatter] = []
    for edge in edges:
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if src not in pos or tgt not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        cap  = max(edge.get("capacity_mw", 1), 1)
        load = edge.get("current_load_mw", 0)
        status = edge.get("status", "LIVE")
        pct = load / cap

        if status == "TRIPPED":
            col, w, opa, dash = C["text_dim"], 0.8, 0.15, "dot"
        elif pct >= 0.95:
            col, w, opa, dash = C["crimson"], 3.2, 0.95, "solid"
        elif pct >= 0.80:
            col, w, opa, dash = C["amber"], 2.4, 0.85, "solid"
        elif pct >= 0.50:
            col, w, opa, dash = C["teal"], 1.6, 0.55, "solid"
        else:
            col, w, opa, dash = C["teal_dim"], 0.9, 0.25, "solid"

        # Compute a slight Bézier bulge by pulling midpoint toward center
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dist = math.sqrt(mx**2 + my**2)
        if dist > 0.01:
            bulge = 0.12
            cx = mx - bulge * mx / dist
            cy = my - bulge * my / dist
        else:
            cx, cy = mx, my

        # 3-point path for curve effect
        pts = 8
        curve_x, curve_y = [], []
        for k in range(pts + 1):
            t = k / pts
            bx = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
            by = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
            curve_x.append(bx)
            curve_y.append(by)
        curve_x.append(None)
        curve_y.append(None)

        edge_traces.append(go.Scatter(
            x=curve_x, y=curve_y, mode="lines",
            line=dict(width=w, color=col, dash=dash),
            opacity=opa, hoverinfo="none", showlegend=False,
        ))

    # ── Glow halo traces for critical / spoofed nodes ────────────
    halo_traces = []
    for nd in nodes:
        nid = nd["id"]
        crit = nd.get("critical", False)
        spoof = nd.get("spoofed", False)
        if not (crit or spoof) or nid not in pos:
            continue
        x, y = pos[nid]
        halo_col = "rgba(255,45,85,0.25)" if crit else "rgba(251,191,36,0.20)"
        halo_size = 38 if crit else 32
        halo_traces.append(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=halo_size, color=halo_col, line=dict(width=0)),
            showlegend=False, hoverinfo="none",
        ))

    # ── Node trace ────────────────────────────────────────────────
    nx_l, ny_l, nc_l, nb_l, ns_l = [], [], [], [], []
    hover_l, label_l = [], []

    for nd in nodes:
        nid = nd["id"]
        t    = nd.get("node_type", "load")
        crit = nd.get("critical", False)
        en   = nd.get("energized", True)
        cap  = nd.get("capacity_mw", 0)
        gen  = nd.get("generation_mw", 0)
        spoof = nd.get("spoofed", False)
        x, y = pos.get(nid, (0, 0))

        nx_l.append(x); ny_l.append(y)
        base = TYPE_COL.get(t, C["text_dim"])
        fill = base if en else "#141c2c"
        border = C["crimson"] if crit else (C["amber"] if spoof else base)
        nc_l.append(fill)
        nb_l.append(border)
        raw = 13 + min(int(cap / 280), 13)
        if crit: raw += 5
        if t == "load": raw = max(raw - 5, 9)
        ns_l.append(raw)

        icon = TYPE_ICON.get(t, "·")
        short = nid.replace("NODE_", "")
        label_l.append(f"{icon}{short}")

        crit_tag  = " 🔴CRIT" if crit else ""
        spoof_tag = " ⚠SPOOF" if spoof else ""
        state_str = "⚡ LIVE" if en else "🌑 DARK"
        hover_l.append(
            f"<b style='color:{base}'>{nid}{crit_tag}{spoof_tag}</b><br>"
            f"Type : {t.upper()}<br>State : {state_str}<br>"
            f"Cap  : {cap:,.0f} MW<br>Gen  : {gen:,.0f} MW"
        )

    node_trace = go.Scatter(
        x=nx_l, y=ny_l, mode="markers+text",
        text=label_l, textposition="top center",
        textfont=dict(size=6, color=C["text_dim"], family="Space Mono"),
        hovertext=hover_l, hoverinfo="text",
        marker=dict(
            size=ns_l, color=nc_l,
            line=dict(width=2.0, color=nb_l),
            opacity=0.95, symbol="circle",
        ),
        showlegend=False,
    )

    # ── Legend ghost traces ───────────────────────────────────────
    seen: set = set()
    legend: List[go.Scatter] = []
    for nd in nodes:
        t = nd.get("node_type", "load")
        if t in seen:
            continue
        seen.add(t)
        legend.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=TYPE_COL.get(t, C["text_dim"]),
                        line=dict(width=1, color=TYPE_COL.get(t, C["text_dim"]))),
            name=f"{TYPE_ICON.get(t, '·')} {t.capitalize()}",
            showlegend=True,
        ))

    # ── Ring labels ───────────────────────────────────────────────
    ring_labels = {0.28: "GENERATION", 0.58: "RENEWABLES / STORAGE", 0.90: "LOAD CENTRES"}
    ring_annots = []
    for r, lbl in ring_labels.items():
        ring_annots.append(dict(
            x=0, y=r + 0.05,
            text=f"<span style='font-size:6px;color:{C['text_dim']};font-family:Orbitron;"
                 f"letter-spacing:0.15em'>{lbl}</span>",
            showarrow=False, xref="x", yref="y", xanchor="center",
        ))

    fig = go.Figure(data=ring_traces + edge_traces + halo_traces + [node_trace] + legend)
    fig.update_layout(
        **_dark_layout(margin=dict(l=6, r=6, t=38, b=38)),
        title=dict(
            text="⚡  NEXUSGRID — 20-NODE TRANSMISSION NETWORK",
            font=dict(size=10, color=C["teal"], family="Orbitron"),
            x=0.02, xanchor="left",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.10,
            xanchor="center", x=0.5,
            font=dict(size=7, color=C["text_dim"], family="Space Mono"),
            bgcolor="rgba(0,0,0,0)", itemsizing="constant",
        ),
        xaxis=dict(visible=False, range=[-1.15, 1.30], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-1.15, 1.15]),
        hoverlabel=dict(bgcolor=C["panel2"], font_size=10,
                        font_family="Space Mono", bordercolor=C["border_hi"]),
        annotations=ring_annots,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# PANEL 2 — FREQUENCY GAUGE + SPARKLINE HISTORY
# ═══════════════════════════════════════════════════════════════════
def create_frequency_gauge(freq_hz: float,
                           history: List[float] | None = None) -> go.Figure:
    freq_hz = round(freq_hz, 3)

    if   freq_hz < 59.0: col, zone, zbg = C["crimson"], "BLACKOUT", "rgba(255,45,85,0.15)"
    elif freq_hz < 59.2: col, zone, zbg = C["red"],     "CRITICAL", "rgba(239,68,68,0.12)"
    elif freq_hz < 59.5: col, zone, zbg = C["orange"],  "DANGER",   "rgba(249,115,22,0.10)"
    elif freq_hz < 59.7: col, zone, zbg = C["amber"],   "WARNING",  "rgba(251,191,36,0.10)"
    elif freq_hz <= 60.3: col, zone, zbg = C["green"],  "NOMINAL",  "rgba(34,211,160,0.10)"
    else:                 col, zone, zbg = C["amber"],  "ELEVATED", "rgba(251,191,36,0.10)"

    delta = round(freq_hz - 60.0, 3)
    delta_sign = "+" if delta >= 0 else ""

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.70, 0.30],
        vertical_spacing=0.04,
        specs=[[{"type": "indicator"}], [{"type": "scatter"}]],
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=freq_hz,
        domain={"x": [0, 1], "y": [0, 1]},
        number=dict(suffix=" Hz",
                    font=dict(size=34, color=col, family="Orbitron"),
                    valueformat=".3f"),
        title=dict(text="GRID FREQUENCY",
                   font=dict(size=8, color=C["teal_dim"], family="Orbitron")),
        gauge=dict(
            axis=dict(range=[58.5, 61.5],
                      tickvals=[59.0, 59.5, 60.0, 60.5, 61.0],
                      ticktext=["59.0", "59.5", "60.0", "60.5", "61.0"],
                      tickwidth=1, tickcolor=C["border_hi"],
                      tickfont=dict(size=7, color=C["text_dim"], family="Space Mono")),
            bar=dict(color=col, thickness=0.18),
            bgcolor=C["void"], borderwidth=1, bordercolor=C["border"],
            steps=[
                dict(range=[58.5, 59.0], color="rgba(255,45,85,0.28)"),
                dict(range=[59.0, 59.2], color="rgba(239,68,68,0.20)"),
                dict(range=[59.2, 59.5], color="rgba(249,115,22,0.16)"),
                dict(range=[59.5, 59.7], color="rgba(251,191,36,0.12)"),
                dict(range=[59.7, 60.3], color="rgba(34,211,160,0.16)"),
                dict(range=[60.3, 61.5], color="rgba(251,191,36,0.12)"),
            ],
            threshold=dict(line=dict(color=C["white"], width=2),
                           thickness=0.70, value=freq_hz),
        ),
    ), row=1, col=1)

    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=0.30,
        text=f"<span style='font-family:Orbitron;font-size:11px;"
             f"font-weight:800;color:{col};letter-spacing:0.18em'>{zone}</span>",
        showarrow=False,
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=0.27,
        text=f"<span style='font-family:Space Mono;font-size:8px;"
             f"color:{C['text_dim']}'>Δ {delta_sign}{delta:.3f} Hz</span>",
        showarrow=False,
    )

    # Sparkline
    hist = list(history) if history else [60.0]
    xs = list(range(len(hist)))
    fig.add_trace(go.Scatter(
        x=xs, y=hist, mode="lines",
        line=dict(color=col, width=1.6, shape="spline"),
        fill="tozeroy", fillcolor=zbg,
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0, max(len(hist)-1, 1)], y=[60.0, 60.0], mode="lines",
        line=dict(color=C["teal_dim"], width=0.8, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(
        range=[58.8, 61.2], gridcolor=C["border"], gridwidth=1,
        tickfont=dict(size=6, color=C["text_dim"], family="Space Mono"),
        tickvals=[59.0, 60.0, 61.0], ticktext=["59", "60", "61"],
        row=2, col=1,
    )
    fig.update_layout(**_dark_layout(margin=dict(l=8, r=8, t=14, b=8)))
    return fig


# ═══════════════════════════════════════════════════════════════════
# PANEL 3 — POWER FLOW SANKEY DIAGRAM
# ═══════════════════════════════════════════════════════════════════
def create_power_flow_sankey(obs_dict: Dict[str, Any]) -> go.Figure:
    topo = obs_dict.get("topology_graph", {})
    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.update_layout(**_dark_layout())
        fig.add_annotation(
            text="◌  AWAITING GRID DATA", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color=C["text_dim"], family="Orbitron"),
        )
        return fig

    # Group nodes by type for Sankey
    type_groups = {}
    node_idx = {}
    for nd in nodes:
        t = nd.get("node_type", "load")
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(nd)
        node_idx[nd["id"]] = len(node_idx)

    # Build Sankey from edges
    sankey_src, sankey_tgt, sankey_val, sankey_col = [], [], [], []
    for edge in edges:
        s, t = edge.get("source", ""), edge.get("target", "")
        if s not in node_idx or t not in node_idx:
            continue
        flow = max(edge.get("current_load_mw", 0), 0)
        if flow <= 0:
            continue
        cap = max(edge.get("capacity_mw", 1), 1)
        pct = flow / cap
        if pct >= 0.95:
            ecol = "rgba(255,45,85,0.6)"
        elif pct >= 0.80:
            ecol = "rgba(251,191,36,0.5)"
        elif pct >= 0.50:
            ecol = "rgba(0,245,212,0.35)"
        else:
            ecol = "rgba(0,184,158,0.18)"
        sankey_src.append(node_idx[s])
        sankey_tgt.append(node_idx[t])
        sankey_val.append(flow)
        sankey_col.append(ecol)

    # Node labels and colors
    all_nodes = list(node_idx.keys())
    ncolors = []
    nlabels = []
    for nid in all_nodes:
        nd = next((n for n in nodes if n["id"] == nid), None)
        if nd:
            t = nd.get("node_type", "load")
            ncolors.append(TYPE_COL.get(t, C["text_dim"]))
            nlabels.append(f"{TYPE_ICON.get(t, '')} {nid.replace('NODE_', 'N')}")
        else:
            ncolors.append(C["text_dim"])
            nlabels.append(nid)

    if not sankey_val:
        sankey_src = [0]
        sankey_tgt = [min(1, len(all_nodes) - 1)]
        sankey_val = [0.1]
        sankey_col = ["rgba(0,0,0,0)"]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12, thickness=14,
            line=dict(color=C["border_hi"], width=0.5),
            label=nlabels, color=ncolors,
        ),
        link=dict(
            source=sankey_src, target=sankey_tgt,
            value=sankey_val, color=sankey_col,
        ),
    ))
    fig.update_layout(
        **_dark_layout(margin=dict(l=6, r=6, t=36, b=6)),
        title=dict(
            text="🔄  POWER FLOW — REAL-TIME ENERGY DISTRIBUTION",
            font=dict(size=10, color=C["cyan"], family="Orbitron"),
            x=0.02, xanchor="left",
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# PANEL 7 — TASK PROGRESS RADAR
# ═══════════════════════════════════════════════════════════════════
def create_task_radar(scores: Dict[int, float]) -> go.Figure:
    categories = [f"T{t['id']}\n{t['name']}" for t in TASK_META]
    values = [scores.get(t["id"], 0.0) for t in TASK_META]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    # Filled area
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=categories_closed,
        fill="toself",
        fillcolor="rgba(0,245,212,0.12)",
        line=dict(color=C["teal"], width=2),
        marker=dict(size=6, color=C["teal"], line=dict(width=1, color=C["white"])),
        name="Score",
    ))

    # Reference line at 0.5
    ref = [0.5] * (len(categories) + 1)
    fig.add_trace(go.Scatterpolar(
        r=ref, theta=categories_closed,
        line=dict(color=C["amber"], width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        **_dark_layout(margin=dict(l=50, r=50, t=36, b=20)),
        title=dict(
            text="🎯  TASK COMPLETION RADAR",
            font=dict(size=10, color=C["purple"], family="Orbitron"),
            x=0.02, xanchor="left",
        ),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1.0],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["0.25", "0.50", "0.75", "1.00"],
                tickfont=dict(size=6, color=C["text_dim"]),
                gridcolor=C["border"], linecolor=C["border"],
            ),
            angularaxis=dict(
                tickfont=dict(size=7, color=C["teal_dim"], family="Space Mono"),
                linecolor=C["border"], gridcolor=C["border"],
            ),
        ),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# PANEL 5 — REWARD BREAKDOWN (dual-bar waterfall)
# ═══════════════════════════════════════════════════════════════════
def create_reward_breakdown(reward_history: List[Dict[str, float]]) -> go.Figure:
    POS = ["fault_isolation", "cyber_detection", "frequency_stable",
           "proactive_dispatch", "reasoning_order", "stability_bonus"]
    NEG = ["overload_penalty", "hallucination_penalty", "redundant_estimation"]
    ICONS = {
        "fault_isolation": "⚡", "cyber_detection": "🔐", "frequency_stable": "〰",
        "proactive_dispatch": "🚀", "reasoning_order": "🧠", "stability_bonus": "✦",
        "overload_penalty": "⚠", "hallucination_penalty": "👻", "redundant_estimation": "🔁",
    }

    acc: Dict[str, float] = {}
    for entry in reward_history:
        for k, v in entry.items():
            acc[k] = acc.get(k, 0.0) + v

    labels, values = [], []
    for sig in POS:
        v = acc.get(sig, 0.0)
        labels.append(f"{ICONS.get(sig, '·')}  {sig.replace('_', ' ').upper()}")
        values.append(v)
    for sig in NEG:
        v = acc.get(sig, 0.0)
        labels.append(f"{ICONS.get(sig, '·')}  {sig.replace('_', ' ').upper()}")
        values.append(v)

    if not any(values):
        labels = ["AWAITING SIGNAL DATA"]
        values = [0.0]

    pos_vals = [v if v > 0 else 0 for v in values]
    neg_vals = [v if v < 0 else 0 for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pos_vals, y=labels, orientation="h",
        marker=dict(color=[C["green"] if v > 0 else "rgba(0,0,0,0)" for v in values],
                    line=dict(width=0), opacity=0.85),
        text=[f" +{v:.3f}" if v > 0 else "" for v in values],
        textposition="outside",
        textfont=dict(color=C["green"], size=8, family="Space Mono"),
        showlegend=False, hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=neg_vals, y=labels, orientation="h",
        marker=dict(color=[C["crimson"] if v < 0 else "rgba(0,0,0,0)" for v in values],
                    line=dict(width=0), opacity=0.85),
        text=[f" {v:.3f}" if v < 0 else "" for v in values],
        textposition="outside",
        textfont=dict(color=C["crimson"], size=8, family="Space Mono"),
        showlegend=False, hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))

    max_abs = max(abs(v) for v in values) if values else 0.5
    fig.add_vline(x=0, line_color=C["border_hi"], line_width=1)
    fig.add_vrect(x0=0, x1=max_abs * 1.4,
                  fillcolor="rgba(34,211,160,0.03)", layer="below", line_width=0)
    if min(values + [0]) < 0:
        fig.add_vrect(x0=-max_abs * 1.4, x1=0,
                      fillcolor="rgba(239,68,68,0.03)", layer="below", line_width=0)

    fig.update_layout(
        **_dark_layout(margin=dict(l=10, r=50, t=38, b=14)),
        barmode="overlay",
        title=dict(text="📊  REWARD SIGNAL ACCUMULATOR",
                   font=dict(size=10, color=C["amber"], family="Orbitron"),
                   x=0.0, xanchor="left"),
        xaxis=dict(title="Accumulated Δ Reward", gridcolor=C["border"],
                   zerolinecolor=C["border_hi"],
                   tickfont=dict(family="Space Mono", size=7)),
        yaxis=dict(autorange="reversed",
                   tickfont=dict(family="Space Mono", size=7, color=C["teal_dim"])),
        bargap=0.30,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# TEXT FORMATTERS
# ═══════════════════════════════════════════════════════════════════
def format_threat_feed(packets: List[Dict], obs: Dict) -> str:
    hdr = (
        "╔════════════════════════════════════════════════════════╗\n"
        "║  🔐  SCADA THREAT MONITOR  ·  KIRCHHOFF ORACLE  ║\n"
        "╚════════════════════════════════════════════════════════╝\n\n"
    )
    if not packets:
        return hdr + "  ◌  System offline — reset to begin monitoring.\n"

    lines = [
        hdr,
        f"  {'TIME':>5}  {'STATUS':<11}  {'ROUTE':<17}  {'LAT':>5}\n",
        f"  {'─'*5}  {'─'*11}  {'─'*17}  {'─'*5}\n",
    ]
    for pkt in packets[-16:]:
        src = str(pkt.get("source_node", "?"))[:8]
        dst = str(pkt.get("dest_node", "?"))[:8]
        lat = pkt.get("latency_ms", 0)
        anomaly = pkt.get("anomaly_flag", False)
        ts  = pkt.get("timestamp", 0)
        route = f"{src}→{dst}"
        if anomaly:
            icon, tag = "🔴", "ANOMALY   "
        elif lat > 50:
            icon, tag = "🟡", "ELEVATED  "
        elif lat > 30:
            icon, tag = "🟠", "CAUTION   "
        else:
            icon, tag = "🟢", "CLEAR     "
        lines.append(f"  {ts:>5.1f}  {icon} {tag} {route:<17}  {lat:>4.0f}ms\n")

    est = obs.get("last_state_estimation")
    if est:
        lines.append(f"\n  ◈  KIRCHHOFF ORACLE RESULT\n  {'─'*36}\n")
        if not est.get("consistent", True):
            vn = est.get("violation_node", "?")
            est_mw = est.get("estimated_true_mw", 0)
            lines.append(
                f"  ⚠  VIOLATION  ·  Node: {vn}\n"
                f"     Physics estimate : {est_mw:.1f} MW\n"
            )
        else:
            lines.append("  ✅  All nodes Kirchhoff-consistent\n")

    spoofs = obs.get("active_spoofs", [])
    if spoofs:
        lines.append(f"\n  ☠  ACTIVE SPOOFS  ·  {', '.join(spoofs)}\n")
    return "".join(lines)


def format_action_trace(actions: List[Dict]) -> str:
    hdr = (
        "╔══════════════════════════════════════════════════════╗\n"
        "║  🤖  AGENT ACTION TRACE  ·  DECISION LOG            ║\n"
        "╚══════════════════════════════════════════════════════╝\n\n"
    )
    if not actions:
        return hdr + "  ◌  No actions — reset and step the environment.\n"

    ICONS = {
        "dispatch_generation": "⚡", "toggle_circuit_breaker": "🔌",
        "run_state_estimation": "🔬", "quarantine_scada_node": "🔐",
        "inject_counter_signal": "〰", "advance_tick": "⏭",
    }
    lines = [
        hdr,
        f"  {'TK':>3}  {'ACTION':<22}  {'ΔRWD':>8}  {'SCORE':>7}\n",
        f"  {'─'*3}  {'─'*22}  {'─'*8}  {'─'*7}\n",
    ]
    for a in actions[-20:]:
        tick = a.get("tick", 0)
        act  = a.get("action", "?")
        reward = a.get("reward", "0.000")
        score  = a.get("score", "0.000")
        params = a.get("params", "{}")
        icon = ICONS.get(act, "·")
        try:
            rv = float(str(reward).replace("+", ""))
            arrow = "▲" if rv > 0 else ("▼" if rv < 0 else " ")
        except Exception:
            arrow = " "
        lines.append(
            f"  {tick:>3}  {icon} {act[:20]:<20}  "
            f"{arrow}{str(reward):>7}  {str(score):>7}\n"
        )
        if params and params not in ("{}", "null", ""):
            try:
                pd = json.loads(params)
                short = "  ".join(f"{k}={v}" for k, v in pd.items())[:46]
            except Exception:
                short = str(params)[:46]
            lines.append(f"       └─ {short}\n")
    return "".join(lines)


def format_status_bar(state) -> str:
    done_str = "✅ COMPLETE" if state.done else "▶ RUNNING"
    return (
        f"  TASK {state.task_id}  ·  TICK {state.tick:03d}  ·  "
        f"FREQ {state.frequency_hz:.3f} Hz  ·  "
        f"SCORE {state.cumulative_score:+.4f}  ·  {done_str}  "
    )


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD STATE
# ═══════════════════════════════════════════════════════════════════
class DashboardState:
    def __init__(self):
        # Pre-populate all 6 task axes at 0.0 so radar always shows full shape
        self.task_scores: Dict[int, float] = {t["id"]: 0.0 for t in TASK_META}
        self.reset()

    def reset(self):
        self.current_obs:      Dict[str, Any] = {}
        self.action_log:       List[Dict]      = []
        self.reward_history:   List[Dict]      = []
        self.packet_log:       List[Dict]      = []
        self.freq_history:     List[float]     = [60.0]
        self.frequency_hz:     float           = 60.0
        self.task_id:          int             = 0
        self.tick:             int             = 0
        self.cumulative_score: float           = 0.0
        self.done:             bool            = False
        self.env                               = None
        # NOTE: task_scores is NOT reset here — it persists across episodes
        # so the radar accumulates best scores across all tasks

    def _push_freq(self, hz: float):
        self.freq_history.append(hz)
        if len(self.freq_history) > 100:
            self.freq_history = self.freq_history[-100:]

    def init_env(self) -> bool:
        try:
            import sys, os
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
            import traceback; traceback.print_exc()
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
        od  = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
        self.current_obs = od
        self.frequency_hz = od.get("grid_frequency_hz", 60.0)
        self.tick = od.get("tick", 0)
        self.packet_log = od.get("network_packet_logs", [])
        self._push_freq(self.frequency_hz)

    def do_step(self, action_dict: Dict[str, Any]):
        if self.env is None:
            return
        import sys, os
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
        od  = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
        reward_val = od.get("reward", 0.0)
        self.current_obs  = od
        self.frequency_hz = od.get("grid_frequency_hz", 60.0)
        self.tick         = od.get("tick", 0)
        self.done         = od.get("done", False)
        self.cumulative_score += reward_val if isinstance(reward_val, (int, float)) else 0
        self._push_freq(self.frequency_hz)

        self.action_log.append({
            "tick":   self.tick,
            "action": action_dict.get("action_type", "?"),
            "params": json.dumps({k: v for k, v in action_dict.items()
                                  if k != "action_type"}),
            "reward": f"{reward_val:+.3f}" if isinstance(reward_val, (int, float)) else str(reward_val),
            "score":  f"{self.cumulative_score:.3f}",
        })
        if pkts := od.get("network_packet_logs", []):
            self.packet_log = pkts
        meta = od.get("metadata", {})
        if isinstance(meta, dict):
            if bd := meta.get("reward_breakdown", {}):
                self.reward_history.append(bd)

        # ── Radar live update every step ──────────────────────────────
        # Show live cumulative score during episode (clamped 0..1)
        live_score = max(0.0, min(1.0, self.cumulative_score))
        self.task_scores[self.task_id] = max(
            self.task_scores.get(self.task_id, 0.0),
            live_score,
        )

        # On episode end, get the authoritative graded score from env
        if self.done and self.env is not None:
            try:
                graded = self.env.get_score()
                if isinstance(graded, (int, float)):
                    self.task_scores[self.task_id] = max(
                        self.task_scores.get(self.task_id, 0.0),
                        max(0.0, min(1.0, graded)),
                    )
            except Exception:
                pass  # keep the live score as fallback


dashboard_state = DashboardState()


# ═══════════════════════════════════════════════════════════════════
# GRADIO CALLBACKS
# ═══════════════════════════════════════════════════════════════════
def on_reset(task_id: int, seed: int):
    dashboard_state.do_reset(int(task_id), int(seed))
    return _refresh()


def on_step(action_type, node_id, edge_id, mw, status, hz_offset, duration, subgraph):
    ad: Dict[str, Any] = {"action_type": action_type}
    if action_type == "dispatch_generation":
        ad["node_id"] = node_id;  ad["mw"] = float(mw)
    elif action_type == "toggle_circuit_breaker":
        ad["edge_id"] = edge_id;  ad["status"] = status
    elif action_type == "run_state_estimation":
        try:   ad["subgraph"] = json.loads(subgraph)
        except Exception:
               ad["subgraph"] = [s.strip() for s in subgraph.split(",") if s.strip()]
    elif action_type == "quarantine_scada_node":
        ad["node_id"] = node_id
    elif action_type == "inject_counter_signal":
        ad["node_id"]   = node_id
        ad["hz_offset"] = float(hz_offset)
        ad["duration"]  = int(duration)
    dashboard_state.do_step(ad)
    return _refresh()


def on_auto_run(task_id: int, seed: int, num_steps: int):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dashboard_state.do_reset(int(task_id), int(seed))
    try:
        from inference import get_fallback_action
    except ImportError:
        def get_fallback_action(tid, t):
            return ({"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 100}
                    if tid == 0 else {"action_type": "advance_tick"})
    for s in range(int(num_steps)):
        if dashboard_state.done:
            break
        dashboard_state.do_step(get_fallback_action(int(task_id), s))
    return _refresh()


def _refresh():
    obs = dashboard_state.current_obs
    return (
        create_topology_graph(obs),
        create_frequency_gauge(dashboard_state.frequency_hz, dashboard_state.freq_history),
        format_threat_feed(dashboard_state.packet_log, obs),
        format_action_trace(dashboard_state.action_log),
        create_reward_breakdown(dashboard_state.reward_history),
        create_power_flow_sankey(obs),
        create_task_radar(dashboard_state.task_scores),
        format_status_bar(dashboard_state),
    )


# ═══════════════════════════════════════════════════════════════════
# CSS — Holographic War-Room aesthetic
# ═══════════════════════════════════════════════════════════════════
CSS = f"""
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {{
  /* Override Gradio 6 built-in variables */
  --bg:       {C['void']} !important;
  --col:      {C['text']} !important;
  --bg-dark:  {C['void']} !important;
  --col-dark: {C['text']} !important;
  --body-background-fill: {C['void']} !important;
  --block-background-fill: {C['panel']} !important;
  --block-border-color: {C['border']} !important;
  --input-background-fill: {C['void']} !important;
  --input-border-color: {C['border_hi']} !important;
  --body-text-color: {C['text']} !important;
  --body-text-color-subdued: {C['text_dim']} !important;
  --block-label-text-color: {C['text_dim']} !important;
  --block-title-text-color: {C['text']} !important;
  --border-color-primary: {C['border']} !important;
  --background-fill-primary: {C['deep']} !important;
  --background-fill-secondary: {C['panel']} !important;
  --shadow-drop: none !important;
  --shadow-drop-lg: none !important;
  /* NexusGrid design tokens */
  --void:     {C['void']};
  --deep:     {C['deep']};
  --panel:    {C['panel']};
  --panel2:   {C['panel2']};
  --panel3:   {C['panel3']};
  --border:   {C['border']};
  --border-hi:{C['border_hi']};
  --teal:     {C['teal']};
  --teal-dim: {C['teal_dim']};
  --cyan:     {C['cyan']};
  --green:    {C['green']};
  --amber:    {C['amber']};
  --orange:   {C['orange']};
  --red:      {C['red']};
  --crimson:  {C['crimson']};
  --purple:   {C['purple']};
  --blue:     {C['blue']};
  --text:     {C['text']};
  --text-dim: {C['text_dim']};
  --gap:      8px;
  --radius:   8px;
}}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; }}

/* ── ROOT ── */
body, .gradio-container, .main, .wrap, .app {{
  background: var(--void) !important;
  background-color: var(--void) !important;
}}
.gradio-container {{
  background: var(--void) !important;
  background-image:
    radial-gradient(ellipse 120% 50% at 50% -8%,
      rgba(0,245,212,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 85% 100%,
      rgba(168,85,247,0.04) 0%, transparent 50%),
    repeating-linear-gradient(0deg,
      transparent, transparent 59px, rgba(0,245,212,0.018) 60px),
    repeating-linear-gradient(90deg,
      transparent, transparent 59px, rgba(0,245,212,0.018) 60px) !important;
  min-height: 100dvh;
  font-family: 'Space Mono', monospace !important;
  padding: 8px !important;
  max-width: 100% !important;
}}

/* ── Force dark everywhere ── */
.block, .form, .wrap, .contain,
.gradio-group, .gradio-accordion,
.gradio-tabitem, .tab-nav,
.gradio-container .block {{  
  background: transparent !important;
  background-color: transparent !important;
  border-color: var(--border) !important;
}}
.gradio-container .block.padded {{
  background: transparent !important;
}}
.gradio-container footer {{
  display: none !important;
}}

/* Scanline */
.gradio-container::after {{
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    to bottom, transparent 0, transparent 2px,
    rgba(0,0,0,0.04) 2px, rgba(0,0,0,0.04) 3px
  );
  pointer-events: none; z-index: 9999;
}}

/* ── HEADER ── */
.ng-header {{
  text-align: center;
  padding: 16px 10px 4px;
  position: relative;
}}
.ng-header h1 {{
  font-family: 'Orbitron', sans-serif !important;
  font-size: clamp(1rem, 4vw, 2.2rem) !important;
  font-weight: 900 !important;
  background: linear-gradient(135deg, {C['teal']} 0%, {C['cyan']} 40%, {C['blue']} 70%, {C['purple']} 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: 0.08em;
  animation: titleShift 8s ease-in-out infinite;
  background-size: 200% 100%;
}}
@keyframes titleShift {{
  0%,100% {{ background-position: 0% 50%; filter: brightness(1); }}
  50%      {{ background-position: 100% 50%; filter: brightness(1.15); }}
}}
.ng-header p {{
  font-size: clamp(0.50rem, 1.2vw, 0.70rem) !important;
  color: var(--text-dim) !important;
  letter-spacing: 0.20em;
  text-transform: uppercase;
  margin-top: 2px !important;
}}
.ng-header-line {{
  height: 1px;
  background: linear-gradient(90deg, transparent 5%, var(--teal), var(--purple), transparent 95%);
  margin: 8px auto;
  animation: lineBreath 5s ease-in-out infinite;
}}
@keyframes lineBreath {{
  0%,100% {{ width: 30%; opacity: 0.4; }}
  50%      {{ width: 75%; opacity: 1.0; }}
}}

/* ── STATUS BAR ── */
#status-bar textarea, #status-bar input {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: clamp(0.50rem, 1vw, 0.68rem) !important;
  color: var(--teal) !important;
  background: linear-gradient(90deg, var(--panel2), var(--panel3)) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: var(--radius) !important;
  text-align: center !important;
  letter-spacing: 0.05em;
  padding: 8px 10px !important;
  animation: barPulse 3s ease-in-out infinite;
}}
@keyframes barPulse {{
  0%,100% {{ box-shadow: 0 0 6px rgba(0,245,212,0.04) inset; }}
  50%      {{ box-shadow: 0 0 16px rgba(0,245,212,0.14) inset, 0 0 2px rgba(0,245,212,0.08); }}
}}

/* ── KPI STRIP ── */
.kpi-strip {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
  gap: 6px;
  margin: 6px 0;
}}
.kpi-card {{
  background: linear-gradient(160deg, var(--panel2) 0%, var(--panel) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px 6px;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s, box-shadow 0.3s, transform 0.2s;
}}
.kpi-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--teal), var(--cyan), var(--purple));
  opacity: 0.7;
}}
.kpi-card:hover {{
  border-color: var(--teal);
  box-shadow: 0 0 20px rgba(0,245,212,0.12), inset 0 0 12px rgba(0,245,212,0.04);
  transform: translateY(-1px);
}}
.kpi-label {{
  font-size: 0.52rem;
  color: var(--text-dim);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 2px;
  font-family: 'Space Mono', monospace;
}}
.kpi-value {{
  font-family: 'Orbitron', sans-serif;
  font-size: clamp(0.82rem, 1.6vw, 1.2rem);
  color: var(--teal);
  font-weight: 700;
}}

/* ── PANEL BOX ── */
.panel-box {{
  background: linear-gradient(160deg, var(--panel) 0%, rgba(10,15,30,0.95) 100%) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden;
  position: relative;
  transition: border-color 0.3s, box-shadow 0.3s;
}}
.panel-box:hover {{
  border-color: var(--border-hi) !important;
  box-shadow: 0 0 16px rgba(0,245,212,0.06), inset 0 0 8px rgba(0,245,212,0.02);
}}

/* ── PANEL LABELS ── */
.panel-label {{
  font-family: 'Orbitron', sans-serif;
  font-size: 0.56rem;
  letter-spacing: 0.20em;
  text-transform: uppercase;
  margin: 0 0 2px;
  padding-left: 3px;
}}
.panel-label-teal   {{ color: var(--teal); }}
.panel-label-cyan   {{ color: var(--cyan); }}
.panel-label-red    {{ color: var(--crimson); }}
.panel-label-amber  {{ color: var(--amber); }}
.panel-label-purple {{ color: var(--purple); }}

/* ── CTRL SECTIONS ── */
.ctrl-section {{
  background: linear-gradient(135deg, var(--panel2), var(--panel));
  border: 1px solid var(--border);
  border-left: 3px solid var(--teal);
  border-radius: var(--radius);
  padding: 8px 10px;
}}
.ctrl-section h3 {{
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.62rem !important;
  color: var(--teal) !important;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin: 0 0 6px !important;
  border: none !important;
  padding: 0 !important;
}}

/* ── LABELS ── */
.gradio-container label span,
.gradio-container .label-wrap {{
  font-family: 'Space Mono', monospace !important;
  font-size: 0.62rem !important;
  color: var(--text-dim) !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}

/* ── INPUTS ── */
.gradio-container input[type=number],
.gradio-container input[type=text],
.gradio-container select,
.gradio-container textarea {{
  background: var(--void) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 4px !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.68rem !important;
}}
.gradio-container input:focus,
.gradio-container select:focus,
.gradio-container textarea:focus {{
  border-color: var(--teal) !important;
  box-shadow: 0 0 10px rgba(0,245,212,0.18) !important;
  outline: none !important;
}}
.gradio-container input[type=range] {{ accent-color: var(--teal); }}

/* ── FEEDS ── */
.mono-feed textarea {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: clamp(0.54rem, 0.85vw, 0.66rem) !important;
  line-height: 1.65 !important;
  padding: 10px 12px !important;
  border-radius: var(--radius) !important;
}}
.threat-feed textarea {{
  color: #d1fae5 !important;
  background: linear-gradient(160deg, #040e18 0%, #080c16 60%, #0a0612 100%) !important;
  border: 1px solid rgba(239,68,68,0.25) !important;
  box-shadow: 0 0 0 1px rgba(239,68,68,0.08), inset 0 0 20px rgba(239,68,68,0.06) !important;
}}
.action-trace textarea {{
  color: #e0f2fe !important;
  background: linear-gradient(160deg, #040c16 0%, #060a16 60%, #040e1c 100%) !important;
  border: 1px solid rgba(0,245,212,0.20) !important;
  box-shadow: 0 0 0 1px rgba(0,245,212,0.06), inset 0 0 20px rgba(0,245,212,0.05) !important;
}}

/* ── BUTTONS ── */
.btn-reset, .btn-step, .btn-auto {{
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.66rem !important;
  letter-spacing: 0.12em !important;
  border-radius: 5px !important;
  transition: all 0.20s !important;
  color: #fff !important;
  cursor: pointer !important;
}}
.btn-reset {{
  background: linear-gradient(135deg, #0c4a6e, #0369a1) !important;
  border: 1px solid var(--blue) !important;
  box-shadow: 0 0 8px rgba(14,165,233,0.15) !important;
}}
.btn-reset:hover {{
  background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
  box-shadow: 0 0 20px rgba(14,165,233,0.35) !important;
  transform: translateY(-1px) !important;
}}
.btn-step {{
  background: linear-gradient(135deg, #054e3b, #065f46) !important;
  border: 1px solid var(--green) !important;
  box-shadow: 0 0 8px rgba(34,211,160,0.15) !important;
}}
.btn-step:hover {{
  background: linear-gradient(135deg, #065f46, #059669) !important;
  box-shadow: 0 0 20px rgba(34,211,160,0.35) !important;
  transform: translateY(-1px) !important;
}}
.btn-auto {{
  background: linear-gradient(135deg, #3b0764, #4c1d95) !important;
  border: 1px solid var(--purple) !important;
}}
.btn-auto:hover {{
  background: linear-gradient(135deg, #4c1d95, #6d28d9) !important;
  box-shadow: 0 0 20px rgba(168,85,247,0.30) !important;
  transform: translateY(-1px) !important;
}}

/* ── Plotly ── */
.js-plotly-plot, .plotly-graph-div {{ width: 100% !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: var(--void); }}
::-webkit-scrollbar-thumb {{ background: var(--border-hi); border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--teal-dim); }}

/* ── Dropdown ── */
.gradio-container .wrap {{
  background: var(--void) !important;
  border-color: var(--border-hi) !important;
}}

/* ── RESPONSIVE ── */
@media (max-width: 768px) {{
  .gradio-container {{ padding: 5px !important; }}
  .kpi-strip {{ grid-template-columns: repeat(3, 1fr); }}
  .kpi-value {{ font-size: 0.9rem !important; }}
}}
@media (max-width: 480px) {{
  .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }}
  .mono-feed textarea {{ font-size: 0.52rem !important; }}
}}
"""


# ═══════════════════════════════════════════════════════════════════
# IDLE CONTENT
# ═══════════════════════════════════════════════════════════════════
IDLE_THREAT = (
    "╔════════════════════════════════════════════════════════╗\n"
    "║  🔐  SCADA THREAT MONITOR  ·  KIRCHHOFF ORACLE  ║\n"
    "╚════════════════════════════════════════════════════════╝\n\n"
    "  ◌  System offline\n"
    "     Reset the environment to begin monitoring.\n"
)
IDLE_TRACE = (
    "╔══════════════════════════════════════════════════════╗\n"
    "║  🤖  AGENT ACTION TRACE  ·  DECISION LOG            ║\n"
    "╚══════════════════════════════════════════════════════╝\n\n"
    "  ◌  No actions recorded\n"
    "     Reset and step the environment.\n"
)


# ═══════════════════════════════════════════════════════════════════
# BUILD GRADIO APP
# ═══════════════════════════════════════════════════════════════════
def create_dashboard() -> gr.Blocks:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Gradio 6.x moved css/theme from constructor to launch().
        # Since we mount via mount_gradio_app(), inject CSS via HTML <style>.
        demo = gr.Blocks(
            title="⚡ NexusGrid — Cyber-Physical Grid Defense",
        )

    with demo:

        # ── INJECT CSS (Gradio 6 compatible) ─────────────────────────
        gr.HTML(f"<style>{CSS}</style>")

        # ── HEADER ──────────────────────────────────────────────────
        gr.HTML("""
        <div class="ng-header">
          <h1>⚡ NEXUSGRID — CYBERPHYS ENV</h1>
          <p>National Power Grid &amp; SCADA Cyber-Warfare Defense · OpenEnv Hackathon</p>
          <div class="ng-header-line"></div>
        </div>
        """)

        # ── STATUS BAR ──────────────────────────────────────────────
        status_bar = gr.Textbox(
            value="  TASK —  ·  TICK 000  ·  FREQ 60.000 Hz  ·  SCORE +0.0000  ·  ⏸ IDLE  ",
            label="", interactive=False, elem_id="status-bar",
        )

        # ── KPI STRIP ───────────────────────────────────────────────
        gr.HTML(f"""
        <div class="kpi-strip">
          <div class="kpi-card">
            <div class="kpi-label">Nodes</div>
            <div class="kpi-value">20</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Edges</div>
            <div class="kpi-value">40</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Attack Vectors</div>
            <div class="kpi-value" style="color:{C['crimson']}">3</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Tasks</div>
            <div class="kpi-value">6</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Physics</div>
            <div class="kpi-value" style="color:{C['cyan']}">DC</div>
          </div>
        </div>
        """)

        # ══════════════════════════════════════════════════════════
        # MAIN LAYOUT — sidebar + panels
        # ══════════════════════════════════════════════════════════
        with gr.Row(equal_height=False):

            # ── LEFT SIDEBAR ──────────────────────────────────────
            with gr.Column(scale=1, min_width=220):

                gr.HTML('<div class="ctrl-section"><h3>⬡ Reset Environment</h3></div>')
                task_select = gr.Slider(
                    0, 5, value=0, step=1,
                    label="Task ID  (0 = Smoke  →  5 = Black Start)"
                )
                seed_input = gr.Number(value=42, label="Episode Seed", precision=0)
                reset_btn  = gr.Button("🔄  INITIALISE", elem_classes=["btn-reset"])

                gr.HTML('<hr style="border-color:var(--border);margin:6px 0">')
                gr.HTML('<div class="ctrl-section"><h3>▶ Execute Action</h3></div>')

                action_type    = gr.Dropdown(
                    ["dispatch_generation", "toggle_circuit_breaker",
                     "run_state_estimation", "quarantine_scada_node",
                     "inject_counter_signal", "advance_tick"],
                    value="dispatch_generation", label="Action Type",
                )
                node_id        = gr.Textbox(value="NODE_01", label="Node ID")
                edge_id        = gr.Textbox(value="LINE_01", label="Edge ID")
                mw_val         = gr.Number(value=100, label="MW Value")
                breaker_status = gr.Dropdown(["OPEN", "CLOSED"], value="CLOSED",
                                             label="Breaker Status")
                hz_offset      = gr.Number(value=-0.5, label="Hz Offset (T4)")
                duration       = gr.Number(value=5, precision=0, label="Duration")
                subgraph       = gr.Textbox(value='["NODE_14","NODE_15"]', label="Subgraph")
                step_btn       = gr.Button("▶  STEP", elem_classes=["btn-step"])

                gr.HTML('<hr style="border-color:var(--border);margin:6px 0">')
                gr.HTML('<div class="ctrl-section"><h3>🤖 Auto-Run</h3></div>')
                auto_steps = gr.Slider(1, 50, value=10, step=1, label="Steps")
                auto_btn   = gr.Button("🤖  AUTO-RUN", elem_classes=["btn-auto"])

                # Legend
                gr.HTML(f"""
                <div style="margin-top:10px;font-family:'Space Mono',monospace;
                            font-size:0.55rem;color:var(--text-dim);line-height:1.8">
                  <div style="color:var(--teal);letter-spacing:0.12em;margin-bottom:3px;
                              font-family:Orbitron">── EDGE LOAD ──</div>
                  <span style="color:{C['teal_dim']}">━</span> &lt;50%
                  <span style="color:{C['teal']}">━</span> 50-80%
                  <span style="color:{C['amber']}">━</span> 80-95%
                  <span style="color:{C['crimson']}">━</span> ≥95%
                  <br><br>
                  <div style="color:var(--teal);letter-spacing:0.12em;margin-bottom:3px;
                              font-family:Orbitron">── FREQ BANDS ──</div>
                  <span style="color:{C['green']}">●</span> 59.7-60.3 NOM<br>
                  <span style="color:{C['amber']}">●</span> 59.5-59.7 WARN<br>
                  <span style="color:{C['orange']}">●</span> 59.2-59.5 −0.05<br>
                  <span style="color:{C['red']}">●</span> 59.0-59.2 −0.15<br>
                  <span style="color:{C['crimson']}">●</span> &lt;59.0 TERM
                </div>
                """)

            # ── RIGHT PANELS ─────────────────────────────────────
            with gr.Column(scale=3):

                # Row 1 — Topology + Frequency
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        gr.HTML('<p class="panel-label panel-label-teal">⬡ Grid Topology Map</p>')
                        topo_plot = gr.Plot(label="", elem_classes=["panel-box"])
                    with gr.Column(scale=2):
                        gr.HTML('<p class="panel-label panel-label-teal">◎ Frequency Monitor</p>')
                        freq_plot = gr.Plot(label="", elem_classes=["panel-box"])

                # Row 2 — Threat Feed + Action Trace
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.HTML('<p class="panel-label panel-label-red">🔐 SCADA Threat Feed</p>')
                        threat_feed = gr.Textbox(
                            value=IDLE_THREAT, label="", lines=14,
                            interactive=False,
                            elem_classes=["mono-feed", "threat-feed", "panel-box"],
                        )
                    with gr.Column(scale=1):
                        gr.HTML('<p class="panel-label panel-label-teal">🤖 Agent Action Trace</p>')
                        action_trace = gr.Textbox(
                            value=IDLE_TRACE, label="", lines=14,
                            interactive=False,
                            elem_classes=["mono-feed", "action-trace", "panel-box"],
                        )

                # Row 3 — Reward Breakdown + Power Flow Sankey
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.HTML('<p class="panel-label panel-label-amber">📊 Reward Accumulator</p>')
                        reward_plot = gr.Plot(label="", elem_classes=["panel-box"])
                    with gr.Column(scale=1):
                        gr.HTML('<p class="panel-label panel-label-cyan">🔄 Power Flow Sankey</p>')
                        sankey_plot = gr.Plot(label="", elem_classes=["panel-box"])

                # Row 4 — Task Radar
                gr.HTML('<p class="panel-label panel-label-purple" '
                        'style="margin-top:4px">🎯 Task Completion Radar</p>')
                radar_plot = gr.Plot(label="", elem_classes=["panel-box"])

        # ── Wire events ─────────────────────────────────────────────
        ALL_OUT = [topo_plot, freq_plot, threat_feed, action_trace,
                   reward_plot, sankey_plot, radar_plot, status_bar]

        reset_btn.click(
            fn=on_reset,
            inputs=[task_select, seed_input],
            outputs=ALL_OUT,
        )
        step_btn.click(
            fn=on_step,
            inputs=[action_type, node_id, edge_id, mw_val,
                    breaker_status, hz_offset, duration, subgraph],
            outputs=ALL_OUT,
        )
        auto_btn.click(
            fn=on_auto_run,
            inputs=[task_select, seed_input, auto_steps],
            outputs=ALL_OUT,
        )

    return demo