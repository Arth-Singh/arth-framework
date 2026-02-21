"""Plotly Dash dashboard for exploring red-team evaluation results.

Professional multi-tab dashboard with dark theme, responsive layout,
and interactive exploration of technique results, category analysis,
sample comparison, model connection management, and experiment execution.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import dash
    from dash import dcc, html, dash_table, Input, Output, State, callback_context, ALL, MATCH
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px

    _HAS_DASH = True
except ImportError:
    _HAS_DASH = False


# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#21262d",
    "text": "#c9d1d9",
    "text_muted": "#8b949e",
    "primary": "#58a6ff",
    "danger": "#f85149",
    "success": "#3fb950",
    "warning": "#d29922",
    "info": "#bc8cff",
    "header_bg": "#010409",
}

FONT_FAMILY = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results(results_dir: Path) -> dict:
    """Load the most recent audit_report.json from the results directory."""
    default: dict[str, Any] = {
        "techniques": {},
        "samples": [],
        "metrics": {},
        "model_name": "N/A",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    report_path = results_dir / "audit_report.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                data = json.load(f)
                for k, v in default.items():
                    data.setdefault(k, v)
                return data
        except (json.JSONDecodeError, OSError):
            return default
    # Try to find any JSON report file
    json_files = sorted(
        results_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
                for k, v in default.items():
                    data.setdefault(k, v)
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return default


# ---------------------------------------------------------------------------
# Reusable style helpers
# ---------------------------------------------------------------------------

def _card_style(**overrides: Any) -> dict:
    base = {
        "backgroundColor": COLORS["card"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "12px",
        "padding": "1.25rem",
        "marginBottom": "1rem",
    }
    base.update(overrides)
    return base


def _section_heading(text: str) -> html.H4:
    return html.H4(
        text,
        style={
            "color": COLORS["primary"],
            "marginBottom": "1rem",
            "fontWeight": "600",
            "fontSize": "1.1rem",
            "borderBottom": f"1px solid {COLORS['border']}",
            "paddingBottom": "0.5rem",
        },
    )


# ---------------------------------------------------------------------------
# Metric card builder
# ---------------------------------------------------------------------------

def _metric_card(
    title: str,
    value: float | int | str,
    fmt: str = ".1%",
    color: str = COLORS["primary"],
    icon: str = "",
) -> dbc.Col:
    if isinstance(value, (int, float)):
        try:
            display_value = f"{value:{fmt}}"
        except (ValueError, TypeError):
            display_value = str(value)
    else:
        display_value = str(value)

    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        icon,
                        style={
                            "fontSize": "1.5rem",
                            "marginBottom": "0.25rem",
                            "opacity": "0.7",
                        },
                    ) if icon else html.Div(),
                    html.Div(
                        title,
                        style={
                            "color": COLORS["text_muted"],
                            "fontSize": "0.8rem",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.05em",
                            "fontWeight": "500",
                        },
                    ),
                    html.Div(
                        display_value,
                        style={
                            "color": color,
                            "fontSize": "1.75rem",
                            "fontWeight": "700",
                            "lineHeight": "1.2",
                            "marginTop": "0.25rem",
                        },
                    ),
                ],
                style={"textAlign": "center", "padding": "1rem 0.75rem"},
            ),
            style={
                "backgroundColor": COLORS["card"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "12px",
                "borderTop": f"3px solid {color}",
            },
        ),
        xs=6, sm=4, md=2,
        className="mb-3",
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font=dict(family=FONT_FAMILY, color=COLORS["text"], size=12),
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color=COLORS["text_muted"]),
    ),
)


def _empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_CHART_LAYOUT,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color=COLORS["text_muted"]),
            )
        ],
    )
    return fig


def _asr_bar_chart(data: dict) -> go.Figure:
    """Grouped bar chart showing ASR per technique."""
    techniques = data.get("techniques", {})
    names, asr_vals, refusal_vals = [], [], []
    for name, metrics in sorted(techniques.items()):
        if isinstance(metrics, dict) and "error" not in metrics:
            names.append(name)
            asr_vals.append(metrics.get("attack_success_rate", 0.0))
            refusal_vals.append(metrics.get("refusal_rate", 0.0))

    if not names:
        return _empty_figure("No technique data for ASR chart")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Attack Success Rate",
        x=names, y=asr_vals,
        marker_color=COLORS["danger"],
        marker_line=dict(width=0),
        opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name="Refusal Rate",
        x=names, y=refusal_vals,
        marker_color=COLORS["primary"],
        marker_line=dict(width=0),
        opacity=0.9,
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text="ASR vs Refusal Rate by Technique", font=dict(size=14)),
        barmode="group",
        yaxis=dict(tickformat=".0%", gridcolor=COLORS["border"], title="Rate"),
        xaxis=dict(title="Technique", gridcolor=COLORS["border"]),
        bargap=0.2,
        bargroupgap=0.1,
    )
    return fig


def _radar_chart(data: dict) -> go.Figure:
    """Radar/spider chart for multi-metric comparison across techniques."""
    techniques = data.get("techniques", {})
    metric_keys = ["attack_success_rate", "refusal_rate", "coherence_score"]
    metric_labels = ["ASR", "Refusal Rate", "Coherence"]

    if not techniques:
        return _empty_figure("No technique data for radar chart")

    fig = go.Figure()
    colors_cycle = [COLORS["danger"], COLORS["primary"], COLORS["success"],
                    COLORS["warning"], COLORS["info"]]

    for i, (name, metrics) in enumerate(sorted(techniques.items())):
        if not isinstance(metrics, dict) or "error" in metrics:
            continue
        values = [metrics.get(k, 0.0) for k in metric_keys]
        values.append(values[0])  # close the polygon
        labels = metric_labels + [metric_labels[0]]
        c = colors_cycle[i % len(colors_cycle)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name=name,
            line=dict(color=c, width=2),
            fillcolor=c.replace(")", ",0.1)").replace("rgb", "rgba") if c.startswith("rgb") else c + "22",
            opacity=0.8,
        ))

    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text="Multi-Metric Radar Comparison", font=dict(size=14)),
        polar=dict(
            bgcolor=COLORS["card"],
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                tickfont=dict(size=10, color=COLORS["text_muted"]),
            ),
            angularaxis=dict(
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                tickfont=dict(size=11, color=COLORS["text"]),
            ),
        ),
    )
    return fig


def _stacked_refusal_chart(data: dict) -> go.Figure:
    """Stacked bar chart: refusal rate before vs after per technique."""
    techniques = data.get("techniques", {})
    names, orig_refused, mod_refused = [], [], []
    for name, metrics in sorted(techniques.items()):
        if not isinstance(metrics, dict) or "error" in metrics:
            continue
        names.append(name)
        orig_rate = metrics.get("original_refusal_rate", metrics.get("refusal_rate", 0.0) - metrics.get("refusal_delta", 0.0))
        mod_rate = metrics.get("refusal_rate", 0.0)
        orig_refused.append(abs(orig_rate))
        mod_refused.append(abs(mod_rate))

    if not names:
        return _empty_figure("No technique data for refusal comparison")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Original Refusal Rate",
        x=names, y=orig_refused,
        marker_color=COLORS["warning"],
        opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name="Modified Refusal Rate",
        x=names, y=mod_refused,
        marker_color=COLORS["success"],
        opacity=0.9,
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text="Refusal Rate: Before vs After", font=dict(size=14)),
        barmode="group",
        yaxis=dict(tickformat=".0%", gridcolor=COLORS["border"], title="Refusal Rate"),
        xaxis=dict(title="Technique", gridcolor=COLORS["border"]),
    )
    return fig


def _category_heatmap(data: dict) -> go.Figure:
    """Heatmap of ASR by technique x category."""
    samples = data.get("samples", [])
    if not samples:
        return _empty_figure("No sample data for heatmap")

    # Build technique x category ASR matrix
    matrix: dict[str, dict[str, dict[str, int]]] = {}
    for s in samples:
        tech = s.get("technique", "unknown")
        cat = s.get("category", "unknown")
        if tech not in matrix:
            matrix[tech] = {}
        if cat not in matrix[tech]:
            matrix[tech][cat] = {"orig_refused": 0, "mod_not_refused": 0, "total": 0}
        matrix[tech][cat]["total"] += 1
        orig_refused = s.get("original_score", {}).get("refused", False)
        mod_refused = s.get("modified_score", {}).get("refused", False)
        if orig_refused:
            matrix[tech][cat]["orig_refused"] += 1
            if not mod_refused:
                matrix[tech][cat]["mod_not_refused"] += 1

    if not matrix:
        return _empty_figure("No data for heatmap")

    techniques_list = sorted(matrix.keys())
    categories_list = sorted(set(
        cat for tech_data in matrix.values() for cat in tech_data.keys()
    ))

    z = []
    text_vals = []
    for tech in techniques_list:
        row = []
        text_row = []
        for cat in categories_list:
            cell = matrix.get(tech, {}).get(cat, {"orig_refused": 0, "mod_not_refused": 0, "total": 0})
            if cell["orig_refused"] > 0:
                asr = cell["mod_not_refused"] / cell["orig_refused"]
            else:
                asr = 0.0
            row.append(asr)
            text_row.append(f"{asr:.0%}")
        z.append(row)
        text_vals.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=categories_list,
        y=techniques_list,
        colorscale=[
            [0, COLORS["success"]],
            [0.5, COLORS["warning"]],
            [1, COLORS["danger"]],
        ],
        zmin=0, zmax=1,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="Technique: %{y}<br>Category: %{x}<br>ASR: %{text}<extra></extra>",
        colorbar=dict(
            title="ASR",
            tickformat=".0%",
            tickfont=dict(color=COLORS["text_muted"]),
            titlefont=dict(color=COLORS["text"]),
        ),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text="Attack Success Rate: Technique x Category", font=dict(size=14)),
        xaxis=dict(title="Category", gridcolor=COLORS["border"]),
        yaxis=dict(title="Technique", gridcolor=COLORS["border"]),
    )
    return fig


def _category_breakdown_chart(data: dict) -> go.Figure:
    """Per-category aggregated metrics bar chart."""
    samples = data.get("samples", [])
    if not samples:
        return _empty_figure("No sample data for category breakdown")

    categories: dict[str, dict[str, int]] = {}
    for s in samples:
        cat = s.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "orig_refused": 0, "mod_refused": 0, "bypassed": 0}
        categories[cat]["total"] += 1
        orig_refused = s.get("original_score", {}).get("refused", False)
        mod_refused = s.get("modified_score", {}).get("refused", False)
        if orig_refused:
            categories[cat]["orig_refused"] += 1
        if mod_refused:
            categories[cat]["mod_refused"] += 1
        if orig_refused and not mod_refused:
            categories[cat]["bypassed"] += 1

    cat_names = sorted(categories.keys())
    asr_vals = []
    orig_rates = []
    mod_rates = []
    for cat in cat_names:
        c = categories[cat]
        asr_vals.append(c["bypassed"] / c["orig_refused"] if c["orig_refused"] > 0 else 0.0)
        orig_rates.append(c["orig_refused"] / c["total"] if c["total"] > 0 else 0.0)
        mod_rates.append(c["mod_refused"] / c["total"] if c["total"] > 0 else 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="ASR", x=cat_names, y=asr_vals, marker_color=COLORS["danger"], opacity=0.9))
    fig.add_trace(go.Bar(name="Orig Refusal", x=cat_names, y=orig_rates, marker_color=COLORS["warning"], opacity=0.9))
    fig.add_trace(go.Bar(name="Mod Refusal", x=cat_names, y=mod_rates, marker_color=COLORS["success"], opacity=0.9))
    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text="Per-Category Metrics Breakdown", font=dict(size=14)),
        barmode="group",
        yaxis=dict(tickformat=".0%", gridcolor=COLORS["border"], title="Rate"),
        xaxis=dict(title="Category", gridcolor=COLORS["border"]),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------

def _build_overview_tab(data: dict) -> dbc.Container:
    """Tab 1: Overview with hero banner, metric cards, and summary table."""
    metrics = data.get("metrics", {})
    techniques = data.get("techniques", {})
    model_name = data.get("model_name", "N/A")
    timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())

    try:
        ts_display = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        ts_display = str(timestamp)

    asr = metrics.get("attack_success_rate", 0.0)
    refusal = metrics.get("refusal_rate", 0.0)
    delta = metrics.get("refusal_delta", 0.0)
    coherence = metrics.get("coherence_score", 0.0)
    n_samples = metrics.get("n_samples", 0)
    n_techniques = len([t for t in techniques.values() if isinstance(t, dict) and "error" not in t])

    # Determine colors based on values
    asr_color = COLORS["danger"] if asr > 0.5 else COLORS["warning"] if asr > 0.2 else COLORS["success"]
    refusal_color = COLORS["success"] if refusal < 0.3 else COLORS["warning"] if refusal < 0.6 else COLORS["danger"]
    delta_color = COLORS["danger"] if delta < -0.2 else COLORS["warning"] if delta < 0 else COLORS["success"]

    # Summary table data
    table_rows = []
    for tech_name, tech_data in sorted(techniques.items()):
        if not isinstance(tech_data, dict) or "error" in tech_data:
            table_rows.append({
                "Technique": tech_name,
                "ASR": "Error",
                "Refusal Rate": "Error",
                "Refusal Delta": "Error",
                "Coherence": "Error",
                "Samples": "0",
            })
            continue
        table_rows.append({
            "Technique": tech_name,
            "ASR": f"{tech_data.get('attack_success_rate', 0.0):.1%}",
            "Refusal Rate": f"{tech_data.get('refusal_rate', 0.0):.1%}",
            "Refusal Delta": f"{tech_data.get('refusal_delta', 0.0):+.1%}",
            "Coherence": f"{tech_data.get('coherence_score', 0.0):.3f}",
            "Samples": str(int(tech_data.get("n_samples", 0))),
        })

    return dbc.Container(
        fluid=True,
        children=[
            # Hero Banner
            dbc.Card(
                dbc.CardBody(
                    dbc.Row([
                        dbc.Col([
                            html.H2(
                                "Arth Mech-Interp Toolkit",
                                style={"color": COLORS["primary"], "fontWeight": "700", "marginBottom": "0.25rem"},
                            ),
                            html.Div(
                                f"Model: {model_name}",
                                style={"color": COLORS["text"], "fontSize": "1rem", "marginBottom": "0.15rem"},
                            ),
                            html.Div(
                                f"Report generated: {ts_display}",
                                style={"color": COLORS["text_muted"], "fontSize": "0.85rem"},
                            ),
                        ], md=8),
                        dbc.Col([
                            html.Div(
                                [
                                    html.Div("Overall Status", style={
                                        "color": COLORS["text_muted"],
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.05em",
                                    }),
                                    html.Div(
                                        "HIGH RISK" if asr > 0.5 else "MODERATE" if asr > 0.2 else "LOW RISK",
                                        style={
                                            "color": asr_color,
                                            "fontSize": "1.25rem",
                                            "fontWeight": "700",
                                        },
                                    ),
                                ],
                                style={"textAlign": "right", "paddingTop": "0.5rem"},
                            ),
                        ], md=4),
                    ]),
                ),
                style={
                    "backgroundColor": COLORS["card"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "12px",
                    "borderLeft": f"4px solid {COLORS['primary']}",
                    "marginBottom": "1.5rem",
                },
            ),

            # Metric Cards Row
            dbc.Row(
                [
                    _metric_card("Attack Success Rate", asr, fmt=".1%", color=asr_color),
                    _metric_card("Refusal Rate", refusal, fmt=".1%", color=refusal_color),
                    _metric_card("Refusal Delta", delta, fmt="+.1%", color=delta_color),
                    _metric_card("Coherence", coherence, fmt=".3f", color=COLORS["info"]),
                    _metric_card("Total Samples", int(n_samples), fmt="d", color=COLORS["text"]),
                    _metric_card("Techniques Run", n_techniques, fmt="d", color=COLORS["primary"]),
                ],
                className="mb-4",
            ),

            # Summary Table
            _section_heading("Technique Results Summary"),
            dbc.Card(
                dbc.CardBody(
                    dash_table.DataTable(
                        id="overview-summary-table",
                        columns=[{"name": c, "id": c} for c in
                                 ["Technique", "ASR", "Refusal Rate", "Refusal Delta", "Coherence", "Samples"]],
                        data=table_rows if table_rows else [{"Technique": "No data", "ASR": "-", "Refusal Rate": "-",
                                                             "Refusal Delta": "-", "Coherence": "-", "Samples": "-"}],
                        sort_action="native",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": COLORS["header_bg"],
                            "color": COLORS["primary"],
                            "fontWeight": "600",
                            "border": f"1px solid {COLORS['border']}",
                            "fontSize": "0.85rem",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.03em",
                        },
                        style_cell={
                            "backgroundColor": COLORS["card"],
                            "color": COLORS["text"],
                            "border": f"1px solid {COLORS['border']}",
                            "padding": "0.6rem 1rem",
                            "fontFamily": FONT_FAMILY,
                            "fontSize": "0.9rem",
                            "textAlign": "left",
                        },
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": COLORS["bg"],
                            },
                        ],
                    ),
                ),
                style=_card_style(),
            ),
        ],
    )


def _build_technique_tab(data: dict) -> dbc.Container:
    """Tab 2: Technique Comparison with charts and data table."""
    techniques = data.get("techniques", {})

    # Build table data
    table_rows = []
    for tech_name, tech_data in sorted(techniques.items()):
        if not isinstance(tech_data, dict) or "error" in tech_data:
            continue
        table_rows.append({
            "Technique": tech_name,
            "ASR": f"{tech_data.get('attack_success_rate', 0.0):.2%}",
            "Refusal Rate": f"{tech_data.get('refusal_rate', 0.0):.2%}",
            "Refusal Delta": f"{tech_data.get('refusal_delta', 0.0):+.2%}",
            "Coherence": f"{tech_data.get('coherence_score', 0.0):.4f}",
            "Samples": str(int(tech_data.get("n_samples", 0))),
        })

    return dbc.Container(
        fluid=True,
        children=[
            # Row 1: ASR bar chart + Radar
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(dcc.Graph(
                            figure=_asr_bar_chart(data),
                            config={"displayModeBar": True, "displaylogo": False},
                            style={"height": "400px"},
                        )),
                        style=_card_style(),
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(dcc.Graph(
                            figure=_radar_chart(data),
                            config={"displayModeBar": True, "displaylogo": False},
                            style={"height": "400px"},
                        )),
                        style=_card_style(),
                    ),
                    md=6,
                ),
            ], className="mb-3"),

            # Row 2: Stacked refusal chart
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(dcc.Graph(
                            figure=_stacked_refusal_chart(data),
                            config={"displayModeBar": True, "displaylogo": False},
                            style={"height": "380px"},
                        )),
                        style=_card_style(),
                    ),
                    md=12,
                ),
            ], className="mb-3"),

            # Row 3: Sortable data table
            _section_heading("Detailed Technique Metrics"),
            dbc.Card(
                dbc.CardBody(
                    dash_table.DataTable(
                        id="technique-detail-table",
                        columns=[{"name": c, "id": c} for c in
                                 ["Technique", "ASR", "Refusal Rate", "Refusal Delta", "Coherence", "Samples"]],
                        data=table_rows if table_rows else [{"Technique": "No data", "ASR": "-",
                                                             "Refusal Rate": "-", "Refusal Delta": "-",
                                                             "Coherence": "-", "Samples": "-"}],
                        sort_action="native",
                        sort_mode="multi",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": COLORS["header_bg"],
                            "color": COLORS["primary"],
                            "fontWeight": "600",
                            "border": f"1px solid {COLORS['border']}",
                            "fontSize": "0.85rem",
                        },
                        style_cell={
                            "backgroundColor": COLORS["card"],
                            "color": COLORS["text"],
                            "border": f"1px solid {COLORS['border']}",
                            "padding": "0.6rem 1rem",
                            "fontFamily": FONT_FAMILY,
                            "fontSize": "0.9rem",
                            "textAlign": "left",
                        },
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg"]},
                        ],
                    ),
                ),
                style=_card_style(),
            ),
        ],
    )


def _build_category_tab(data: dict) -> dbc.Container:
    """Tab 3: Category Analysis with heatmap and breakdowns."""
    samples = data.get("samples", [])

    # Per-category aggregated metrics for the expandable sections
    categories: dict[str, dict[str, Any]] = {}
    for s in samples:
        cat = s.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "orig_refused": 0, "mod_refused": 0, "bypassed": 0, "techniques": set()}
        categories[cat]["total"] += 1
        orig_refused = s.get("original_score", {}).get("refused", False)
        mod_refused = s.get("modified_score", {}).get("refused", False)
        if orig_refused:
            categories[cat]["orig_refused"] += 1
        if mod_refused:
            categories[cat]["mod_refused"] += 1
        if orig_refused and not mod_refused:
            categories[cat]["bypassed"] += 1
        categories[cat]["techniques"].add(s.get("technique", "unknown"))

    # Build accordion items for each category
    accordion_items = []
    for cat_name in sorted(categories.keys()):
        c = categories[cat_name]
        asr = c["bypassed"] / c["orig_refused"] if c["orig_refused"] > 0 else 0.0
        orig_rate = c["orig_refused"] / c["total"] if c["total"] > 0 else 0.0
        mod_rate = c["mod_refused"] / c["total"] if c["total"] > 0 else 0.0
        asr_color = COLORS["danger"] if asr > 0.5 else COLORS["warning"] if asr > 0.2 else COLORS["success"]

        accordion_items.append(
            dbc.AccordionItem(
                title=f"{cat_name}  |  ASR: {asr:.1%}  |  Samples: {c['total']}",
                children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div("Total Samples", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                            html.Div(str(c["total"]), style={"color": COLORS["text"], "fontSize": "1.2rem", "fontWeight": "600"}),
                        ], md=2),
                        dbc.Col([
                            html.Div("ASR", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                            html.Div(f"{asr:.1%}", style={"color": asr_color, "fontSize": "1.2rem", "fontWeight": "600"}),
                        ], md=2),
                        dbc.Col([
                            html.Div("Orig Refusal Rate", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                            html.Div(f"{orig_rate:.1%}", style={"color": COLORS["warning"], "fontSize": "1.2rem", "fontWeight": "600"}),
                        ], md=2),
                        dbc.Col([
                            html.Div("Mod Refusal Rate", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                            html.Div(f"{mod_rate:.1%}", style={"color": COLORS["success"], "fontSize": "1.2rem", "fontWeight": "600"}),
                        ], md=2),
                        dbc.Col([
                            html.Div("Bypassed", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                            html.Div(str(c["bypassed"]), style={"color": COLORS["danger"], "fontSize": "1.2rem", "fontWeight": "600"}),
                        ], md=2),
                        dbc.Col([
                            html.Div("Techniques", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                            html.Div(", ".join(sorted(c["techniques"])), style={"color": COLORS["info"], "fontSize": "0.9rem"}),
                        ], md=2),
                    ], className="py-2"),
                ],
            )
        )

    return dbc.Container(
        fluid=True,
        children=[
            # Heatmap
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(dcc.Graph(
                            figure=_category_heatmap(data),
                            config={"displayModeBar": True, "displaylogo": False},
                            style={"height": "450px"},
                        )),
                        style=_card_style(),
                    ),
                    md=12,
                ),
            ], className="mb-3"),

            # Category breakdown chart
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(dcc.Graph(
                            figure=_category_breakdown_chart(data),
                            config={"displayModeBar": True, "displaylogo": False},
                            style={"height": "380px"},
                        )),
                        style=_card_style(),
                    ),
                    md=12,
                ),
            ], className="mb-3"),

            # Expandable category sections
            _section_heading("Per-Category Breakdown"),
            dbc.Accordion(
                accordion_items if accordion_items else [
                    dbc.AccordionItem(title="No categories available", children=[
                        html.P("No sample data available for category breakdown.",
                               style={"color": COLORS["text_muted"]}),
                    ]),
                ],
                start_collapsed=True,
                style={
                    "backgroundColor": COLORS["card"],
                    "--bs-accordion-bg": COLORS["card"],
                    "--bs-accordion-border-color": COLORS["border"],
                    "--bs-accordion-btn-color": COLORS["text"],
                    "--bs-accordion-active-bg": COLORS["bg"],
                    "--bs-accordion-active-color": COLORS["primary"],
                    "--bs-accordion-btn-bg": COLORS["card"],
                },
            ),
        ],
    )


def _build_sample_explorer_tab(data: dict) -> dbc.Container:
    """Tab 4: Sample Explorer with filters, side-by-side comparison, and pagination."""
    samples = data.get("samples", [])

    # Extract unique values for filter dropdowns
    all_techniques = sorted(set(s.get("technique", "unknown") for s in samples)) if samples else []
    all_categories = sorted(set(s.get("category", "unknown") for s in samples)) if samples else []
    all_statuses = ["All", "Refused", "Compliant", "Bypassed"]

    return dbc.Container(
        fluid=True,
        children=[
            # Filter row
            dbc.Card(
                dbc.CardBody([
                    _section_heading("Filters"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Technique", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                            dcc.Dropdown(
                                id="sample-filter-technique",
                                options=[{"label": "All Techniques", "value": "__all__"}] +
                                        [{"label": t, "value": t} for t in all_techniques],
                                value="__all__",
                                clearable=False,
                                style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                                className="dash-dropdown-dark",
                            ),
                        ], md=3),
                        dbc.Col([
                            dbc.Label("Category", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                            dcc.Dropdown(
                                id="sample-filter-category",
                                options=[{"label": "All Categories", "value": "__all__"}] +
                                        [{"label": c, "value": c} for c in all_categories],
                                value="__all__",
                                clearable=False,
                                style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                                className="dash-dropdown-dark",
                            ),
                        ], md=3),
                        dbc.Col([
                            dbc.Label("Status", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                            dcc.Dropdown(
                                id="sample-filter-status",
                                options=[{"label": s, "value": s} for s in all_statuses],
                                value="All",
                                clearable=False,
                                style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                                className="dash-dropdown-dark",
                            ),
                        ], md=3),
                        dbc.Col([
                            dbc.Label("Results", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                            html.Div(
                                id="sample-count-display",
                                style={"color": COLORS["primary"], "fontSize": "1.1rem", "fontWeight": "600",
                                       "paddingTop": "0.3rem"},
                            ),
                        ], md=3),
                    ]),
                ]),
                style=_card_style(marginBottom="1.5rem"),
            ),

            # Sample display area
            html.Div(id="sample-cards-container"),

            # Pagination
            dbc.Card(
                dbc.CardBody(
                    dbc.Row([
                        dbc.Col(
                            dbc.ButtonGroup([
                                dbc.Button(
                                    "Previous",
                                    id="sample-prev-btn",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    style={"borderColor": COLORS["border"], "color": COLORS["text"]},
                                ),
                                dbc.Button(
                                    "Next",
                                    id="sample-next-btn",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    style={"borderColor": COLORS["border"], "color": COLORS["text"]},
                                ),
                            ]),
                            md=4,
                        ),
                        dbc.Col(
                            html.Div(
                                id="sample-page-info",
                                style={"textAlign": "center", "color": COLORS["text_muted"],
                                       "paddingTop": "0.3rem", "fontSize": "0.9rem"},
                            ),
                            md=4,
                        ),
                        dbc.Col(md=4),
                    ], justify="between"),
                ),
                style=_card_style(),
            ),

            # Hidden stores for pagination state
            dcc.Store(id="sample-page-store", data=0),
            dcc.Store(id="sample-filtered-indices", data=[]),
        ],
    )


def _build_model_connection_tab() -> dbc.Container:
    """Tab 5: Model Connection management."""
    providers = [
        {"label": "TransformerLens", "value": "transformerlens"},
        {"label": "HuggingFace Local", "value": "hf_local"},
        {"label": "HuggingFace API", "value": "hf_api"},
        {"label": "OpenAI", "value": "openai"},
        {"label": "vLLM", "value": "vllm"},
    ]

    return dbc.Container(
        fluid=True,
        children=[
            dbc.Row([
                # Left column: Connection form
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            _section_heading("Model Provider Configuration"),
                            dbc.Label("Provider", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                            dcc.Dropdown(
                                id="model-provider-select",
                                options=providers,
                                value="transformerlens",
                                clearable=False,
                                style={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                                       "marginBottom": "1rem"},
                                className="dash-dropdown-dark",
                            ),

                            # Dynamic form container
                            html.Div(id="model-form-container"),

                            # Buttons
                            dbc.Row([
                                dbc.Col(
                                    dbc.Button(
                                        "Test Connection",
                                        id="model-test-btn",
                                        color="primary",
                                        className="w-100",
                                        style={"marginTop": "1rem"},
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Save Configuration",
                                        id="model-save-btn",
                                        color="success",
                                        outline=True,
                                        className="w-100",
                                        style={"marginTop": "1rem"},
                                    ),
                                    md=6,
                                ),
                            ]),

                            # Status indicator
                            html.Div(
                                id="model-connection-status",
                                style={"marginTop": "1rem"},
                            ),
                        ]),
                        style=_card_style(),
                    ),
                ], md=7),

                # Right column: Saved configurations
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            _section_heading("Saved Configurations"),
                            html.Div(id="saved-configs-list"),
                        ]),
                        style=_card_style(),
                    ),
                ], md=5),
            ]),

            # Hidden stores
            dcc.Store(id="saved-model-configs", data=[]),
        ],
    )


def _build_run_experiments_tab() -> dbc.Container:
    """Tab 6: Run Experiments interface."""
    available_techniques = [
        "refusal_direction",
        "activation_steering",
        "probing",
        "sae_features",
        "representation_engineering",
        "difference_in_means",
        "logit_attribution",
    ]
    available_datasets = [
        "harmful_harmless_pairs",
        "jailbreak_prompts",
        "over_refusal_benign",
        "category_specific",
        "custom",
    ]

    return dbc.Container(
        fluid=True,
        children=[
            dbc.Row([
                # Left column: Configuration
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            _section_heading("Experiment Configuration"),

                            # Technique selector
                            dbc.Label("Techniques", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                            dbc.Checklist(
                                id="exp-technique-select",
                                options=[{"label": f"  {t}", "value": t} for t in available_techniques],
                                value=[],
                                inline=False,
                                style={"marginBottom": "1rem"},
                                labelStyle={"color": COLORS["text"], "display": "block",
                                            "padding": "0.3rem 0", "fontSize": "0.9rem"},
                                inputStyle={"marginRight": "0.5rem"},
                            ),

                            html.Hr(style={"borderColor": COLORS["border"]}),

                            # Dataset selector
                            dbc.Label("Datasets", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                            dbc.Checklist(
                                id="exp-dataset-select",
                                options=[{"label": f"  {d}", "value": d} for d in available_datasets],
                                value=[],
                                inline=False,
                                style={"marginBottom": "1rem"},
                                labelStyle={"color": COLORS["text"], "display": "block",
                                            "padding": "0.3rem 0", "fontSize": "0.9rem"},
                                inputStyle={"marginRight": "0.5rem"},
                            ),
                        ]),
                        style=_card_style(),
                    ),
                ], md=5),

                # Right column: Parameters + Execution
                dbc.Col([
                    # Parameters card
                    dbc.Card(
                        dbc.CardBody([
                            _section_heading("Parameters"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Batch Size", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                                    dbc.Input(
                                        id="exp-batch-size",
                                        type="number",
                                        value=32,
                                        min=1, max=512, step=1,
                                        style={
                                            "backgroundColor": COLORS["bg"],
                                            "color": COLORS["text"],
                                            "border": f"1px solid {COLORS['border']}",
                                        },
                                    ),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Max Tokens", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                                    dbc.Input(
                                        id="exp-max-tokens",
                                        type="number",
                                        value=128,
                                        min=1, max=4096, step=1,
                                        style={
                                            "backgroundColor": COLORS["bg"],
                                            "color": COLORS["text"],
                                            "border": f"1px solid {COLORS['border']}",
                                        },
                                    ),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Layers", style={"color": COLORS["text_muted"], "fontSize": "0.85rem"}),
                                    dbc.Input(
                                        id="exp-layers",
                                        type="text",
                                        placeholder="e.g. 10,15,20 or all",
                                        value="all",
                                        style={
                                            "backgroundColor": COLORS["bg"],
                                            "color": COLORS["text"],
                                            "border": f"1px solid {COLORS['border']}",
                                        },
                                    ),
                                ], md=4),
                            ]),
                        ]),
                        style=_card_style(),
                    ),

                    # Execution card
                    dbc.Card(
                        dbc.CardBody([
                            _section_heading("Execution"),
                            dbc.Button(
                                "Start Extraction",
                                id="exp-start-btn",
                                color="primary",
                                size="lg",
                                className="w-100 mb-3",
                                style={"fontWeight": "600"},
                            ),

                            # Progress bar
                            html.Div(
                                id="exp-progress-container",
                                children=[
                                    dbc.Progress(
                                        id="exp-progress-bar",
                                        value=0,
                                        striped=True,
                                        animated=True,
                                        color="info",
                                        style={"height": "24px", "marginBottom": "0.5rem",
                                               "backgroundColor": COLORS["bg"]},
                                    ),
                                    html.Div(
                                        id="exp-status-text",
                                        style={"color": COLORS["text_muted"], "fontSize": "0.85rem",
                                               "textAlign": "center"},
                                        children="Ready to start.",
                                    ),
                                ],
                                style={"display": "none"},
                            ),

                            # Log output area
                            html.Div(
                                id="exp-log-container",
                                children=[
                                    html.Div(
                                        "Experiment logs will appear here...",
                                        style={"color": COLORS["text_muted"], "fontStyle": "italic"},
                                    ),
                                ],
                                style={
                                    "backgroundColor": COLORS["bg"],
                                    "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "8px",
                                    "padding": "1rem",
                                    "maxHeight": "300px",
                                    "overflowY": "auto",
                                    "fontFamily": "'SF Mono',SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace",
                                    "fontSize": "0.8rem",
                                    "marginTop": "1rem",
                                },
                            ),
                        ]),
                        style=_card_style(),
                    ),
                ], md=7),
            ]),

            # Hidden store for experiment state
            dcc.Store(id="exp-state", data={"running": False, "progress": 0, "logs": []}),
        ],
    )


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Global dark theme overrides */
body {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif !important;
}

/* Tab styling */
.nav-tabs {
    border-bottom: 1px solid #21262d !important;
}
.nav-tabs .nav-link {
    color: #8b949e !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.25rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    transition: color 0.2s ease, border-color 0.2s ease !important;
    background-color: transparent !important;
}
.nav-tabs .nav-link:hover {
    color: #c9d1d9 !important;
    border-bottom-color: #30363d !important;
}
.nav-tabs .nav-link.active {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
    background-color: transparent !important;
}

/* Dropdown dark styling */
.dash-dropdown-dark .Select-control {
    background-color: #0d1117 !important;
    border-color: #21262d !important;
    color: #c9d1d9 !important;
}
.dash-dropdown-dark .Select-menu-outer {
    background-color: #161b22 !important;
    border-color: #21262d !important;
}
.dash-dropdown-dark .Select-option {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
}
.dash-dropdown-dark .Select-option.is-focused {
    background-color: #21262d !important;
}
.dash-dropdown-dark .Select-value-label {
    color: #c9d1d9 !important;
}
.dash-dropdown-dark .Select-placeholder {
    color: #8b949e !important;
}

/* Dash dropdown v2 styling */
.Select-control, .Select--single > .Select-control {
    background-color: #0d1117 !important;
    border-color: #21262d !important;
}
.Select-menu-outer {
    background-color: #161b22 !important;
    border-color: #21262d !important;
}
.VirtualizedSelectOption {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
}
.VirtualizedSelectFocusedOption {
    background-color: #21262d !important;
}

/* Card transitions */
.card {
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.card:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}

/* Accordion dark styling */
.accordion-item {
    background-color: #161b22 !important;
    border-color: #21262d !important;
}
.accordion-button {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    box-shadow: none !important;
    font-size: 0.9rem !important;
}
.accordion-button:not(.collapsed) {
    background-color: #0d1117 !important;
    color: #58a6ff !important;
}
.accordion-button::after {
    filter: invert(0.7) !important;
}
.accordion-body {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}

/* Badge styling */
.badge-refused {
    background-color: rgba(248,81,73,0.15) !important;
    color: #f85149 !important;
    font-weight: 600 !important;
    padding: 0.3em 0.7em !important;
    border-radius: 20px !important;
    font-size: 0.75rem !important;
}
.badge-compliant {
    background-color: rgba(63,185,80,0.15) !important;
    color: #3fb950 !important;
    font-weight: 600 !important;
    padding: 0.3em 0.7em !important;
    border-radius: 20px !important;
    font-size: 0.75rem !important;
}
.badge-partial {
    background-color: rgba(210,153,34,0.15) !important;
    color: #d29922 !important;
    font-weight: 600 !important;
    padding: 0.3em 0.7em !important;
    border-radius: 20px !important;
    font-size: 0.75rem !important;
}

/* Progress bar */
.progress {
    background-color: #21262d !important;
}

/* Form controls dark */
.form-control, .form-select {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    border-color: #21262d !important;
}
.form-control:focus, .form-select:focus {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 0.2rem rgba(88,166,255,0.25) !important;
}
.form-check-input {
    background-color: #21262d !important;
    border-color: #30363d !important;
}
.form-check-input:checked {
    background-color: #58a6ff !important;
    border-color: #58a6ff !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0d1117;
}
::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #484f58;
}

/* Tab content animation */
.tab-content > .tab-pane {
    animation: fadeIn 0.3s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Button styling */
.btn-primary {
    background-color: #238636 !important;
    border-color: #238636 !important;
}
.btn-primary:hover {
    background-color: #2ea043 !important;
    border-color: #2ea043 !important;
}
.btn-outline-secondary {
    color: #c9d1d9 !important;
    border-color: #30363d !important;
}
.btn-outline-secondary:hover {
    background-color: #21262d !important;
    color: #c9d1d9 !important;
}
"""


# ---------------------------------------------------------------------------
# Main app factory
# ---------------------------------------------------------------------------

def create_app(results_dir: Path = Path("results")) -> "dash.Dash":
    """Create the Dash application for exploring results.

    Parameters
    ----------
    results_dir : Path
        Directory containing audit_report.json and/or other JSON result files.

    Returns
    -------
    dash.Dash
        The configured Dash application, ready to run.

    Raises
    ------
    ImportError
        If dash, plotly, or dash-bootstrap-components are not installed.
    """
    if not _HAS_DASH:
        raise ImportError(
            "Dashboard requires 'dash', 'plotly', and 'dash-bootstrap-components'. "
            "Install with: pip install arth-mech-interp[dashboard]"
        )

    data = _load_results(Path(results_dir))

    app = dash.Dash(
        __name__,
        title="Arth Red Team Dashboard",
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )

    # Store the data in a dcc.Store for client-side caching
    app.layout = html.Div(
        style={
            "backgroundColor": COLORS["bg"],
            "minHeight": "100vh",
            "fontFamily": FONT_FAMILY,
        },
        children=[
            # Custom CSS injection
            html.Style(CUSTOM_CSS),

            # Client-side data store
            dcc.Store(id="report-data-store", data=data),

            # Header
            html.Div(
                style={
                    "backgroundColor": COLORS["header_bg"],
                    "borderBottom": f"1px solid {COLORS['border']}",
                    "padding": "0.75rem 2rem",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                },
                children=[
                    # Logo + Title
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "0.75rem"},
                        children=[
                            # Logo placeholder (shield icon via unicode)
                            html.Div(
                                html.Span(
                                    "\u26a1",
                                    style={"fontSize": "1.5rem"},
                                ),
                                style={
                                    "width": "36px", "height": "36px",
                                    "borderRadius": "8px",
                                    "backgroundColor": COLORS["primary"] + "22",
                                    "display": "flex", "alignItems": "center",
                                    "justifyContent": "center",
                                },
                            ),
                            html.Span(
                                "Arth Red Team Dashboard",
                                style={
                                    "color": COLORS["text"],
                                    "fontSize": "1.1rem",
                                    "fontWeight": "600",
                                    "letterSpacing": "0.02em",
                                },
                            ),
                        ],
                    ),
                    # Right side: model info
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "1rem"},
                        children=[
                            html.Span(
                                f"Model: {data.get('model_name', 'N/A')}",
                                style={"color": COLORS["text_muted"], "fontSize": "0.85rem"},
                            ),
                            html.Div(
                                style={
                                    "width": "8px", "height": "8px",
                                    "borderRadius": "50%",
                                    "backgroundColor": COLORS["success"],
                                },
                            ),
                        ],
                    ),
                ],
            ),

            # Main content with tabs
            html.Div(
                style={"padding": "1.5rem 2rem"},
                children=[
                    dbc.Tabs(
                        id="main-tabs",
                        active_tab="tab-overview",
                        children=[
                            dbc.Tab(
                                label="Overview",
                                tab_id="tab-overview",
                                children=html.Div(
                                    _build_overview_tab(data),
                                    style={"paddingTop": "1.5rem"},
                                ),
                            ),
                            dbc.Tab(
                                label="Technique Comparison",
                                tab_id="tab-techniques",
                                children=html.Div(
                                    _build_technique_tab(data),
                                    style={"paddingTop": "1.5rem"},
                                ),
                            ),
                            dbc.Tab(
                                label="Category Analysis",
                                tab_id="tab-categories",
                                children=html.Div(
                                    _build_category_tab(data),
                                    style={"paddingTop": "1.5rem"},
                                ),
                            ),
                            dbc.Tab(
                                label="Sample Explorer",
                                tab_id="tab-samples",
                                children=html.Div(
                                    _build_sample_explorer_tab(data),
                                    style={"paddingTop": "1.5rem"},
                                ),
                            ),
                            dbc.Tab(
                                label="Model Connection",
                                tab_id="tab-model",
                                children=html.Div(
                                    _build_model_connection_tab(),
                                    style={"paddingTop": "1.5rem"},
                                ),
                            ),
                            dbc.Tab(
                                label="Run Experiments",
                                tab_id="tab-experiments",
                                children=html.Div(
                                    _build_run_experiments_tab(),
                                    style={"paddingTop": "1.5rem"},
                                ),
                            ),
                        ],
                    ),
                ],
            ),

            # Footer
            html.Div(
                style={
                    "borderTop": f"1px solid {COLORS['border']}",
                    "padding": "1rem 2rem",
                    "textAlign": "center",
                    "color": COLORS["text_muted"],
                    "fontSize": "0.8rem",
                },
                children=[
                    html.Span("Arth Mech-Interp Toolkit"),
                    html.Span(" | ", style={"margin": "0 0.5rem"}),
                    html.Span("Red Team Evaluation Dashboard"),
                ],
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    # --- Sample Explorer: filter and paginate ---

    @app.callback(
        [
            Output("sample-filtered-indices", "data"),
            Output("sample-page-store", "data"),
            Output("sample-count-display", "children"),
        ],
        [
            Input("sample-filter-technique", "value"),
            Input("sample-filter-category", "value"),
            Input("sample-filter-status", "value"),
        ],
        [State("report-data-store", "data")],
    )
    def filter_samples(technique: str | None, category: str | None,
                       status: str | None, report_data: dict | None):
        if not report_data:
            return [], 0, "0 results"
        samples = report_data.get("samples", [])
        if not samples:
            return [], 0, "0 results"

        filtered_indices = []
        for i, s in enumerate(samples):
            # Technique filter
            if technique and technique != "__all__":
                if s.get("technique", "unknown") != technique:
                    continue
            # Category filter
            if category and category != "__all__":
                if s.get("category", "unknown") != category:
                    continue
            # Status filter
            if status and status != "All":
                orig_refused = s.get("original_score", {}).get("refused", False)
                mod_refused = s.get("modified_score", {}).get("refused", False)
                if status == "Refused" and not mod_refused:
                    continue
                elif status == "Compliant" and mod_refused:
                    continue
                elif status == "Bypassed" and not (orig_refused and not mod_refused):
                    continue
            filtered_indices.append(i)

        count_text = f"{len(filtered_indices)} result{'s' if len(filtered_indices) != 1 else ''}"
        return filtered_indices, 0, count_text

    @app.callback(
        Output("sample-page-store", "data", allow_duplicate=True),
        [
            Input("sample-prev-btn", "n_clicks"),
            Input("sample-next-btn", "n_clicks"),
        ],
        [
            State("sample-page-store", "data"),
            State("sample-filtered-indices", "data"),
        ],
        prevent_initial_call=True,
    )
    def paginate_samples(prev_clicks: int | None, next_clicks: int | None,
                         current_page: int, filtered_indices: list):
        if not filtered_indices:
            return 0
        page_size = 10
        max_page = max(0, (len(filtered_indices) - 1) // page_size)
        ctx = callback_context
        if not ctx.triggered:
            return current_page

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "sample-prev-btn":
            return max(0, current_page - 1)
        elif trigger_id == "sample-next-btn":
            return min(max_page, current_page + 1)
        return current_page

    @app.callback(
        [
            Output("sample-cards-container", "children"),
            Output("sample-page-info", "children"),
        ],
        [
            Input("sample-page-store", "data"),
            Input("sample-filtered-indices", "data"),
        ],
        [State("report-data-store", "data")],
    )
    def render_sample_cards(page: int, filtered_indices: list, report_data: dict | None):
        if not report_data or not filtered_indices:
            empty_msg = html.Div(
                "No samples match the current filters.",
                style={
                    "color": COLORS["text_muted"],
                    "textAlign": "center",
                    "padding": "3rem",
                    "fontSize": "1rem",
                },
            )
            return empty_msg, "Page 0 of 0"

        samples = report_data.get("samples", [])
        page_size = 10
        max_page = max(0, (len(filtered_indices) - 1) // page_size)
        page = min(page, max_page)
        start = page * page_size
        end = min(start + page_size, len(filtered_indices))
        page_indices = filtered_indices[start:end]

        cards = []
        for idx in page_indices:
            if idx < 0 or idx >= len(samples):
                continue
            s = samples[idx]
            prompt = s.get("prompt", "N/A")
            original = s.get("original", "N/A")
            modified = s.get("modified", "N/A")
            technique = s.get("technique", "unknown")
            category = s.get("category", "unknown")
            orig_refused = s.get("original_score", {}).get("refused", False)
            mod_refused = s.get("modified_score", {}).get("refused", False)

            # Determine badge for modified output
            if orig_refused and not mod_refused:
                badge_class = "badge-compliant"
                badge_text = "BYPASSED"
            elif mod_refused:
                badge_class = "badge-refused"
                badge_text = "REFUSED"
            else:
                badge_class = "badge-compliant"
                badge_text = "COMPLIANT"

            # Determine original badge
            orig_badge_class = "badge-refused" if orig_refused else "badge-compliant"
            orig_badge_text = "REFUSED" if orig_refused else "COMPLIANT"

            pre_style = {
                "whiteSpace": "pre-wrap",
                "wordBreak": "break-word",
                "fontSize": "0.82rem",
                "maxHeight": "200px",
                "overflowY": "auto",
                "backgroundColor": COLORS["bg"],
                "padding": "0.75rem",
                "borderRadius": "6px",
                "border": f"1px solid {COLORS['border']}",
                "color": COLORS["text"],
                "lineHeight": "1.5",
            }

            card = dbc.Card(
                dbc.CardBody([
                    # Header row: sample number + technique + category + badges
                    dbc.Row([
                        dbc.Col([
                            html.Span(
                                f"Sample {idx + 1}",
                                style={"color": COLORS["primary"], "fontWeight": "600",
                                       "fontSize": "0.95rem"},
                            ),
                            html.Span(
                                f"  {technique}",
                                style={"color": COLORS["info"], "fontSize": "0.8rem",
                                       "marginLeft": "0.75rem"},
                            ),
                            html.Span(
                                f"  {category}",
                                style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                                       "marginLeft": "0.5rem"},
                            ),
                        ], md=8),
                        dbc.Col([
                            html.Span("Orig: ", style={"color": COLORS["text_muted"], "fontSize": "0.8rem"}),
                            html.Span(orig_badge_text, className=orig_badge_class),
                            html.Span("  Mod: ", style={"color": COLORS["text_muted"], "fontSize": "0.8rem",
                                                         "marginLeft": "0.75rem"}),
                            html.Span(badge_text, className=badge_class),
                        ], md=4, style={"textAlign": "right"}),
                    ], className="mb-2"),

                    # Prompt
                    html.Div([
                        html.Span("Prompt: ", style={"color": COLORS["text_muted"], "fontWeight": "600",
                                                      "fontSize": "0.85rem"}),
                        html.Span(prompt[:300] + ("..." if len(prompt) > 300 else ""),
                                  style={"color": COLORS["text"], "fontSize": "0.85rem"}),
                    ], style={
                        "backgroundColor": COLORS["bg"],
                        "padding": "0.6rem 0.75rem",
                        "borderRadius": "6px",
                        "border": f"1px solid {COLORS['border']}",
                        "marginBottom": "0.75rem",
                    }),

                    # Side-by-side comparison
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                [
                                    html.Span("Original Output", style={
                                        "color": COLORS["text_muted"], "fontSize": "0.8rem",
                                        "fontWeight": "600", "textTransform": "uppercase",
                                        "letterSpacing": "0.03em",
                                    }),
                                    html.Span(f"  {orig_badge_text}", className=orig_badge_class,
                                              style={"marginLeft": "0.5rem"}),
                                ],
                                style={"marginBottom": "0.5rem"},
                            ),
                            html.Pre(original[:500] + ("..." if len(original) > 500 else ""),
                                     style=pre_style),
                        ], md=6),
                        dbc.Col([
                            html.Div(
                                [
                                    html.Span("Modified Output", style={
                                        "color": COLORS["text_muted"], "fontSize": "0.8rem",
                                        "fontWeight": "600", "textTransform": "uppercase",
                                        "letterSpacing": "0.03em",
                                    }),
                                    html.Span(f"  {badge_text}", className=badge_class,
                                              style={"marginLeft": "0.5rem"}),
                                ],
                                style={"marginBottom": "0.5rem"},
                            ),
                            html.Pre(modified[:500] + ("..." if len(modified) > 500 else ""),
                                     style=pre_style),
                        ], md=6),
                    ]),
                ]),
                style={
                    "backgroundColor": COLORS["card"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "12px",
                    "marginBottom": "0.75rem",
                },
            )
            cards.append(card)

        page_info = f"Page {page + 1} of {max_page + 1}  ({len(filtered_indices)} total)"
        return html.Div(cards), page_info

    # --- Model Connection: dynamic form fields ---

    @app.callback(
        Output("model-form-container", "children"),
        Input("model-provider-select", "value"),
    )
    def update_model_form(provider: str | None):
        if not provider:
            return html.Div()

        input_style = {
            "backgroundColor": COLORS["bg"],
            "color": COLORS["text"],
            "border": f"1px solid {COLORS['border']}",
        }
        label_style = {"color": COLORS["text_muted"], "fontSize": "0.85rem", "marginTop": "0.75rem"}

        fields = []

        if provider == "transformerlens":
            fields = [
                dbc.Label("Model Name", style=label_style),
                dbc.Input(id="model-field-name", placeholder="e.g. gpt2-small", value="", style=input_style),
                dbc.Label("Device", style=label_style),
                dcc.Dropdown(
                    id="model-field-device",
                    options=[{"label": "CUDA", "value": "cuda"}, {"label": "CPU", "value": "cpu"},
                             {"label": "MPS", "value": "mps"}],
                    value="cuda", clearable=False,
                    style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                    className="dash-dropdown-dark",
                ),
                dbc.Label("Dtype", style=label_style),
                dcc.Dropdown(
                    id="model-field-dtype",
                    options=[{"label": "float16", "value": "float16"},
                             {"label": "float32", "value": "float32"},
                             {"label": "bfloat16", "value": "bfloat16"}],
                    value="float16", clearable=False,
                    style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                    className="dash-dropdown-dark",
                ),
                # Hidden fields for unused inputs
                dbc.Input(id="model-field-token", type="hidden", value=""),
                dbc.Input(id="model-field-url", type="hidden", value=""),
                dbc.Input(id="model-field-quant", type="hidden", value=""),
            ]
        elif provider == "hf_local":
            fields = [
                dbc.Label("Model Name", style=label_style),
                dbc.Input(id="model-field-name", placeholder="e.g. meta-llama/Llama-2-7b-chat-hf",
                          value="", style=input_style),
                dbc.Label("Device", style=label_style),
                dcc.Dropdown(
                    id="model-field-device",
                    options=[{"label": "CUDA", "value": "cuda"}, {"label": "CPU", "value": "cpu"},
                             {"label": "MPS", "value": "mps"}, {"label": "Auto", "value": "auto"}],
                    value="cuda", clearable=False,
                    style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                    className="dash-dropdown-dark",
                ),
                dbc.Label("Dtype", style=label_style),
                dcc.Dropdown(
                    id="model-field-dtype",
                    options=[{"label": "float16", "value": "float16"},
                             {"label": "float32", "value": "float32"},
                             {"label": "bfloat16", "value": "bfloat16"}],
                    value="float16", clearable=False,
                    style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                    className="dash-dropdown-dark",
                ),
                dbc.Label("Quantization", style=label_style),
                dcc.Dropdown(
                    id="model-field-quant",
                    options=[{"label": "None", "value": "none"},
                             {"label": "4-bit (bitsandbytes)", "value": "4bit"},
                             {"label": "8-bit (bitsandbytes)", "value": "8bit"},
                             {"label": "GPTQ", "value": "gptq"}],
                    value="none", clearable=False,
                    style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                    className="dash-dropdown-dark",
                ),
                # Hidden fields for unused inputs
                dbc.Input(id="model-field-token", type="hidden", value=""),
                dbc.Input(id="model-field-url", type="hidden", value=""),
            ]
        elif provider == "hf_api":
            fields = [
                dbc.Label("Model Name", style=label_style),
                dbc.Input(id="model-field-name", placeholder="e.g. meta-llama/Llama-2-7b-chat-hf",
                          value="", style=input_style),
                dbc.Label("API Token", style=label_style),
                dbc.Input(id="model-field-token", type="password",
                          placeholder="hf_...", value="", style=input_style),
                # Hidden fields for unused inputs
                dbc.Input(id="model-field-device", type="hidden", value=""),
                dbc.Input(id="model-field-dtype", type="hidden", value=""),
                dbc.Input(id="model-field-url", type="hidden", value=""),
                dbc.Input(id="model-field-quant", type="hidden", value=""),
            ]
        elif provider == "openai":
            fields = [
                dbc.Label("Model Name", style=label_style),
                dbc.Input(id="model-field-name", placeholder="e.g. gpt-4", value="", style=input_style),
                dbc.Label("API Key", style=label_style),
                dbc.Input(id="model-field-token", type="password",
                          placeholder="sk-...", value="", style=input_style),
                dbc.Label("Base URL (optional)", style=label_style),
                dbc.Input(id="model-field-url", placeholder="https://api.openai.com/v1",
                          value="", style=input_style),
                # Hidden fields for unused inputs
                dbc.Input(id="model-field-device", type="hidden", value=""),
                dbc.Input(id="model-field-dtype", type="hidden", value=""),
                dbc.Input(id="model-field-quant", type="hidden", value=""),
            ]
        elif provider == "vllm":
            fields = [
                dbc.Label("Model Name", style=label_style),
                dbc.Input(id="model-field-name", placeholder="e.g. meta-llama/Llama-2-7b-chat-hf",
                          value="", style=input_style),
                dbc.Label("Server URL", style=label_style),
                dbc.Input(id="model-field-url", placeholder="http://localhost:8000",
                          value="", style=input_style),
                # Hidden fields for unused inputs
                dbc.Input(id="model-field-device", type="hidden", value=""),
                dbc.Input(id="model-field-dtype", type="hidden", value=""),
                dbc.Input(id="model-field-token", type="hidden", value=""),
                dbc.Input(id="model-field-quant", type="hidden", value=""),
            ]
        else:
            fields = [html.P("Select a provider", style={"color": COLORS["text_muted"]})]

        return html.Div(fields)

    # --- Model Connection: test connection ---

    @app.callback(
        Output("model-connection-status", "children"),
        Input("model-test-btn", "n_clicks"),
        [
            State("model-provider-select", "value"),
            State("model-field-name", "value"),
        ],
        prevent_initial_call=True,
    )
    def test_model_connection(n_clicks: int | None, provider: str | None,
                              model_name: str | None):
        if not n_clicks or not provider:
            return html.Div()

        if not model_name or not model_name.strip():
            return dbc.Alert(
                [
                    html.Span(
                        "\u2716 ",
                        style={"fontWeight": "bold", "marginRight": "0.5rem"},
                    ),
                    "Model name is required.",
                ],
                color="danger",
                style={"fontSize": "0.9rem"},
            )

        # Simulate connection test (in production, this would actually attempt connection)
        return dbc.Alert(
            [
                html.Span(
                    "\u2714 ",
                    style={"fontWeight": "bold", "marginRight": "0.5rem"},
                ),
                f"Configuration validated for {provider}: {model_name}. "
                f"Actual connection test requires a running backend.",
            ],
            color="info",
            style={"fontSize": "0.9rem"},
        )

    # --- Model Connection: save config ---

    @app.callback(
        [
            Output("saved-model-configs", "data"),
            Output("saved-configs-list", "children"),
        ],
        Input("model-save-btn", "n_clicks"),
        [
            State("model-provider-select", "value"),
            State("model-field-name", "value"),
            State("model-field-device", "value"),
            State("model-field-dtype", "value"),
            State("model-field-token", "value"),
            State("model-field-url", "value"),
            State("saved-model-configs", "data"),
        ],
        prevent_initial_call=True,
    )
    def save_model_config(n_clicks: int | None, provider: str | None,
                          name: str | None, device: str | None,
                          dtype: str | None, token: str | None,
                          url: str | None, existing_configs: list | None):
        if not n_clicks or not provider or not name:
            configs = existing_configs or []
            return configs, _render_saved_configs(configs)

        configs = existing_configs or []
        config_entry = {
            "id": str(uuid.uuid4())[:8],
            "provider": provider,
            "name": name or "",
            "device": device or "",
            "dtype": dtype or "",
            "url": url or "",
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        }
        configs.append(config_entry)
        return configs, _render_saved_configs(configs)

    def _render_saved_configs(configs: list) -> html.Div:
        if not configs:
            return html.Div(
                "No saved configurations yet.",
                style={"color": COLORS["text_muted"], "fontStyle": "italic", "padding": "1rem"},
            )
        items = []
        for cfg in configs:
            items.append(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                            children=[
                                html.Div([
                                    html.Span(
                                        cfg.get("provider", "unknown").upper(),
                                        style={
                                            "color": COLORS["info"],
                                            "fontSize": "0.7rem",
                                            "fontWeight": "600",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "0.05em",
                                            "backgroundColor": COLORS["info"] + "22",
                                            "padding": "0.15rem 0.5rem",
                                            "borderRadius": "4px",
                                        },
                                    ),
                                    html.Span(
                                        f"  {cfg.get('name', 'N/A')}",
                                        style={"color": COLORS["text"], "fontSize": "0.9rem",
                                               "fontWeight": "500", "marginLeft": "0.5rem"},
                                    ),
                                ]),
                                html.Span(
                                    cfg.get("saved_at", ""),
                                    style={"color": COLORS["text_muted"], "fontSize": "0.75rem"},
                                ),
                            ],
                        ),
                        html.Div(
                            f"Device: {cfg.get('device', 'N/A')}  |  Dtype: {cfg.get('dtype', 'N/A')}",
                            style={"color": COLORS["text_muted"], "fontSize": "0.8rem", "marginTop": "0.25rem"},
                        ) if cfg.get("device") else html.Div(),
                    ], style={"padding": "0.5rem 0.75rem"}),
                    style={
                        "backgroundColor": COLORS["bg"],
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "8px",
                        "marginBottom": "0.5rem",
                    },
                )
            )
        return html.Div(items)

    # --- Run Experiments: start extraction ---

    @app.callback(
        [
            Output("exp-progress-container", "style"),
            Output("exp-progress-bar", "value"),
            Output("exp-status-text", "children"),
            Output("exp-log-container", "children"),
        ],
        Input("exp-start-btn", "n_clicks"),
        [
            State("exp-technique-select", "value"),
            State("exp-dataset-select", "value"),
            State("exp-batch-size", "value"),
            State("exp-max-tokens", "value"),
            State("exp-layers", "value"),
        ],
        prevent_initial_call=True,
    )
    def start_experiment(n_clicks: int | None, techniques: list | None,
                         datasets: list | None, batch_size: int | None,
                         max_tokens: int | None, layers: str | None):
        if not n_clicks:
            return {"display": "none"}, 0, "Ready to start.", [
                html.Div("Experiment logs will appear here...",
                         style={"color": COLORS["text_muted"], "fontStyle": "italic"})
            ]

        techniques = techniques or []
        datasets = datasets or []

        if not techniques:
            return {"display": "block"}, 0, "Error: No techniques selected.", [
                html.Div(
                    "[ERROR] Please select at least one technique before starting.",
                    style={"color": COLORS["danger"]},
                ),
            ]

        if not datasets:
            return {"display": "block"}, 0, "Error: No datasets selected.", [
                html.Div(
                    "[ERROR] Please select at least one dataset before starting.",
                    style={"color": COLORS["danger"]},
                ),
            ]

        # Parse layers
        layers_str = layers or "all"
        if layers_str.strip().lower() == "all":
            layers_display = "all layers"
        else:
            layers_display = f"layers [{layers_str}]"

        # Generate log entries simulating experiment setup
        log_entries = [
            html.Div(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Experiment configuration validated",
                style={"color": COLORS["success"], "marginBottom": "0.25rem"},
            ),
            html.Div(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Techniques: {', '.join(techniques)}",
                style={"color": COLORS["text"], "marginBottom": "0.25rem"},
            ),
            html.Div(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Datasets: {', '.join(datasets)}",
                style={"color": COLORS["text"], "marginBottom": "0.25rem"},
            ),
            html.Div(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                f"Parameters: batch_size={batch_size}, max_tokens={max_tokens}, {layers_display}",
                style={"color": COLORS["text"], "marginBottom": "0.25rem"},
            ),
            html.Div(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                f"Queued {len(techniques)} technique(s) x {len(datasets)} dataset(s) = "
                f"{len(techniques) * len(datasets)} job(s)",
                style={"color": COLORS["info"], "marginBottom": "0.25rem"},
            ),
            html.Div(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                f"To run experiments, use the CLI: arth run --config <your-config.yaml>",
                style={"color": COLORS["warning"], "marginBottom": "0.25rem"},
            ),
        ]

        status_text = (
            f"Configured: {len(techniques)} techniques, {len(datasets)} datasets. "
            f"Use CLI to execute."
        )

        return {"display": "block"}, 100, status_text, log_entries

    return app
