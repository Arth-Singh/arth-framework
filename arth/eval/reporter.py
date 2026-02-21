"""Report generation: JSON and HTML."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class Reporter:
    """Generate evaluation reports in JSON and HTML formats."""

    def generate_json(self, results: dict, output_path: Path) -> Path:
        """Save full results as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _default(obj: object) -> object:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=_default)
        return output_path

    def generate_html(self, results: dict, output_path: Path) -> Path:
        """Generate HTML report with tables and charts (inline CSS, no external deps)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = results.get("metrics", {})
        techniques = results.get("techniques", {})
        samples = results.get("samples", [])

        html = _build_html(metrics, techniques, samples)
        output_path.write_text(html, encoding="utf-8")
        return output_path


def _build_html(
    metrics: dict, techniques: dict, samples: list[dict]
) -> str:
    """Build the complete HTML report string."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Metrics table rows
    metrics_rows = ""
    for name, value in metrics.items():
        if isinstance(value, float):
            display = f"{value:.4f}"
        else:
            display = str(value)
        metrics_rows += f"<tr><td>{_esc(name)}</td><td>{_esc(display)}</td></tr>\n"

    # Technique comparison rows
    technique_rows = ""
    for tech_name, tech_data in techniques.items():
        tech_metrics = tech_data if isinstance(tech_data, dict) else {}
        asr = tech_metrics.get("attack_success_rate", 0.0)
        rr = tech_metrics.get("refusal_rate", 0.0)
        rd = tech_metrics.get("refusal_delta", 0.0)
        n = tech_metrics.get("n_samples", 0)
        bar_width = max(0, min(100, int(asr * 100)))
        technique_rows += (
            f"<tr>"
            f"<td>{_esc(tech_name)}</td>"
            f"<td>{asr:.2%}</td>"
            f"<td>{rr:.2%}</td>"
            f"<td>{rd:+.2%}</td>"
            f"<td>{int(n)}</td>"
            f'<td><div class="bar" style="width:{bar_width}%"></div></td>'
            f"</tr>\n"
        )

    # Category breakdown (group samples by category if present)
    categories: dict[str, list[dict]] = {}
    for s in samples:
        cat = s.get("category", "unknown")
        categories.setdefault(cat, []).append(s)
    category_rows = ""
    for cat, cat_samples in sorted(categories.items()):
        n_total = len(cat_samples)
        n_refused_orig = sum(
            1 for s in cat_samples
            if s.get("original_score", {}).get("refused", False)
        )
        n_refused_mod = sum(
            1 for s in cat_samples
            if s.get("modified_score", {}).get("refused", False)
        )
        category_rows += (
            f"<tr>"
            f"<td>{_esc(cat)}</td>"
            f"<td>{n_total}</td>"
            f"<td>{n_refused_orig}</td>"
            f"<td>{n_refused_mod}</td>"
            f"</tr>\n"
        )

    # Sample before/after viewer (first 20)
    sample_cards = ""
    for i, s in enumerate(samples[:20]):
        prompt = s.get("prompt", "")
        original = s.get("original", "")
        modified = s.get("modified", "")
        orig_refused = s.get("original_score", {}).get("refused", False)
        mod_refused = s.get("modified_score", {}).get("refused", False)
        orig_tag = '<span class="tag refused">REFUSED</span>' if orig_refused else '<span class="tag compliant">COMPLIANT</span>'
        mod_tag = '<span class="tag refused">REFUSED</span>' if mod_refused else '<span class="tag compliant">COMPLIANT</span>'
        sample_cards += f"""
        <div class="sample-card">
            <h4>Sample {i + 1}</h4>
            <div class="prompt"><strong>Prompt:</strong> {_esc(prompt[:200])}</div>
            <div class="comparison">
                <div class="before">
                    <h5>Original {orig_tag}</h5>
                    <pre>{_esc(original[:500])}</pre>
                </div>
                <div class="after">
                    <h5>Modified {mod_tag}</h5>
                    <pre>{_esc(modified[:500])}</pre>
                </div>
            </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arth Red Team Evaluation Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
        background:#0d1117; color:#c9d1d9; padding:2rem; line-height:1.6; }}
h1 {{ color:#58a6ff; margin-bottom:0.5rem; }}
h2 {{ color:#58a6ff; margin:2rem 0 1rem; border-bottom:1px solid #21262d; padding-bottom:0.5rem; }}
h3 {{ color:#8b949e; margin:1.5rem 0 0.5rem; }}
h4 {{ color:#c9d1d9; margin-bottom:0.5rem; }}
h5 {{ color:#8b949e; margin-bottom:0.3rem; }}
.timestamp {{ color:#8b949e; font-size:0.9rem; margin-bottom:2rem; }}
table {{ border-collapse:collapse; width:100%; margin-bottom:1.5rem; }}
th, td {{ padding:0.6rem 1rem; text-align:left; border-bottom:1px solid #21262d; }}
th {{ background:#161b22; color:#58a6ff; font-weight:600; }}
tr:hover {{ background:#161b22; }}
.bar {{ height:18px; background:linear-gradient(90deg,#f85149,#d29922,#3fb950);
        border-radius:3px; min-width:2px; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.75rem;
        font-weight:600; margin-left:0.5rem; }}
.tag.refused {{ background:#f8514922; color:#f85149; }}
.tag.compliant {{ background:#3fb95022; color:#3fb950; }}
.sample-card {{ background:#161b22; border:1px solid #21262d; border-radius:8px;
                padding:1rem; margin-bottom:1rem; }}
.prompt {{ background:#0d1117; padding:0.5rem; border-radius:4px; margin:0.5rem 0;
           font-size:0.9rem; }}
.comparison {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:0.5rem; }}
.before, .after {{ background:#0d1117; padding:0.5rem; border-radius:4px; }}
pre {{ white-space:pre-wrap; word-wrap:break-word; font-size:0.85rem;
       max-height:200px; overflow-y:auto; }}
</style>
</head>
<body>
<h1>Arth Red Team Evaluation Report</h1>
<div class="timestamp">Generated: {timestamp}</div>

<h2>Summary Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{metrics_rows}
</table>

<h2>Per-Technique Comparison</h2>
<table>
<tr><th>Technique</th><th>ASR</th><th>Refusal Rate</th><th>Delta</th><th>Samples</th><th>ASR Bar</th></tr>
{technique_rows}
</table>

<h2>Per-Category Breakdown</h2>
<table>
<tr><th>Category</th><th>Total</th><th>Refused (Original)</th><th>Refused (Modified)</th></tr>
{category_rows}
</table>

<h2>Sample Before/After Outputs</h2>
{sample_cards if sample_cards else "<p>No samples available.</p>"}

</body>
</html>"""


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
