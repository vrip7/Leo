# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
HTML report generation using Jinja2 templates.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from leo.evaluation.scorer import EvaluationReport
from leo.utils.logging import get_logger

logger = get_logger("reporting.html")

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leo Benchmark Report — {{ model_name }}</title>
    <style>
        :root {
            --bg: #0d1117;
            --card: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --warning: #d29922;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg); color: var(--text);
            line-height: 1.6; padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: var(--accent); margin-bottom: 0.5rem; font-size: 2rem; }
        h2 { color: var(--accent); margin: 2rem 0 1rem; font-size: 1.4rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
        .header {
            background: var(--card); border: 1px solid var(--border);
            border-radius: 10px; padding: 2rem; margin-bottom: 2rem;
        }
        .meta { color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem; }
        .score-card {
            display: inline-block; background: linear-gradient(135deg, #1a2332, #0d1117);
            border: 2px solid var(--accent); border-radius: 12px; padding: 1.5rem 2.5rem;
            text-align: center; margin: 1rem 0;
        }
        .score-card .label { font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; }
        .score-card .value { font-size: 3rem; font-weight: bold; color: var(--success); }
        table {
            width: 100%; border-collapse: collapse; margin: 1rem 0;
            background: var(--card); border-radius: 8px; overflow: hidden;
        }
        th {
            background: #21262d; color: var(--accent); padding: 0.8rem 1rem;
            text-align: left; font-weight: 600; font-size: 0.85rem;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        td { padding: 0.7rem 1rem; border-top: 1px solid var(--border); }
        tr:hover { background: #1c2128; }
        .metric-value { font-weight: 600; color: var(--success); font-family: 'SF Mono', monospace; }
        .stderr { color: var(--text-secondary); font-size: 0.85rem; }
        .category-badge {
            display: inline-block; padding: 0.15rem 0.6rem; border-radius: 12px;
            font-size: 0.75rem; font-weight: 500; background: #1c2128; border: 1px solid var(--border);
        }
        .perf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
        .perf-card {
            background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem;
        }
        .perf-card h3 { color: var(--accent); font-size: 1rem; margin-bottom: 0.8rem; }
        .perf-item { display: flex; justify-content: space-between; padding: 0.3rem 0; }
        .perf-label { color: var(--text-secondary); }
        .perf-value { font-family: 'SF Mono', monospace; color: var(--success); }
        .footer {
            margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
            color: var(--text-secondary); font-size: 0.85rem; text-align: center;
        }
        .footer a { color: var(--accent); text-decoration: none; }
        .bar {
            height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin-top: 0.3rem;
        }
        .bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--success)); border-radius: 3px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Leo Benchmark Report</h1>
        <p style="font-size: 1.3rem; margin: 0.5rem 0;">{{ model_name }}</p>
        <p class="meta">Generated: {{ timestamp }} | Benchmarks: {{ benchmarks|length }}</p>
        <div class="score-card">
            <div class="label">Composite Score</div>
            <div class="value">{{ "%.2f"|format(composite_score * 100) }}%</div>
        </div>
    </div>

    <h2>Benchmark Results</h2>
    <table>
        <thead>
            <tr>
                <th>Benchmark</th>
                <th>Category</th>
                <th>Metric</th>
                <th>Score</th>
                <th>Normalized</th>
                <th>Stderr</th>
                <th>Samples</th>
            </tr>
        </thead>
        <tbody>
        {% for b in benchmarks %}
            <tr>
                <td><strong>{{ b.display_name }}</strong></td>
                <td><span class="category-badge">{{ b.category }}</span></td>
                <td>{{ b.primary_metric }}</td>
                <td>
                    <span class="metric-value">{{ "%.4f"|format(b.primary_value) }}</span>
                    <div class="bar"><div class="bar-fill" style="width: {{ (b.primary_value * 100)|round }}%"></div></div>
                </td>
                <td class="metric-value">{{ "%.4f"|format(b.normalized_score) }}</td>
                <td class="stderr">{% if b.stderr %}±{{ "%.4f"|format(b.stderr) }}{% else %}—{% endif %}</td>
                <td>{{ b.samples }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>

    {% if performance %}
    <h2>Performance Metrics</h2>
    <div class="perf-grid">
        {% for section, metrics in performance.items() %}
        <div class="perf-card">
            <h3>{{ section.replace('performance:', '').replace('_', ' ').title() }}</h3>
            {% for key, value in metrics.items() %}
            <div class="perf-item">
                <span class="perf-label">{{ key.replace('_', ' ') }}</span>
                <span class="perf-value">{% if value is number %}{{ "%.2f"|format(value) }}{% else %}{{ value }}{% endif %}</span>
            </div>
            {% endfor %}
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if system_info %}
    <h2>System Information</h2>
    <div class="perf-card">
        {% for key, value in system_info.items() %}
            {% if value is mapping %}
                <div class="perf-item">
                    <span class="perf-label">{{ key.replace('_', ' ').title() }}</span>
                    <span class="perf-value">{{ value }}</span>
                </div>
            {% elif value is iterable and value is not string %}
                <div class="perf-item">
                    <span class="perf-label">{{ key.replace('_', ' ').title() }}</span>
                    <span class="perf-value">{{ value|length }} items</span>
                </div>
            {% else %}
                <div class="perf-item">
                    <span class="perf-label">{{ key.replace('_', ' ').title() }}</span>
                    <span class="perf-value">{{ value }}</span>
                </div>
            {% endif %}
        {% endfor %}
    </div>
    {% endif %}

    <div class="footer">
        <p>
            Generated by <strong>Leo</strong> — AI Model Benchmarking Platform v1.0.0<br>
            Built by <a href="https://pradyumntandon.com" target="_blank">Pradyumn Tandon</a>
            at <a href="https://vrip7.com" target="_blank">VRIP7</a>
            · <a href="https://github.com/vrip7/leo" target="_blank">GitHub</a>
        </p>
    </div>
</div>
</body>
</html>"""


class HTMLReportGenerator:
    """Generates a self-contained HTML evaluation report."""

    def __init__(self, template: str | None = None) -> None:
        self.template = Template(template or _HTML_TEMPLATE)

    def generate(self, report: EvaluationReport) -> str:
        """Render the report to an HTML string."""
        data = report.to_dict()

        # Re-create structured benchmark objects for the template
        from leo.evaluation.scorer import ScoredBenchmark

        html = self.template.render(
            model_name=report.model_name,
            composite_score=report.composite_score, 
            timestamp=report.timestamp,
            benchmarks=sorted(report.benchmarks, key=lambda b: (b.category, b.name)),
            performance=report.performance_metrics,
            system_info=report.system_info,
            config=report.config,
        )
        return html

    def save(self, report: EvaluationReport, path: str | Path) -> None:
        """Render and save the HTML report to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        html = self.generate(report)
        path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to: %s", path)
