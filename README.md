<p align="center">
  <img src="logos/lucidata_lockup_terminal_v2.svg" alt="LUCIDATA" width="700"/>
</p>

<p align="center">
  <img src="logos/lucidata_icon_terminal_v2 (1).svg" alt="LUCIDATA Icon" width="120"/>
</p>

# LUCIDATA

> Local-first AI-powered Exploratory Data Analysis (EDA) reports with privacy-preserving narrative summaries.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Phase 5 of 8](https://img.shields.io/badge/phase-5%20of%208-green.svg)](PROGRESS.md)
[![Coverage 92%](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](#)

Transform raw tabular datasets (CSV, Parquet, Excel, SQLite) into interactive EDA reports featuring plain-English AI narrative summaries — all running locally with zero raw data sent to external APIs.

## Why LUCIDATA?

- **Privacy-first** — Zero raw data leaves your machine. Only aggregated statistical metadata goes to LLMs (Ollama, OpenAI, Anthropic, etc.).
- **Local-first** — Runs entirely offline with Ollama; graceful fallback to heuristic narrative engine if LLM unavailable.
- **Fast** — Polars LazyFrame backend; sub-second DQI on 100MB datasets; correlations on 100k rows × 10 cols in < 10s.
- **Plain-English insights** — AI narrative engine translates statistics into executive summaries with actionable recommendations.
- **Open source (MIT)** — Extensible plugin architecture for analyzers, LLM providers, and exporters.

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Multi-format ingestion | ✅ Phase 2 | CSV, Parquet, Excel (.xlsx/.xls), SQLite |
| Data Quality Index (DQI) | ✅ Phase 2 | 0–100 health score, per-column audit, IQR outliers, grade bands |
| Correlation mining | ✅ Phase 3 | Pearson + Spearman, `|r| ≥ 0.35` filter, strength buckets |
| Feature driver ranking | ✅ Phase 3 | Random Forest + Mutual Info (50/50 blend), regression & classification |
| Categorical profiling | ✅ Phase 3 | Top-k values, entropy, cardinality |
| AI narrative engine | ✅ Phase 4 | LiteLLM + Ollama, structured Pydantic output, heuristic fallback |
| Interactive Visualization Engine | ✅ Phase 5 | Plotly charts: distributions, categorical bars, correlation heatmaps, feature importance |
| HTML dashboard export | ⏳ Phase 6 | Jinja2 + Plotly, dark/light mode, responsive |
| PDF report export | ⏳ Phase 6 | WeasyPrint CSS Paged Media, 2-page executive layout |
| CLI (Typer + Rich) | ⏳ Phase 7 | `lucidata analyze data.csv --target sales` |
| Streamlit Web UI | ⏳ Phase 7 | Drag-and-drop, interactive target selector |

## Installation

### Prerequisites
- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick setup
```bash
# Clone
git clone https://github.com/ibbu0101/lucidata.git
cd lucidata

# Install all dependencies (creates .venv/)
uv sync

# Optional: install pre-commit hooks
uv run pre-commit install
```

### Development install (editable)
```bash
uv pip install -e ".[dev]"
```

### Without uv (pip)
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Quick Start (Python API)

```python
import polars as pl
import lucidata as dd

# Load data
df = dd.ingest("data.csv")          # or .parquet, .xlsx, .db, or pass pl.DataFrame/pd.DataFrame

# Data Quality Index (DQI)
dqi = dd.audit(df)
print(f"Health: {dqi.health_grade} ({dqi.overall_score:.1f}/100)")
print(f"Rows: {dqi.total_rows}, Cols: {dqi.total_columns}")
print(f"Null %: {dqi.total_null_percentage:.1f}, Duplicates: {dqi.duplicate_rows_count}")

# Correlation mining
corrs = dd.correlations(df)
for c in corrs[:5]:
    print(f"  {c.feature_a} ↔ {c.feature_b}: r={c.pearson_coef:.3f} ({c.strength})")

# Feature driver importance (requires target)
drivers = dd.drivers(df, target="sales")
for d in drivers[:5]:
    print(f"  #{d.rank} {d.feature_name}: {d.importance_score:.3f} — {d.relationship_summary}")

# Categorical profiling
profiles = dd.categorical_profile(df)
for name, p in profiles.items():
    print(f"  {name}: {p.unique_count} unique, entropy={p.entropy:.2f}")
    for val, cnt in p.top_values:
        print(f"    {val}: {cnt}")

# Visualization (Phase 5)
html_dist = dd.plot_distributions(df)
html_cats = dd.plot_categorical_bars(df)
html_corr = dd.plot_correlation_heatmap(df)
html_drivers = dd.plot_feature_importance(drivers)

# Save to file
with open("distributions.html", "w") as f:
    f.write(html_dist)
```

## CLI (Phase 7 preview)

```bash
# After Phase 7 is complete
lucidata analyze data.csv --target sales --output-html report.html
lucidata analyze data.parquet --target category --output-pdf report.pdf
lucidata --help
```

## Architecture

```
INPUT (CSV/Parquet/Excel/SQLite)
       │
       ▼
┌──────────────────────┐
│  INGESTION (Polars)  │──▶ LazyFrame / DataFrame
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  AUDITOR (DQI)       │──▶ ColumnHealth, DataQualityIndex
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  STATS ENGINE        │──▶ CorrelationPair, FeatureDriver
│  - correlations()    │
│  - drivers()         │
│  - categorical_profile()
└──────────────────────┘
       │
       ├───────────────────┐
       ▼                   ▼
┌───────────────┐   ┌─────────────────┐
│ PRIVACY LAYER │   │ VIZ ENGINE      │
│ (sanitize)    │   │ (Plotly/Seaborn)│
└───────────────┘   └─────────────────┘
       │                   │
       ▼                   ▼
┌──────────────────────────────────────┐
│      AI NARRATIVE ENGINE             │
│  - LiteLLM (Ollama/OpenAI/Anthropic) │
│  - Heuristic fallback                │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│      EXPORT ENGINE                   │
│  - Jinja2 HTML dashboard             │
│  - WeasyPrint PDF                    │
└──────────────────────────────────────┘
```

See [Blueprint §3](DataDoc_AI_Master_Blueprint.md#3-high-level-architecture--data-flow) for full data flow diagram.

## Project Structure

```
lucidata/
├── .github/workflows/       # CI/CD
├── docs/                    # Architecture, usage guides
├── examples/                # Sample datasets, demo scripts
├── logos/                   # Brand assets (banner + icon)
├── src/lucidata/
│   ├── __init__.py          # Main API (ingest, audit, correlations, drivers, viz, ...)
│   ├── config.py            # Global settings
│   ├── core/                # Datatypes, schema (Pydantic), exceptions
│   ├── engine/
│   │   ├── ingest.py        # Multi-format loader (Polars-first)
│   │   ├── auditor.py       # DQI calculation
│   │   └── stats.py         # Correlations, drivers, categorical profiling
│   ├── ai/                  # LLM abstraction, privacy layer, fallback (Phase 4)
│   ├── viz/                 # Plotly chart builders (Phase 5)
│   ├── export/              # Jinja2 templates, WeasyPrint PDF (Phase 6)
│   ├── cli/                 # Typer CLI (Phase 7)
│   └── app/                 # Streamlit Web UI (Phase 7)
├── tests/                   # pytest fixtures + unit tests
├── pyproject.toml           # Project metadata, deps, tool config
├── Makefile                 # Dev automation (install, lint, test, ...)
└── PROGRESS.md              # Session log, phase status
```

## Development

```bash
# Install deps
make install          # uv sync

# Format + lint
make format           # ruff format + ruff check --fix
make lint             # ruff check only

# Test
make test             # full suite with coverage
make test-fast        # exclude @pytest.mark.slow benchmarks

# Quick validation
uv run pytest -m "not slow" -q
uv run ruff check .
uv run ruff format --check .
```

### Ruff rules
- Line length: 100
- Target: Python 3.10
- Select: E, F, W, I, UP, B, SIM, PL

## Performance (Phase 3 benchmarks)

| Operation | Dataset | Time |
|-----------|---------|------|
| `correlations()` | 100k rows × 10 numeric cols | < 10s |
| `drivers()` (RF 50 trees) | 100k rows × 10 numeric cols | < 60s |

Run benchmarks locally:
```bash
uv run pytest -m slow -v
```

## Roadmap

| Phase | Target | Status |
|-------|--------|--------|
| 1 | Environment & Scaffolding | ✅ Complete (2026-08-06) |
| 2 | Core Ingestion & Audit Engine | ✅ Complete (2026-08-07) |
| 3 | Statistical Mining Engine | ✅ Complete (2026-08-08) |
| 4 | Privacy Abstraction & AI Narrative | ✅ Complete (2026-08-09) |
| 5 | Visualization Engine | ✅ Complete (2026-08-10) |
| 6 | Export Engine (HTML/PDF) | ⏳ Next |
| 7 | Interfaces (CLI, Streamlit, Python SDK) | ⏳ Pending |
| 8 | Testing, CI/CD, PyPI Packaging | ⏳ Pending |

See [PROGRESS.md](PROGRESS.md) for detailed session log and decisions.

## Contributing

1. Fork → branch → PR
2. `make format && make lint && make test-fast` before committing
3. Add tests for new functionality
4. Follow existing code style (type hints, frozen Pydantic models, Polars-first)

## License

MIT — see [LICENSE](LICENSE).