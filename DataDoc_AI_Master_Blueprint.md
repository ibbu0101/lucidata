# DataDoc-AI: Production Engineering Specification & AI Coding Agent Blueprint
> **Document Version:** 2.0 (Enhanced Architectural Blueprint)  
> **Project Name:** DataDoc-AI  
> **Target Audience:** AI Coding Agents (Cursor, Claude Dev, Roo Code, Aider, GitHub Copilot) & Lead Developers  
> **License:** MIT (Open-Source)  

---

## 1. Executive Summary & Vision

### 1.1 Vision Statement
**DataDoc-AI** is an open-source, local-first Python library, CLI, and interactive web application designed to transform raw tabular datasets (`CSV`, `Parquet`, `Excel`, `SQLite`) into interactive Exploratory Data Analysis (EDA) reports featuring plain-English AI narrative summaries in seconds.

### 1.2 Core Problem Statement
* **Time Inefficiency:** Data analysts and data scientists spend 60–80% of their time conducting repetitive data auditing, calculating summary statistics, plotting distribution charts, and translating technical metrics into executive summaries for business stakeholders.
* **Privacy & Compliance Risks:** Cloud-first automated EDA tools require sending sensitive enterprise datasets to third-party APIs (OpenAI, Anthropic), violating GDPR, HIPAA, and corporate data governance policies.
* **Static & Non-Interactive Outputs:** Legacy tools like `pandas-profiling` / `ydata-profiling` generate heavy, static, or uninterpreted HTML files that lack actionable narrative insights and deep business context.

### 1.3 Target Personas
1. **Data Analyst / Scientist:** Wants instant statistical audits, correlation matrices, feature driver rankings, and baseline report generation without boilerplate code.
2. **Product / Business Manager:** Needs non-technical, high-level summaries with actionable takeaways, operational highlights, and clean PDF exports.
3. **Data Engineering Lead / CISO:** Requires strict local execution, zero-raw-data cloud egress, lightweight memory footprint, and high-throughput processing.

### 1.4 Architecture Improvements over Initial Draft
* **Polars Lazy Evaluation:** Replaced eager computations with Polars `LazyFrame` processing pipelines to lower memory usage and process datasets exceeding RAM limits.
* **Privacy Abstraction Layer:** Guarantees that **zero raw row data** is sent to LLM providers. Only serialized, sanitized statistical JSON metadata is passed to the AI Storytelling Engine.
* **Structured AI Outputs via Pydantic:** Replaced unconstrained LLM text generation with strictly validated JSON responses via Pydantic V2 and `instructor` / `LiteLLM`.
* **Heuristic Narrative Fallback Engine:** If an LLM provider (e.g., local Ollama) is offline or unavailable, the system automatically degrades gracefully to a rule-based statistical narrative generator without failing.
* **Plugin Architecture for Analysis & Export:** Implemented abstract base classes for `BaseAnalyzer`, `BaseLLMProvider`, and `BaseExporter` to allow third-party extensions.

---

## 2. Tech Stack & Dependency Matrix

| Category | Technology | Purpose & Selection Justification |
| :--- | :--- | :--- |
| **Core Language** | Python 3.10+ | Enables modern type hinting (`type | None`), pattern matching, and compatibility with modern async features. |
| **Data Processing** | Polars (Primary) | High-performance Rust-backed DataFrame library with SIMD vectorization and lazy execution. |
| **Data Fallback** | Pandas + PyArrow | Fallback engine for edge-case file formats and specific pandas-native legacy APIs. |
| **Validation** | Pandera + Pydantic V2 | Type enforcement and runtime schema validation for dataframes and JSON payloads. |
| **Statistical Engine**| SciPy + Scikit-Learn | Correlation metrics (Pearson/Spearman), Mutual Information, Random Forest feature importance. |
| **Viz Engine** | Plotly + Seaborn | Plotly for responsive, interactive HTML charts; Seaborn/Matplotlib for static high-res PDF rendering. |
| **AI Orchestration** | LiteLLM + Ollama | Unified abstraction for local LLMs (Ollama/llama3/mistral) and remote cloud APIs with retries. |
| **Structured Output**| Instructor / Pydantic | Guarantees LLM responses conform to defined Pydantic schemas. |
| **Templating & Export**| Jinja2 + WeasyPrint | Jinja2 HTML layout engine with WeasyPrint CSS-paged media renderer for publication-ready PDFs. |
| **CLI Framework** | Typer + Rich | Modern CLI with auto-completion, rich terminal color formatting, spinners, and progress bars. |
| **Web UI** | Streamlit / FastAPI | Drag-and-drop web dashboard for interactive browser-based analysis. |
| **Testing & CI** | PyTest + Coverage + Ruff | Rigorous unit testing, benchmark suites, and lightning-fast linting/formatting. |

---

## 3. High-Level Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             INPUT DATA ENGINE                                    │
│       [CSV]  /  [Parquet]  /  [Excel]  /  [SQLite]  /  [Polars/Pandas DF]       │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 1: DATA AUDIT & INGESTION PIPELINE                      │
│ - Ingestion & Schema Profiling (Type Inference: Numeric, Categorical, Datetime)  │
│ - Quality Health Index (DQI) Calculation (Nulls, Duplicates, IQR Outliers, Zero-Var)│
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                MODULE 2: STATISTICAL & FEATURE MINING ENGINE                     │
│ - Correlation Matrix (Pearson & Spearman)                                        │
│ - Target Feature Driver Importance (Random Forest Gini / Mutual Info Scoring)    │
│ - Categorical Cardinality & Value Distribution Profiling                         │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
                   ▼                                           ▼
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│ MODULE 3: PRIVACY ABSTRACTION LAYER │     │    MODULE 4: VISUALIZATION ENGINE    │
│ - Summarize stats into JSON Schema  │     │ - Auto-generate interactive Plotly │
│ - Strip ALL raw rows & PII tokens   │     │   charts (Distributions, Heatmaps)  │
└──────────────────┬──────────────────┘     └──────────────────┬──────────────────┘
                   │                                           │
                   ▼                                           │
┌─────────────────────────────────────┐                        │
│ MODULE 5: AI STORYTELLING ENGINE    │                        │
│ - Local LLM (Ollama) / Cloud API    │                        │
│ - Offline Rule-Based Fallback Engine│                        │
└──────────────────┬──────────────────┘                        │
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 6: REPORT & EXPORT GENERATOR ENGINE                     │
│ - Jinja2 Responsive Dashboard Rendering (.html)                                  │
│ - WeasyPrint PDF Rendering with CSS Paged Media (.pdf)                           │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT ARTIFACTS                                    │
│     [Standalone HTML Dashboard]   /   [2-Page Executive PDF Report]               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Repository Directory Structure Blueprint

To guide AI coding agents, the codebase MUST strictly follow this layout:

```text
datadoc-ai/
├── .github/
│   └── workflows/
│       ├── ci.yml                # Automated tests, linting, and coverage
│       └── publish.yml           # PyPI release pipeline
├── docs/
│   ├── ARCHITECTURE.md          # Technical architecture overview
│   └── USAGE.md                 # User guide & CLI commands
├── examples/
│   ├── sample_sales.csv         # Benchmark sample dataset
│   └── analyze_demo.py          # Demo script showing Python API usage
├── src/
│   └── datadoc/
│       ├── __init__.py          # Main library interface exposing analyze()
│       ├── py.typed              # PEP 561 type marker
│       ├── config.py            # Global settings & configuration models
│       ├── core/
│       │   ├── __init__.py
│       │   ├── datatypes.py     # Custom type definitions and enums
│       │   ├── exceptions.py    # Custom domain exceptions
│       │   └── schema.py        # Pydantic data models for audit results
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── ingest.py        # High-speed Polars ingestion layer
│       │   ├── auditor.py       # Data Quality Index (DQI) & health check
│       │   └── stats.py         # Correlations & Feature Driver Importance
│       ├── ai/
│       │   ├── __init__.py
│       │   ├── abstraction.py   # Privacy layer (strips raw data -> JSON metadata)
│       │   ├── llm_client.py    # LiteLLM/Ollama client wrapper
│       │   ├── prompts.py       # Prompt templates & context formatting
│       │   └── fallback.py      # Rule-based heuristic generator (offline mode)
│       ├── viz/
│       │   ├── __init__.py
│       │   ├── plots.py         # Plotly chart builders (distributions, heatmaps)
│       │   └── themes.py        # Color palettes & CSS theme definitions
│       ├── export/
│       │   ├── __init__.py
│       │   ├── renderer.py      # Jinja2 template rendering logic
│       │   ├── pdf.py           # WeasyPrint PDF compilation engine
│       │   └── templates/
│       │       ├── dashboard.html.j2 # Interactive HTML dashboard template
│       │       ├── report.html.j2    # Printable HTML layout for PDF
│       │       └── style.css         # CSS for WeasyPrint & HTML styling
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py          # Typer CLI application implementation
│       └── app/
│           ├── __init__.py
│           └── streamlit_app.py # Streamlit Web UI implementation
├── tests/
│   ├── conftest.py              # Shared PyTest fixtures
│   ├── test_ingest.py           # Unit tests for data ingestion
│   ├── test_auditor.py          # Unit tests for statistical auditing
│   ├── test_stats.py            # Unit tests for correlation & drivers
│   ├── test_ai.py               # Unit tests for AI abstraction & fallback
│   ├── test_export.py           # End-to-end report generation tests
│   └── test_cli.py              # Unit tests for Typer CLI commands
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml               # Poetry/uv project specifications
└── Makefile                     # Developer automation commands
```

---

## 5. Core Data Schemas & Pydantic Specifications

The entire internal pipeline relies on strong runtime validation using Pydantic V2. AI Coding agents MUST implement these contracts directly.

### 5.1 Quality Audit Metrics (`datadoc.core.schema`)

```python
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

class DataType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"

class ColumnHealth(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    name: str
    data_type: DataType
    total_count: int
    null_count: int
    null_percentage: float = Field(ge=0.0, le=100.0)
    unique_count: int
    is_constant: bool
    iqr_outliers_count: int = 0
    sample_values: List[Any] = Field(default_factory=list)

class DataQualityIndex(BaseModel):
    overall_score: float = Field(ge=0.0, le=100.0, description="0-100 overall health score")
    total_rows: int
    total_columns: int
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    total_null_cells: int
    total_null_percentage: float
    health_grade: str = Field(description="A, B, C, D, or F based on overall score")
    column_health: Dict[str, ColumnHealth]

class CorrelationPair(BaseModel):
    feature_a: str
    feature_b: str
    pearson_coef: float
    spearman_coef: float
    strength: str = Field(description="Weak, Moderate, Strong, Very Strong")

class FeatureDriver(BaseModel):
    feature_name: str
    importance_score: float = Field(ge=0.0, le=1.0)
    rank: int
    relationship_summary: str

class ExecutiveSummaryNarrative(BaseModel):
    headline: str = Field(description="High-level title summarizing the dataset narrative")
    key_highlights: List[str] = Field(description="3-5 critical insights derived from statistical findings")
    data_anomalies: List[str] = Field(description="Noteworthy quality warnings, outliers, or missingness patterns")
    actionable_next_steps: List[str] = Field(description="Concrete operational or analytical recommendations")

class FullReportData(BaseModel):
    metadata: Dict[str, Any]
    dqi: DataQualityIndex
    correlations: List[CorrelationPair]
    feature_drivers: Optional[List[FeatureDriver]] = None
    narrative: ExecutiveSummaryNarrative
```

---

## 6. Functional Requirements & Module Specifications

### Module 1: High-Speed Ingestion & Quality Audit Engine (`datadoc.engine`)

#### Functional Requirements:
1. **FR-1.1 Multi-Format Ingestion:** Support `.csv`, `.parquet`, `.xlsx`, and `.db`/`.sqlite` inputs. Ingest a 100MB dataset in under 3 seconds using Polars.
2. **FR-1.2 Data Quality Index (DQI) Calculation:** Compute an aggregate health score (0–100%) calculated as:
   $$	ext{DQI} = 100 - \left( w_1 \cdot \%_{	ext{nulls}} + w_2 \cdot \%_{	ext{duplicates}} + w_3 \cdot \%_{	ext{outliers}} + w_4 \cdot \%_{	ext{constant\_cols}} ight)$$
   Where default weights are $w_1 = 0.4$, $w_2 = 0.3$, $w_3 = 0.2$, $w_4 = 0.1$.
3. **FR-1.3 Outlier Detection:** Use Interquartile Range (IQR) logic:
   $$	ext{Outlier Bounds} = [Q_1 - 1.5 	imes 	ext{IQR}, Q_3 + 1.5 	imes 	ext{IQR}]$$
4. **FR-1.4 Type Inference:** Automatically detect types including Datetime strings, categorical cardinality thresholds, and continuous numerical features.

---

### Module 2: Statistical Insights & Driver Mining (`datadoc.engine.stats`)

#### Functional Requirements:
1. **FR-2.1 Correlation Mining:** Compute Pearson ($r$) and Spearman ($ho$) coefficients for all numerical feature pairs. Filter and surface top correlations where $|r| \ge 0.35$.
2. **FR-2.2 Feature Driver Ranking:** When a target variable is specified:
   * For continuous targets: Train a `RandomForestRegressor` and compute `Mutual Information` regression scores.
   * For categorical targets: Train a `RandomForestClassifier` and compute `Mutual Information` classification scores.
   * Normalize feature importance scores to sum to 1.0.
3. **FR-2.3 Categorical Value Profiling:** For categorical columns, compute frequency counts, top 5 value distributions, and entropy-based cardinality indices.

---

### Module 3: Privacy Abstraction Layer (`datadoc.ai.abstraction`)

#### Functional Requirements:
1. **FR-3.1 Complete Data Anonymization:** Under NO circumstance should raw row data, user PII, or raw dataframe strings be sent to the LLM.
2. **FR-3.2 Metadata Serialization:** Construct a lightweight JSON object containing only aggregated statistics:
   * Dataset shape (rows, columns).
   * Column names and inferred data types.
   * Summary stats (Mean, Std, Min, Max, Null % per column).
   * Top 5 highest correlated column pairs.
   * Top 5 feature driver rankings (if target variable is defined).
   * DQI aggregate score and anomaly flags.

---

### Module 4: AI Storytelling Engine & Provider Abstraction (`datadoc.ai`)

#### Functional Requirements:
1. **FR-4.1 LiteLLM Integration:** Provide seamless unified connectivity to:
   * **Local Providers:** Ollama (`llama3`, `mistral`, `phi3`), LocalAI, vLLM.
   * **Cloud Providers:** OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-5-sonnet`), Google (`gemini-1.5-flash`).
2. **FR-4.2 Structured Pydantic Output Enforcement:** Use structured prompting / `instructor` to enforce output conformity to the `ExecutiveSummaryNarrative` schema.
3. **FR-4.3 Heuristic Rule-Based Fallback:** If the LLM call times out (default timeout: 10s) or fails, the engine MUST fallback to `datadoc.ai.fallback.generate_heuristic_narrative()` without throwing an unhandled exception.

#### Prompt Engineering Specification (`datadoc/ai/prompts.py`):
```text
SYSTEM PROMPT:
You are an expert Chief Data Scientist and Business Intelligence Executive.
Your task is to analyze statistical metadata from a data health audit report and write a clear, concise, plain-English narrative summary for business stakeholders.

STRICT CONSTRAINTS:
1. Base your commentary ONLY on the provided JSON statistical metadata. Do NOT hallucinate data points.
2. Provide concrete operational insights.
3. Keep tone professional, authoritative, and actionable.
4. Respond ONLY with valid JSON matching the required schema.

METADATA INPUT:
{json_metadata_payload}
```

---

### Module 5: Visualization Engine (`datadoc.viz`)

#### Functional Requirements:
1. **FR-5.1 Auto-Generated Chart Gallery:**
   * **Distribution Plots:** Histograms with KDE overlays for continuous numerical columns.
   * **Categorical Bar Charts:** Horizontal bar charts of top 10 categories.
   * **Correlation Heatmap:** Interactive Plotly heatmap for numerical feature pairs.
   * **Feature Importance Chart:** Horizontal rank chart highlighting driver influence on target variable.
2. **FR-5.2 Standalone HTML Embeds:** Render charts as interactive, standalone JavaScript Plotly objects embedded directly into Jinja2 templates without external CDN dependencies where possible.

---

### Module 6: Jinja2 & WeasyPrint Export Engine (`datadoc.export`)

#### Functional Requirements:
1. **FR-6.1 Interactive HTML Dashboard:** Modern single-page web dashboard featuring dark/light mode, filterable statistical tables, embedded Plotly charts, and styled executive summary cards.
2. **FR-6.2 Publication-Ready PDF Report:**
   * Styled via CSS Paged Media (`@page` rules).
   * Compact 2-page layout.
   * Pure CSS typography, no broken layout margins, zero page-break clipping on headings.

---

## 7. Phased Implementation Roadmap for AI Coding Agents

AI Coding Agents (e.g., Cursor / Roo Code) should execute this project sequentially across **8 Granular Execution Phases**.

```
[Phase 1: Environment & Scaffolding] ──► [Phase 2: Core Ingestion Engine]
                                                  │
                                                  ▼
[Phase 4: AI Abstraction & Fallback]  ◄── [Phase 3: Statistical Mining]
          │
          ▼
[Phase 5: Viz Engine] ──► [Phase 6: Export Engine] ──► [Phase 7: Interfaces] ──► [Phase 8: Test & Package]
```

---

### Phase 1: Environment Setup & Project Scaffolding
**Goal:** Initialize project structure, set up package manager (`uv` or `poetry`), define dependencies in `pyproject.toml`, and set up PyTest configuration.

* **Task 1.1:** Create folder hierarchy as outlined in Section 4.
* **Task 1.2:** Draft `pyproject.toml` with dependencies: `polars`, `pandas`, `scipy`, `scikit-learn`, `pydantic`, `litellm`, `jinja2`, `weasyprint`, `plotly`, `typer`, `rich`, `streamlit`.
* **Task 1.3:** Create `datadoc/core/exceptions.py` defining standard exceptions (`DataDocError`, `IngestionError`, `LLMProviderError`).
* **Task 1.4:** Verify installation and setup standard pre-commit linting using `ruff`.

---

### Phase 2: Core Data Ingestion & Audit Engine
**Goal:** Implement fast data loading, type inference, and Data Quality Index (DQI) calculation using Polars.

* **Task 2.1:** Implement `datadoc/engine/ingest.py` supporting `.csv`, `.parquet`, `.xlsx`, `.db`. Ensure Polars is used natively with fallback to Pandas.
* **Task 2.2:** Implement `datadoc/engine/auditor.py` to calculate missing value counts, distinct counts, zero-variance columns, and IQR outliers.
* **Task 2.3:** Calculate composite `DataQualityIndex` score and return validated `DataQualityIndex` Pydantic objects.
* **Task 2.4:** Write comprehensive unit tests in `tests/test_auditor.py` checking edge cases (empty dataframes, single-column datasets, all-null columns).

---

### Phase 3: Statistical Relationships & Driver Mining Engine
**Goal:** Build statistical computation pipeline for correlations and feature driver discovery.

* **Task 3.1:** Implement `datadoc/engine/stats.py` calculating Pearson and Spearman correlation matrices using NumPy/SciPy.
* **Task 3.2:** Implement feature driver importance ranking using `RandomForestRegressor`/`Classifier` combined with `mutual_info_classif`/`regression`.
* **Task 3.3:** Output results directly into validated Pydantic models (`CorrelationPair`, `FeatureDriver`).
* **Task 3.4:** Add unit tests in `tests/test_stats.py` benchmarking computation speed on synthetic 100,000-row datasets.

---

### Phase 4: Privacy Abstraction & AI Narrative Engine
**Goal:** Serialize statistical results into sanitized JSON metadata and interface with LiteLLM / Ollama with rule-based fallbacks.

* **Task 4.1:** Implement `datadoc/ai/abstraction.py` to strip all raw data and export clean JSON statistical payload.
* **Task 4.2:** Build `datadoc/ai/llm_client.py` using `LiteLLM` to query local Ollama instances or remote APIs.
* **Task 4.3:** Implement `datadoc/ai/fallback.py` to generate heuristic narrative text when LLM is unavailable or times out.
* **Task 4.4:** Write unit tests in `tests/test_ai.py` using mocked LLM responses to verify JSON validation resilience.

---

### Phase 5: Interactive Visualization Engine
**Goal:** Auto-generate responsive Plotly charts and Seaborn graphic fallbacks.

* **Task 5.1:** Implement `datadoc/viz/plots.py` with functions: `plot_distributions()`, `plot_correlation_heatmap()`, `plot_feature_importance()`.
* **Task 5.2:** Implement `datadoc/viz/themes.py` defining modern color schemes (e.g., Slate Blue, Emerald Green, Muted Charcoal).
* **Task 5.3:** Ensure all charts render cleanly to standalone HTML strings (`fig.to_html(include_plotlyjs='cdn')`).

---

### Phase 6: HTML Dashboard & PDF Template Export Engine
**Goal:** Build Jinja2 rendering pipeline and WeasyPrint PDF compiler.

* **Task 6.1:** Build `src/datadoc/export/templates/dashboard.html.j2` featuring responsive cards, statistical tables, and embedded Plotly figures.
* **Task 6.2:** Build `src/datadoc/export/templates/report.html.j2` and `style.css` tailored specifically for CSS Paged Media and WeasyPrint PDF output.
* **Task 6.3:** Implement `datadoc/export/pdf.py` wrapping `weasyprint.HTML().write_pdf()`.
* **Task 6.4:** Write end-to-end export tests in `tests/test_export.py` verifying generated HTML and PDF files exist and are non-empty.

---

### Phase 7: Public Interfaces (Python SDK, CLI, Streamlit App)
**Goal:** Expose clean user interfaces for programmatic, terminal, and browser workflows.

* **Task 7.1 Python API:** Implement main `datadoc.analyze()` entry point in `src/datadoc/__init__.py`.
  ```python
  import datadoc as dd
  report = dd.analyze("data.csv", target="sales", llm_provider="ollama")
  report.to_html("report.html")
  report.to_pdf("report.pdf")
  ```
* **Task 7.2 CLI Application:** Build `src/datadoc/cli/main.py` using Typer & Rich. Support options `--target`, `--llm`, `--output-html`, `--output-pdf`.
* **Task 7.3 Streamlit Web Application:** Build `src/datadoc/app/streamlit_app.py` enabling file drag-and-drop, interactive target selector, and live report download buttons.

---

### Phase 8: Comprehensive Testing, CI/CD, and PyPI Packaging
**Goal:** Finalize documentation, achieve >85% test coverage, configure PyPI packaging.

* **Task 8.1:** Write integration tests executing full pipeline from CSV input to PDF creation.
* **Task 8.2:** Set up GitHub Actions workflow `.github/workflows/ci.yml` running PyTest across Python 3.10, 3.11, and 3.12.
* **Task 8.3:** Configure `pyproject.toml` entry points for CLI command `datadoc`.
* **Task 8.4:** Build distribution wheel (`python -m build`) and verify installation in a clean virtual environment.

---

## 8. Explicit Guidelines & Directives for AI Coding Agents

When prompt-instructing AI agents (Cursor, Claude Dev, Aider) to build this repository, enforce these **strict development directives**:

1. **Type Annotations Required:** Every function signature MUST include explicit Python type hints (e.g., `def analyze(df: polars.DataFrame, target: Optional[str] = None) -> FullReportData:`).
2. **Polars-First Standard:** Do not convert Polars DataFrames to Pandas unless executing Scikit-Learn models. Keep data operations in native Polars.
3. **Zero Raw Data to LLMs:** Never pass dataframe slices (`df.head()`) to LLM prompt templates. Only pass aggregated, serialized statistical dicts from `datadoc.ai.abstraction`.
4. **Resilient Exception Handling:** Wrap external operations (file I/O, LLM API network calls, WeasyPrint rendering) in try-except blocks with explicit, descriptive error messages using custom `DataDocError` subclasses.
5. **No Monolithic Files:** Keep individual module files under 300 lines of code. Split large functions into testable helper functions.
6. **Self-Contained Test Fixtures:** Use PyTest fixtures in `conftest.py` to generate synthetic pandas and polars datasets (continuous, categorical, datetime, and null-heavy datasets) for offline unit testing without external data downloads.

---
*End of Blueprint — Ready for Execution by AI LLM Coding Agents.*
