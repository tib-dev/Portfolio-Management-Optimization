# Portfolio Management Optimization

**Designing, Optimizing, and Evaluating Investment Portfolios Against a 60/40 Benchmark**  
An end-to-end quantitative investment pipeline covering data ingestion, forecasting, portfolio optimization, backtesting, and executive-level reporting.

This project builds a **production-grade portfolio analytics system** that transforms raw market data into optimized portfolios and compares performance against a standard **60/40 SPY/BND benchmark**, generating insights suitable for both technical teams and business leadership.

---

## Table of Contents



- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Objectives](#objectives)
- [Data & Features](#data--features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Pipeline & Processing Steps](#pipeline--processing-steps)
- [Benchmark](#benchmark)
- [Engineering Practices](#engineering-practices)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Technologies Used](#technologies-used)
- [Author](#author)



---

## Project Overview

This project implements a **full investment research and portfolio optimization workflow**, similar to what is used by asset managers, advisory firms, and fintech platforms.

The pipeline handles the complete lifecycle:

- Ingesting historical market data
- Forecasting expected returns and risk
- Constructing optimized portfolios
- Backtesting strategies over time
- Benchmarking against a passive 60/40 portfolio
- Producing executive-ready insights

This is a **decision-support system**, not a price-prediction toy project.

---

## Business Context

Investment teams must justify whether sophisticated strategies outperform simple, low-cost allocations.

The **60/40 SPY/BND portfolio** is an industry-standard benchmark. Any advanced strategy must demonstrate:

- Better risk-adjusted returns
- Controlled drawdowns
- Transparency and explainability

This project answers those questions using a disciplined, auditable workflow.

---

## Objectives

- Build a reproducible and modular investment analytics pipeline
- Forecast expected returns using time-series techniques
- Apply Modern Portfolio Theory under realistic constraints
- Evaluate performance through backtesting
- Compare results against a 60/40 benchmark
- Communicate results clearly to non-technical stakeholders

---

## Data & Features

### Assets

- **SPY** – US equity market proxy
- **BND** – US total bond market proxy
- **TSLA** – High-growth equity (used in extended scenarios)

### Core Fields

| Column | Description |
|------|------------|
| date | Trading date |
| asset | Asset ticker |
| price | Adjusted close price |
| return | Daily or periodic return |

### Derived Features

- Log returns
- Rolling volatility
- Rolling correlations
- Expected returns
- Portfolio-level metrics (Sharpe, drawdown, volatility)

---

## Project Structure

```text
portfolio-management-optimization/
├── config/                        # Global configs (model, backtest, constraints)
├── data/
│   ├── raw/                       # Raw market data
│   ├── interim/                   # Cleaned & feature-engineered data
│   └── processed/                 # Model-ready datasets
├── notebooks/                     # EDA and research notebooks
├── reports/                       # Backtest & executive reports
├── src/
│   └── portfolio_management_optimization/
│       ├── core/                  # Settings and configuration loaders
│       ├── data/                  # Data ingestion & preprocessing
│       ├── forecasting/           # ARIMA / baseline forecasts
│       ├── optimization/          # MPT & Efficient Frontier logic
│       ├── backtesting/            # Strategy simulation & evaluation
│       ├── benchmarks/             # 60/40 benchmark logic
│       ├── reporting/              # Metrics & visualization
│       ├── pipeline/               # Orchestration logic
│       └── utils/                  # Helpers and shared utilities
├── tests/                         # Unit and integration tests
├── scripts/                       # CLI scripts to run pipeline stages
├── docker/                        # Containerization (optional)
├── pyproject.toml
├── init_setup.sh
└── README.md
```

---

## Architecture

```text
┌─────────────────────┐
│   Market Data API   │
│  (Yahoo Finance)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Data Ingestion &   │
│  Preprocessing      │
│  - Cleaning         │
│  - Returns          │
│  - Volatility       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Forecasting Layer   │
│ - Historical Mean   │
│ - ARIMA (optional)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Portfolio           │
│ Optimization        │
│ - Covariance        │
│ - Efficient Frontier│
│ - Constraints       │
└─────────┬───────────┘
          │
          ├──────────────┐
          ▼              ▼
┌─────────────────┐  ┌─────────────────┐
│ Strategy        │  │ 60/40 Benchmark  │
│ Backtesting     │  │ (SPY/BND)        │
└─────────┬───────┘  └─────────┬───────┘
          │                     │
          └──────────┬──────────┘
                     ▼
            ┌───────────────────┐
            │ Performance &     │
            │ Risk Reporting    │
            │ - Sharpe          │
            │ - Drawdown        │
            │ - Volatility      │
            └───────────────────┘
```

**Key architectural principles:**

- Separation of research, optimization, and evaluation
- Benchmark treated as a first-class component
- Forecasts used as *inputs*, not absolute truths
- Emphasis on interpretability and decision support

---

## Pipeline & Processing Steps

1. Load historical market data
2. Compute returns and rolling risk metrics
3. Forecast expected returns
4. Optimize portfolio allocations
5. Backtest optimized vs benchmark portfolios
6. Generate comparative performance reports

---

## Benchmark

- 60% SPY (equities)
- 40% BND (bonds)

Used as a passive baseline for performance comparison.

Metrics evaluated:

- Cumulative return
- Volatility
- Sharpe ratio
- Maximum drawdown

---

## Engineering Practices

- Reproducible, configuration-driven pipelines
- Modular and testable design
- Clear separation of concerns
- Focus on business explainability

---

## Setup & Installation

```bash
git clone https://github.com/<username>/portfolio-management-optimization.git
cd portfolio-management-optimization

python -m venv .venv
source .venv/bin/activate

pip install -e .
cp .env.example .env
```

---

## Running the Project

Run full pipeline:

```bash
scripts/run_pipeline.sh
```

Run backtest only:

```bash
scripts/run_backtest.sh
```

Results are written to the `reports/` directory.

---

## Technologies Used

- Python 3.10+
- NumPy, Pandas, SciPy
- Statsmodels
- PyPortfolioOpt
- Matplotlib / Seaborn
- Pytest
- Docker (optional)

---

## Author

Tibebu Kaleb  
Full-stack AI/ML engineer