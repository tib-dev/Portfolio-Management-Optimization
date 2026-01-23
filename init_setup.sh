#!/usr/bin/env bash

# ==============================================================================
# init_setup.sh
# Project: portfolio-management-optimization
# Description: Bootstrap full project structure (package-based src layout)
# Safe to re-run (conditional creation)
# ==============================================================================

set -e
set -o pipefail

echo "=============================================================="
echo "Portfolio Management Optimization - Project Structure Setup"
echo "=============================================================="

# ----------------------------
# Helper functions
# ----------------------------
create_dir () {
    [ -d "$1" ] || mkdir -p "$1"
}

create_file () {
    [ -f "$1" ] || touch "$1"
}

# ----------------------------
# 1. Project root directories
# ----------------------------
dirs=(
    "config"
    "data/raw"
    "data/interim"
    "data/processed"
    "data/external"
    "notebooks"
    "reports"
    "scripts"
    "docker"
    ".github/workflows"

    "src/pmo_forcasting"
    "src/pmo_forcasting/core"
    "src/pmo_forcasting/data"
    "src/pmo_forcasting/forecasting"
    "src/pmo_forcasting/optimization"
    "src/pmo_forcasting/backtesting"
    "src/pmo_forcasting/benchmarks"
    "src/pmo_forcasting/reporting"
    "src/pmo_forcasting/pipeline"
    "src/pmo_forcasting/utils"

    "tests"
)

for d in "${dirs[@]}"; do
    create_dir "$d"
done

echo "✓ Directories created"

# ----------------------------
# 2. Python package files
# ----------------------------
py_files=(

    # Root package
    "src/pmo_forcasting/__init__.py"

    # Core
    "src/pmo_forcasting/core/__init__.py"
    "src/pmo_forcasting/core/settings.py"
    "src/pmo_forcasting/core/config.py"
    "src/pmo_forcasting/core/paths.py"

    # Data
    "src/pmo_forcasting/data/__init__.py"
    "src/pmo_forcasting/data/load_market_data.py"
    "src/pmo_forcasting/data/clean_prices.py"
    "src/pmo_forcasting/data/returns.py"
    "src/pmo_forcasting/data/split.py"

    # Forecasting
    "src/pmo_forcasting/forecasting/__init__.py"
    "src/pmo_forcasting/forecasting/baseline.py"
    "src/pmo_forcasting/forecasting/arima.py"
    "src/pmo_forcasting/forecasting/lstm.py"
    "src/pmo_forcasting/forecasting/evaluate.py"

    # Optimization
    "src/pmo_forcasting/optimization/__init__.py"
    "src/pmo_forcasting/optimization/expected_returns.py"
    "src/pmo_forcasting/optimization/risk_models.py"
    "src/pmo_forcasting/optimization/efficient_frontier.py"
    "src/pmo_forcasting/optimization/constraints.py"
    "src/pmo_forcasting/optimization/optimize.py"

    # Backtesting
    "src/pmo_forcasting/backtesting/__init__.py"
    "src/pmo_forcasting/backtesting/simulator.py"
    "src/pmo_forcasting/backtesting/metrics.py"
    "src/pmo_forcasting/backtesting/plots.py"

    # Benchmarks
    "src/pmo_forcasting/benchmarks/__init__.py"
    "src/pmo_forcasting/benchmarks/spy_bnd_6040.py"
    "src/pmo_forcasting/benchmarks/compare.py"

    # Reporting
    "src/pmo_forcasting/reporting/__init__.py"
    "src/pmo_forcasting/reporting/tables.py"
    "src/pmo_forcasting/reporting/figures.py"
    "src/pmo_forcasting/reporting/executive_summary.py"

    # Pipeline
    "src/pmo_forcasting/pipeline/__init__.py"
    "src/pmo_forcasting/pipeline/run_forecasting.py"
    "src/pmo_forcasting/pipeline/run_optimization.py"
    "src/pmo_forcasting/pipeline/run_backtest.py"
    "src/pmo_forcasting/pipeline/run_full_pipeline.py"

    # Utils
    "src/pmo_forcasting/utils/__init__.py"
    "src/pmo_forcasting/utils/logger.py"
    "src/pmo_forcasting/utils/metrics.py"
    "src/pmo_forcasting/utils/helpers.py"

    # Tests
    "tests/__init__.py"
    "tests/test_data_pipeline.py"
    "tests/test_forecasting.py"
    "tests/test_optimization.py"
    "tests/test_backtesting.py"
    "tests/test_benchmarks.py"
)

for f in "${py_files[@]}"; do
    create_file "$f"
done

echo "✓ Python files created"

# ----------------------------
# 3. YAML configuration files
# ----------------------------
yaml_files=(
    "config/data.yaml"
    "config/forecasting.yaml"
    "config/optimization.yaml"
    "config/backtesting.yaml"
    "config/benchmarks.yaml"
    "config/reporting.yaml"
)

for y in "${yaml_files[@]}"; do
    create_file "$y"
done

echo "✓ Config files created"

# ----------------------------
# 4. Core project files
# ----------------------------
core_files=(
    "README.md"
    ".gitignore"
    ".env.example"
    "pyproject.toml"
    "docker/Dockerfile"
    "docker/docker-compose.yml"
)

for f in "${core_files[@]}"; do
    create_file "$f"
done

echo "✓ Core files created"

# ----------------------------
# 5. Scripts
# ----------------------------
scripts=(
    "scripts/run_pipeline.sh"
    "scripts/run_backtest.sh"
)

for s in "${scripts[@]}"; do
    create_file "$s"
    chmod +x "$s"
done

if [ ! -s scripts/run_pipeline.sh ]; then
cat <<EOF > scripts/run_pipeline.sh
#!/usr/bin/env bash
echo "Running full portfolio optimization pipeline..."
python -m portfolio_management_optimization.pipeline.run_full_pipeline
EOF
fi

if [ ! -s scripts/run_backtest.sh ]; then
cat <<EOF > scripts/run_backtest.sh
#!/usr/bin/env bash
echo "Running backtest and benchmark comparison..."
python -m portfolio_management_optimization.pipeline.run_backtest
EOF
fi

echo "✓ Scripts ready"


# ----------------------------
# 7. Final message
# ----------------------------
echo "=============================================================="
echo "Portfolio Management Optimization project is ready"
echo "Run pipeline: scripts/run_pipeline.sh"
echo "Run backtest: scripts/run_backtest.sh"
echo "Edit configs in ./config"
echo "=============================================================="
