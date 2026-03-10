#!/usr/bin/env bash
# Launch GenMol Studio
# Usage: bash run_app.sh [--port 8501]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Install app dependencies if missing
pip install -q streamlit plotly scikit-learn 2>/dev/null || true

echo "🧬 Starting GenMol Studio …"
echo "   Open http://localhost:${2:-8501} in your browser"
echo ""

streamlit run "$SCRIPT_DIR/app/app.py" \
    --server.headless true \
    --server.port "${2:-8501}" \
    --theme.base dark \
    --theme.primaryColor "#7C3AED" \
    "$@"
