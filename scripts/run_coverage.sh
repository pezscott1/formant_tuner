#!/usr/bin/env bash
set -e

export QT_QPA_PLATFORM=offscreen

pytest \
  --cov=. \
  --cov-branch \
  --cov-report=term:skip-covered \
  --cov-report=html \
  --cov-report=xml \
  -q

echo ""
echo "=============================================="
echo " Branch + Condition Coverage Enabled"
echo " Coverage report generated:"
echo "  • Terminal summary"
echo "  • HTML:   htmlcov/index.html"
echo "  • XML:    coverage.xml"
echo "=============================================="