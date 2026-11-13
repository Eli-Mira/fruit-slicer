#!/bin/bash
set -e
echo "=== Installing fruit-slicer ==="
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gesture.txt
echo "Done. Run: python game.py"
