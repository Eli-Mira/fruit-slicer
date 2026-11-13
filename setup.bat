@echo off
echo === Installing fruit-slicer ===
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-gesture.txt
echo Done. Run: python game.py
