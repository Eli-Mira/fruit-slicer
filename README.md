# Fruit Slicer — Arcane Swipe

A gesture-controlled fruit slicing game built with pygame. Swipe with your hand to cast spells and slice fruit.

## Features

- Hand gesture control via webcam (MediaPipe)
- Spell casting: swipe = slice, pinch = slow-mo
- Particle effects and combo scoring
- Local high-score persistence

## Requirements

- Python 3.11+
- Webcam
- Windows (tested) / Linux / macOS

## Install

```bash
# Create venv
python -m venv .venv
.venv\Scripts\activate

# Install deps
pip install -r requirements.txt
pip install -r requirements-gesture.txt
```

## Run

```bash
# Windows
run_game.bat

# Or directly
python game.py
```

## Controls

| Gesture | Action |
|---------|--------|
| Swipe hand | Cast slice spell |
| Pinch fingers | Slow-motion mode |
| Escape / Close window | Quit |
| Any key (game over) | Restart |

## Project Structure

```
fruit-slicer/
├── game.py              # Main game loop
├── gesture_engine.py    # Gesture detection (MediaPipe)
├── requirements.txt     # Core deps (pygame, etc.)
├── requirements-gesture.txt # MediaPipe, OpenCV
└── run_game.bat         # Windows launcher
```

## License

MIT License — see [LICENSE](LICENSE)
