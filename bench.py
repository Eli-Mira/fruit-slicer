"""Headless frame-time benchmark for game.py.

Runs the real update/draw code against SDL's dummy video driver (software
rendering, no window) with the engine seam faked to a deterministic sweeping
cursor and permanent swipe — worst-case slicing load. Absolute numbers differ
from a GPU-composited desktop window; relative comparisons and scaling trends
carry over.

Usage:
    python bench.py [frames_per_scene]
"""

from __future__ import annotations

import math
import os
import random
import statistics as stats
import sys
import time

median = stats.median

os.environ.setdefault("SDL_VIDEODUMMY", "1")

import pygame  # noqa: E402  (after dummy driver env)

import game as G  # noqa: E402

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SCENES = [  # (name, target_fruits, particle_cap)
    ("typical  (6 fruit / 80 particles)", 6, 80),
    ("heavy   (24 fruit / 300 particles)", 24, 300),
    ("stress  (60 fruit / 800 particles)", 60, 800),
]

PALETTE = G.FRUIT_COLORS + [G.GOLD_COLOR]


def make_fruit() -> G.Fruit:
    r = random.randint(G.FRUIT_MIN_R, G.FRUIT_MAX_R)
    golden = random.random() < 0.12
    return G.Fruit(
        x=random.uniform(r, G.WIDTH - r),
        y=random.uniform(-40, G.HEIGHT * 0.8),
        r=float(r),
        vy=float(random.randint(120, 200)),
        vx=random.uniform(-G.FRUIT_DRIFT_MAX, G.FRUIT_DRIFT_MAX),
        color=G.GOLD_COLOR if golden else random.choice(G.FRUIT_COLORS),
        points=5 if golden else 1,
        golden=golden,
    )


def main() -> None:
    random.seed(1234)
    pygame.init()
    screen = pygame.display.set_mode((G.WIDTH, G.HEIGHT))
    font = pygame.font.SysFont(None, 32)
    big_font = pygame.font.SysFont(None, 72)
    hud_font = pygame.font.SysFont(None, G.SCORE_FONT_SIZE)

    # --- fake the seam: deterministic sweeping blade, always swiping -------
    t0 = time.perf_counter()
    G.camera_alive = lambda: True
    G.read_gesture = lambda: "swipe_left"

    def fake_cursor():
        t = time.perf_counter() - t0
        return (400 + 370 * math.sin(t * 0.9), 300 + 260 * math.sin(t * 1.37 + 1.0))

    G.read_cursor_px = fake_cursor

    # --- per-section timers -------------------------------------------------
    acc: dict[str, list[float]] = {}
    names = ["poll_input", "step_spawns", "step_fruits", "try_slice",
             "update_particles", "update_popups", "update_ambient",
             "draw_world", "flip"]
    for name in names:
        acc[name] = []

    def timed(name, fn):
        def wrapper(*a, **k):
            start = time.perf_counter()
            result = fn(*a, **k)
            acc[name].append(time.perf_counter() - start)
            return result
        return wrapper

    for name in names:
        fn = getattr(G, name, None)
        if fn is not None:
            setattr(G, name, timed(name, fn))


    print(f"{'scene':<38}{'busy ms':>9}{'  (work only)':>7}")
    overall_summary = []

    # --- warmup: pay all lazy-init costs (fonts, caches) outside measurements --
    warm = G.new_game_state()
    warm.lives = 500
    warm.fruits = [make_fruit() for _ in range(24)]
    dt = 1.0 / G.FPS
    for _ in range(60):
        G.update_state(warm, dt, None)
        G.draw_world(screen, warm, font, big_font, hud_font)
    pygame.display.flip()

    for scene_name, target_fruits, particle_cap in SCENES:
        state = G.new_game_state()
        state.lives = 500                        # effectively endless; HUD capped
        state.fruits = [make_fruit() for _ in range(target_fruits)]
        # clear section buffers between scenes
        for lst in acc.values():
            lst.clear()

        frames_done = 0
        # No clock.tick pacing: run back-to-back with fixed dt so we measure
        # pure processing (busy) time rather than the deliberate 30fps idle.
        dt = 1.0 / G.FPS
        while frames_done < FRAMES:
            frame_start = time.perf_counter()
            G.update_state(state, dt, None)
            # hold object counts at scenario targets regardless of slicing
            if len(state.fruits) < target_fruits:
                state.fruits.append(make_fruit())
            elif len(state.fruits) > target_fruits:
                del state.fruits[target_fruits:]
            while len(state.particles) > particle_cap:
                state.particles.sprites().pop().kill()
            G.draw_world(screen, state, font, big_font, hud_font)
            flip_start = time.perf_counter()
            pygame.display.flip()
            acc["flip"].append(time.perf_counter() - flip_start)
            acc.setdefault("frame_total", []).append(
                time.perf_counter() - frame_start)
            frames_done += 1

        total_ms = median(acc["frame_total"][-FRAMES:]) * 1000
        overall_summary.append((scene_name, total_ms))
        row = {n: median(v[-FRAMES:]) * 1000 for n, v in acc.items()
               if len(v[-FRAMES:]) > 0}
        print(f"{scene_name:<38}{total_ms:>9.3f}{'busy':>7}")
        detail = ", ".join(f"{n}={row[n]:.3f}" for n in names
                          if acc[n] and n != "flip")
        print(f"    breakdown ms/call (median): {detail}")

    print("\nsummary (avg ms/frame):")
    for scene_name, total_ms in overall_summary:
        print(f"    {scene_name}: {total_ms:.2f}")


if __name__ == "__main__":
    main()
