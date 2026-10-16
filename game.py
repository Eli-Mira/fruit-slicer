"""Arcane Swipe — gesture spell duel with pygame.

This file only consumes gesture_engine.py's public API:
    read_gesture()        -> swipe labels trigger casting
    read_cursor_px()      -> normalized (0..1) cursor = wand position
    camera_alive()        -> whether the capture thread is running
    release_camera()      -> clean shutdown

Run with the gesture_control_app venv:
    C:\\Users\\emadg\\gesture_control_app\\.venv\\Scripts\\python.exe game.py

Quit: window close or Escape. Game over: press any key to restart.
"""

from __future__ import annotations

import dataclasses
import math
import random
import sys
import time
from collections import deque

import numpy as np
import pygame

try:
    import gesture_engine as _engine  # noqa: E402
except ImportError as exc:
    raise SystemExit(
        f"\n[game] Missing dependency: {exc!s}\n\n"
        "This game needs pygame + numpy + opencv + mediapipe.\n"
        "Install them into your active Python environment:\n"
        "    python -m pip install -r requirements.txt\n\n"
        "Or launch via run_game.bat / run_game.ps1, which auto-picks a\n"
        "Python that already has pygame (e.g. the gesture_control_app venv).\n"
    ) from exc


# --- tunables ---------------------------------------------------------------
WIDTH, HEIGHT = 800, 600
FPS = 30
MAX_FRAME_DT = 0.10
ENGINE_RETRY_INTERVAL = 0.5
BG_TOP = (14, 16, 32)
BG_BOTTOM = (4, 5, 10)

HUD_COLOR = (240, 240, 240)
HUD_SHADOW = (20, 20, 20)

BLADE_RADIUS = 18
BLADE_COLOR = (160, 210, 255)
TRAIL_LEN = 10

SPELLS = {
    "swipe_up":    {"name": "fire",    "color": (240, 120, 40),  "speed": 420, "size": 10, "lifetime": 1.4, "particles": 18},
    "swipe_down":  {"name": "ice",     "color": (110, 220, 255), "speed": 260, "size": 14, "lifetime": 0.55,"particles": 26},
    "swipe_left":  {"name": "lightning","color": (255, 230, 120),"speed": 720, "size": 6,  "lifetime": 0.7, "particles": 14},
    "swipe_right": {"name": "arcane",  "color": (180, 120, 255), "speed": 340, "size": 14, "lifetime": 1.1, "particles": 22},
}
SWIPE_LABELS = set(SPELLS)

PARTICLE_LIFE = 0.55
SHAKE_TIME = 0.30
SHAKE_AMPLITUDE = 8.0
SHAKE_COMBO_WINDOW = 0.25
SHAKE_COMBO_THRESHOLD = 2

SCORE_FONT_SIZE = 44
HUD_PADDING = (16, 12)
POPUP_LIFE = 0.75
POPUP_RISE_SPEED = 50.0

START_LIVES = 5
ENEMY_BASE_SPEED = 110.0
ENEMY_SPEED_RAMP = 2.5
SPAWN_BASE_INTERVAL = 1.35
SPAWN_MIN_INTERVAL = 0.45
SPAWN_RAMP = 0.020

CURSOR_SMOOTHING = 0.45
GOLDEN_CHANCE = 0.15
GOLDEN_POINTS = 5

# --- accessibility: spell indicator shapes -----------------------------------
SPELL_SHAPES = {
    "fire": "circle",
    "ice": "diamond",
    "lightning": "triangle",
    "arcane": "hexagon",
}
SPELL_HUD_COLORS = {
    "fire": (240, 120, 40),
    "ice": (110, 220, 255),
    "lightning": (255, 230, 120),
    "arcane": (180, 120, 255),
}


# --- engine seam ------------------------------------------------------------
def read_gesture() -> str | None:
    return _engine.get_gesture()


def read_cursor_px() -> tuple[float, float] | None:
    pos = _engine.get_hand_position()
    return (pos[0] * WIDTH, pos[1] * HEIGHT) if pos else None


def camera_alive() -> bool:
    return _engine.is_running()


def release_camera() -> None:
    _engine.stop()


# --- ambient background -----------------------------------------------------
_ambient_dots: list[list[float]] | None = None
_ambient_t = 0.0
_AMBIENT_TINTS = [(80, 90, 140), (100, 80, 130), (70, 120, 140)]


def init_ambient() -> None:
    global _ambient_dots
    if _ambient_dots is None:
        _ambient_dots = []
        for _ in range(48):
            _ambient_dots.append([
                random.uniform(0.0, WIDTH),
                random.uniform(0.0, HEIGHT),
                random.uniform(5.0, 18.0),
                random.uniform(1.0, 2.4),
                random.uniform(0.0, math.tau),
                random.choice(_AMBIENT_TINTS),
            ])


def update_ambient(dt: float) -> None:
    global _ambient_t
    if _ambient_dots is None:
        init_ambient()
    _ambient_t += dt
    for dot in _ambient_dots:
        dot[1] += dot[2] * dt
        if dot[1] > HEIGHT + 3:
            dot[1] = -3.0
            dot[0] = random.uniform(0.0, WIDTH)


def draw_ambient(screen: pygame.Surface) -> None:
    if _ambient_dots is None:
        return
    for x, y, _spd, r, phase, tint in _ambient_dots:
        twinkle = 0.55 + 0.45 * math.sin(phase + _ambient_t * 1.5)
        shade = tuple(int(c * twinkle) for c in tint)
        pygame.draw.circle(screen, shade, (int(x), int(y)), int(r))


# --- data types -------------------------------------------------------------
@dataclasses.dataclass
class Enemy:
    x: float
    y: float
    r: float
    vx: float
    vy: float
    color: tuple[int, int, int]
    kind: str = "normal"
    angle: float = 0.0
    angular_vel: float = 0.0


@dataclasses.dataclass
class Spell:
    x: float
    y: float
    kind: str
    vx: float
    vy: float
    life: float
    max_life: float
    r: float
    color: tuple[int, int, int]
    trail: list[tuple[float, float]]


class Particle(pygame.sprite.Sprite):
    _pool: list[Particle] | None = None

    @classmethod
    def acquire(cls, x: float, y: float, vx: float, vy: float,
                life: float, color: tuple[int, int, int], size: float) -> Particle:
        p = cls._pool.pop() if cls._pool else cls.__new__(cls)
        pygame.sprite.Sprite.__init__(p)
        p.x = x
        p.y = y
        p.vx = vx
        p.vy = vy
        p.life = life
        p.max_life = life
        p.color = color
        p.size = size
        p.image = pygame.Surface((size * 2 + 2, size * 2 + 2), pygame.SRCALPHA)
        p.rect = p.image.get_rect(center=(x, y))
        p.draw_image()
        return p

    def draw_image(self) -> None:
        self.image.fill((0, 0, 0, 0))
        pygame.draw.circle(self.image, (*self.color, 200),
                           (int(self.size + 1), int(self.size + 1)), int(self.size))

    def kill(self) -> None:
        super().kill()
        type(self)._pool.append(self)

    def __init__(self, x: float, y: float, vx: float, vy: float,
                 life: float, color: tuple[int, int, int]) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = 4.0
        self.image = pygame.Surface((1, 1), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(x, y))

    def update(self, dt: float) -> None:
        self.life -= dt
        if self.life <= 0.0:
            self.kill()
            return
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.94
        self.vy = self.vy * 0.94 + 240.0 * dt
        self.rect.center = (self.x, self.y)


@dataclasses.dataclass
class Popup:
    text: str
    x: float
    y: float
    life: float
    color: tuple[int, int, int]
    scale: float = 1.0


@dataclasses.dataclass
class GameState:
    score: int = 0
    lives: int = START_LIVES
    elapsed: float = 0.0
    spawn_timer: float = 0.0
    enemies: list[Enemy] = dataclasses.field(default_factory=list)
    spells: list[Spell] = dataclasses.field(default_factory=list)
    particles: pygame.sprite.Group = dataclasses.field(default_factory=pygame.sprite.Group)
    popups: list[Popup] = dataclasses.field(default_factory=list)
    cursor: tuple[float, float] | None = None
    prev_cursor: tuple[float, float] | None = None
    trail: deque = dataclasses.field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    game_over: bool = False
    paused: bool = False
    engine_poll_timer: float = 0.0
    score_pop_timer: float = 0.0
    shake_time: float = 0.0
    combo_timer: float = 0.0
    combo_count: int = 0
    slow_mo_timer: float = 0.0
    cast_times: dict[str, float] = dataclasses.field(default_factory=lambda: {})


def new_game_state() -> GameState:
    return GameState()


# --- audio ------------------------------------------------------------------
def _swoosh(sample_rate: float, duration: float, f_start: float,
            f_end: float, decay: float) -> pygame.sndarray.Sound:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    sweep = np.linspace(f_start, f_end, t.size)
    wave = np.sin(2.0 * math.pi * np.cumsum(sweep) / sample_rate)
    envelope = np.exp(-t * decay)
    samples = (wave * envelope * 32767.0 * 0.5).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([samples, samples]))
    return pygame.sndarray.make_sound(stereo)


def _drone(sample_rate: float, duration: float, freq: float) -> pygame.sndarray.Sound:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2.0 * math.pi * freq * t) * 0.08
    samples = (wave * 32767.0).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([samples, samples]))
    return pygame.sndarray.make_sound(stereo)


def init_audio(mute: bool) -> tuple[dict[str, pygame.mixer.Sound], pygame.mixer.Sound | None] | None:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=256)
        casts = {}
        for label in SPELLS:
            speed = SPELLS[label]["speed"]
            casts[label] = _swoosh(22050, 0.12 + random.uniform(0.0, 0.05), 600 + speed, 160 + random.uniform(0, 80), 26 + random.uniform(0, 8))
        drone = _drone(22050, 2.0, 55.0)
        drone.set_volume(0.25)
        drone.play(loops=-1)
        return casts, drone
    except Exception as exc:
        print(f"[audio] sound disabled ({exc})")
        return None


# --- spawning ---------------------------------------------------------------
def spawn_enemy(state: GameState) -> None:
    side = random.randint(0, 3)
    if side == 0:
        x, y = random.uniform(20, WIDTH - 20), -30
    elif side == 1:
        x, y = random.uniform(20, WIDTH - 20), HEIGHT + 30
    elif side == 2:
        x, y = -30, random.uniform(20, HEIGHT - 20)
    else:
        x, y = WIDTH + 30, random.uniform(20, HEIGHT - 20)

    golden = random.random() < GOLDEN_CHANCE
    r = random.uniform(12, 26)
    speed = min(ENEMY_BASE_SPEED + state.elapsed * ENEMY_SPEED_RAMP, 380.0)
    speed *= 1.25 if golden else 1.0
    dx = (state.cursor[0] if state.cursor else WIDTH / 2) - x
    dy = (state.cursor[1] if state.cursor else HEIGHT / 2) - y
    mag = math.hypot(dx, dy) or 1.0
    color = (255, 215, 0) if golden else random.choice([
        (231, 76, 60), (230, 126, 34), (241, 196, 15),
        (46, 204, 113), (52, 152, 219), (155, 89, 182),
    ])
    kind = "fast" if random.random() < 0.25 else "normal"
    state.enemies.append(Enemy(
        x=x, y=y, r=r,
        vx=dx / mag * speed,
        vy=dy / mag * speed,
        color=color,
        kind=kind,
        angle=random.uniform(0.0, math.tau),
        angular_vel=random.uniform(-3.0, 3.0),
    ))


def spawn_spell(state: GameState, label: str) -> None:
    cfg = SPELLS[label]
    if state.cursor is None:
        return
    cx, cy = state.cursor
    if state.prev_cursor is None:
        return
    dx, dy = cx - state.prev_cursor[0], cy - state.prev_cursor[1]
    mag = math.hypot(dx, dy) or 1.0
    speed = cfg["speed"] + min(mag * 2.5, 180.0)
    vx, vy = dx / mag * speed, dy / mag * speed
    state.spells.append(Spell(
        x=cx, y=cy, kind=cfg["name"], vx=vx, vy=vy,
        life=cfg["lifetime"], max_life=cfg["lifetime"], r=cfg["size"],
        color=cfg["color"], trail=[],
    ))


# --- particles --------------------------------------------------------------
def burst_particles(state: GameState, x: float, y: float,
                    color: tuple[int, int, int], count: int = 18) -> None:
    for _ in range(count):
        angle = random.uniform(0.0, math.tau)
        speed = random.uniform(40.0, 240.0)
        size = random.uniform(2.0, 5.5)
        p = Particle.acquire(
            x=x, y=y,
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed,
            life=PARTICLE_LIFE,
            color=color,
            size=size,
        )
        state.particles.add(p)


# --- popups -----------------------------------------------------------------
def spawn_popup(state: GameState, x: float, y: float, text: str,
                color: tuple[int, int, int]) -> None:
    state.popups.append(Popup(text=text, x=x, y=y, life=POPUP_LIFE, color=color))


# --- physics/combat ---------------------------------------------------------
def update_spells(state: GameState, dt: float) -> None:
    alive: list[Spell] = []
    for s in state.spells:
        s.life -= dt
        if s.life <= 0.0:
            continue
        s.trail.append((s.x, s.y))
        if len(s.trail) > 10:
            s.trail.pop(0)
        s.x += s.vx * dt
        s.y += s.vy * dt
        if s.x < -60 or s.x > WIDTH + 60 or s.y < -60 or s.y > HEIGHT + 60:
            continue
        alive.append(s)
    state.spells = alive


def update_enemies(state: GameState, dt: float) -> None:
    for enemy in state.enemies:
        enemy.x += enemy.vx * dt
        enemy.y += enemy.vy * dt
        enemy.angle += enemy.angular_vel * dt
    despawn = False
    kept: list[Enemy] = []
    for enemy in state.enemies:
        if math.hypot(enemy.x - WIDTH / 2, enemy.y - HEIGHT / 2) < enemy.r + 14:
            despawn = True
            state.lives -= 1
        else:
            kept.append(enemy)
    if despawn:
        state.enemies = kept
    if state.lives <= 0 and not state.game_over:
        state.game_over = True
        print(f"[game] game over — final score {state.score}")


def try_cast(state: GameState, label: str | None) -> None:
    if label not in SPELLS:
        return
    now = time.perf_counter()
    last = state.cast_times.get(label, 0.0)
    if now - last < 0.22:
        return
    state.cast_times[label] = now
    spawn_spell(state, label)


def collide_spells_enemies(state: GameState, sounds: dict[str, pygame.mixer.Sound] | None) -> None:
    enemies_removed = False
    spells_removed = False
    kept_enemies: list[Enemy] = []
    for enemy in state.enemies:
        hit = False
        for s in state.spells:
            if math.hypot(s.x - enemy.x, s.y - enemy.y) <= enemy.r + s.r:
                hit = True
                break
        if hit:
            enemies_removed = True
            state.score += GOLDEN_POINTS if enemy.kind == "fast" else 1
            state.score_pop_timer = 0.22
            state.combo_timer = max(state.combo_timer, SHAKE_COMBO_WINDOW)
            state.combo_count += 1
            if state.combo_count >= SHAKE_COMBO_THRESHOLD:
                state.shake_time = max(state.shake_time, SHAKE_TIME + 0.1)
            burst_particles(state, enemy.x, enemy.y, enemy.color, 24)
            spawn_popup(state, enemy.x, enemy.y,
                        f"+{GOLDEN_POINTS if enemy.kind=='fast' else 1}",
                        (255, 235, 180))
            if sounds:
                sounds.get(s.kind, sounds["fire"]).play()
            continue
        kept_enemies.append(enemy)
    if enemies_removed:
        state.enemies = kept_enemies


# --- input ------------------------------------------------------------------
def poll_input(state: GameState, dt: float) -> str | None:
    gesture = None if state.game_over else read_gesture()

    if camera_alive():
        state.engine_poll_timer = 0.0
        new_cursor = read_cursor_px()
    elif state.engine_poll_timer >= ENGINE_RETRY_INTERVAL:
        state.engine_poll_timer -= ENGINE_RETRY_INTERVAL
        new_cursor = read_cursor_px()
    else:
        state.engine_poll_timer += dt
        new_cursor = None

    state.prev_cursor = state.cursor
    if new_cursor is None:
        state.cursor = None
        state.trail.clear()
    elif state.cursor is None:
        state.cursor = new_cursor
        state.trail.append(state.cursor)
    else:
        k = CURSOR_SMOOTHING * dt * 30.0
        state.cursor = (state.cursor[0] + (new_cursor[0] - state.cursor[0]) * k,
                        state.cursor[1] + (new_cursor[1] - state.cursor[1]) * k)
        state.trail.append(state.cursor)
    return gesture


def current_spawn_interval(elapsed: float) -> float:
    return max(SPAWN_MIN_INTERVAL, SPAWN_BASE_INTERVAL - elapsed * SPAWN_RAMP)


# --- update -----------------------------------------------------------------
def update_state(state: GameState, dt: float,
                 sounds: dict[str, pygame.mixer.Sound] | None) -> str | None:
    effective_dt = dt
    if state.slow_mo_timer > 0.0:
        state.slow_mo_timer = max(0.0, state.slow_mo_timer - dt)
        effective_dt *= 0.45

    gesture = poll_input(state, effective_dt)
    if not state.game_over:
        state.elapsed += effective_dt
        state.spawn_timer += effective_dt
        if state.spawn_timer >= current_spawn_interval(state.elapsed):
            state.spawn_timer = 0.0
            spawn_enemy(state)
        update_spells(state, effective_dt)
        update_enemies(state, effective_dt)
        collide_spells_enemies(state, sounds)
        try_cast(state, gesture)

    if state.score_pop_timer > 0.0:
        state.score_pop_timer = max(0.0, state.score_pop_timer - dt)

    update_particles(state, dt)
    update_popups(state, dt)
    update_ambient(effective_dt)

    if state.combo_timer > 0.0:
        state.combo_timer = max(0.0, state.combo_timer - dt)
    else:
        state.combo_count = 0

    if state.shake_time > 0.0:
        state.shake_time = max(0.0, state.shake_time - dt)

    return gesture


def update_particles(state: GameState, dt: float) -> None:
    state.particles.update(dt)


def update_popups(state: GameState, dt: float) -> None:
    alive: list[Popup] = []
    removed = False
    for p in state.popups:
        p.life -= dt
        if p.life <= 0.0:
            removed = True
            continue
        p.y -= POPUP_RISE_SPEED * dt
        p.scale = 1.0 + 0.3 * (p.life / POPUP_LIFE)
        alive.append(p)
    if removed:
        state.popups = alive


# --- rendering caches -------------------------------------------------------
_TEXT_CACHE: dict[tuple[int, str, tuple[int, int, int]], pygame.Surface] = {}


def get_static_text(font: pygame.font.Font, text: str,
                    color: tuple[int, int, int]) -> pygame.Surface:
    key = (id(font), text, color)
    surf = _TEXT_CACHE.get(key)
    if surf is None:
        surf = font.render(text, True, color)
        _TEXT_CACHE[key] = surf
    return surf


class DynamicText:
    def __init__(self) -> None:
        self._key: tuple[int, str, tuple[int, int, int]] | None = None
        self._surface: pygame.Surface | None = None

    def render(self, font: pygame.font.Font, text: str,
               color: tuple[int, int, int]) -> pygame.Surface:
        key = (id(font), text, color)
        if key != self._key:
            self._surface = font.render(text, True, color)
            self._key = key
        return self._surface


_SCORE_TEXT = DynamicText()
_FINAL_SCORE_TEXT = DynamicText()
_DIM_OVERLAY: pygame.Surface | None = None


def get_dim_overlay() -> pygame.Surface:
    global _DIM_OVERLAY
    if _DIM_OVERLAY is None:
        _DIM_OVERLAY = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        _DIM_OVERLAY.fill((0, 0, 0, 170))
        _DIM_OVERLAY = _DIM_OVERLAY.convert_alpha()
    return _DIM_OVERLAY


_BG: pygame.Surface | None = None


def get_background() -> pygame.Surface:
    global _BG
    if _BG is None:
        top = np.array(BG_TOP, dtype=float)
        bottom = np.array(BG_BOTTOM, dtype=float)
        rows = np.linspace(top, bottom, HEIGHT).astype(np.uint8)
        arr = np.repeat(rows[:, None, :], WIDTH, axis=1)
        _BG = pygame.image.frombuffer(arr.tobytes(), (WIDTH, HEIGHT), "RGB").convert()
    return _BG


# --- spell sprites ----------------------------------------------------------
_SPELL_SPRITES: dict[str, pygame.Surface] = {}


def get_spell_sprite(kind: str, color: tuple[int, int, int], radius: int) -> pygame.Surface:
    key = kind
    if key in _SPELL_SPRITES:
        return _SPELL_SPRITES[key]
    size = radius * 2 + 6
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2
    if kind == "fire":
        pygame.draw.circle(surf, (*color, 90), (cx, cy), radius + 5)
        pygame.draw.circle(surf, (*color, 220), (cx, cy), radius)
        pygame.draw.circle(surf, (255, 220, 160, 180), (cx, cy), max(2, radius - 3))
    elif kind == "ice":
        pygame.draw.circle(surf, (*color, 80), (cx, cy), radius + 4)
        pygame.draw.circle(surf, (*color, 230), (cx, cy), radius)
        pygame.draw.circle(surf, (220, 250, 255, 220), (cx, cy), max(2, radius - 2))
    elif kind == "lightning":
        pts = [(cx, 0), (cx + radius, cy + radius // 2), (cx - radius, cy + radius // 2), (cx, size)]
        pygame.draw.lines(surf, (*color, 240), False, pts, 3)
        pygame.draw.lines(surf, (255, 255, 255, 180), False, pts, 1)
    elif kind == "arcane":
        pygame.draw.circle(surf, (*color, 80), (cx, cy), radius + 6)
        pygame.draw.circle(surf, (*color, 230), (cx, cy), radius)
        pygame.draw.circle(surf, (255, 220, 255, 200), (cx, cy), max(2, radius - 2))
    _SPELL_SPRITES[key] = surf.convert_alpha()
    return _SPELL_SPRITES[key]


# --- trail ------------------------------------------------------------------
_TRAIL_SPRITES = [
    (tuple(int(c * ((i + 1) / TRAIL_LEN)) for c in BLADE_COLOR),
     max(2, int(BLADE_RADIUS * ((i + 1) / TRAIL_LEN) * 0.6)))
    for i in range(TRAIL_LEN)
]


# --- shake ------------------------------------------------------------------
def _shake_offset(state: GameState) -> tuple[int, int]:
    if state.shake_time <= 0.0:
        return 0, 0
    f = state.shake_time / SHAKE_TIME
    amp = SHAKE_AMPLITUDE * f
    return random.randint(-int(amp), int(amp)), random.randint(-int(amp), int(amp))


# --- drawing helpers --------------------------------------------------------
def _blit_centered(screen: pygame.Surface, surface: pygame.Surface,
                   center: tuple[float, float]) -> None:
    screen.blit(surface, surface.get_rect(center=center))


# --- HUD --------------------------------------------------------------------
def draw_hud(screen: pygame.Surface, state: GameState,
             hud_font: pygame.font.Font, label_font: pygame.font.Font) -> None:
    score_surf = _SCORE_TEXT.render(hud_font, f"Score: {state.score}", HUD_COLOR)
    if state.score_pop_timer > 0.0:
        m = 1.0 + 0.35 * (state.score_pop_timer / 0.22)
        w, h = score_surf.get_size()
        score_surf = pygame.transform.smoothscale(
            score_surf, (max(1, int(w * m)), max(1, int(h * m))))
    screen.blit(score_surf, HUD_PADDING)

    lives_label = get_static_text(label_font, "Lives: ", HUD_COLOR)
    screen.blit(lives_label, (WIDTH - 150, HUD_PADDING[1]))
    start_x = WIDTH - 110
    for i in range(min(max(state.lives, 0), 10)):
        x = start_x - i * 18
        pygame.draw.circle(screen, (231, 76, 60), (x, HUD_PADDING[1] + 10), 7)


def draw_spell_bar(screen: pygame.Surface, hud_font: pygame.font.Font) -> None:
    base_x, base_y = 20, HEIGHT - 32
    screen.blit(get_static_text(hud_font, "Spells:", HUD_COLOR), (base_x, base_y - 18))
    labels = ["Up: Fire", "Down: Ice", "Left: Lightning", "Right: Arcane"]
    colors = [SPELL_HUD_COLORS["fire"], SPELL_HUD_COLORS["ice"],
              SPELL_HUD_COLORS["lightning"], SPELL_HUD_COLORS["arcane"]]
    for i, (text, color) in enumerate(zip(labels, colors)):
        screen.blit(get_static_text(hud_font, text, color),
                    (base_x + i * 190, base_y))


# --- blade/wand -------------------------------------------------------------
def draw_blade(screen: pygame.Surface, state: GameState) -> None:
    trail = state.trail
    for i, (tx, ty) in enumerate(trail):
        shade, radius = _TRAIL_SPRITES[i] if i < len(_TRAIL_SPRITES) else _TRAIL_SPRITES[-1]
        pygame.draw.circle(screen, shade, (int(tx), int(ty)), radius)
    if state.cursor:
        pygame.draw.circle(screen, BLADE_COLOR,
                           (int(state.cursor[0]), int(state.cursor[1])),
                           BLADE_RADIUS, width=2)
        pygame.draw.circle(screen, (255, 255, 255),
                           (int(state.cursor[0]), int(state.cursor[1])),
                           BLADE_RADIUS - 6, width=1)


# --- enemies ----------------------------------------------------------------
def draw_enemies(screen: pygame.Surface, state: GameState) -> None:
    for enemy in state.enemies:
        color = enemy.color
        r = int(enemy.r)
        # glow
        for i in range(4, 0, -1):
            alpha = int(40 * (1.0 - i / 4.0))
            pygame.draw.circle(screen, (*color, alpha),
                               (int(enemy.x), int(enemy.y)), r + i)
        # body
        pygame.draw.circle(screen, color, (int(enemy.x), int(enemy.y)), r)
        pygame.draw.circle(screen, (255, 255, 255, 35),
                           (int(enemy.x - r * 0.25), int(enemy.y - r * 0.25)),
                           max(2, r // 3))


# --- spells -----------------------------------------------------------------
def draw_spells(screen: pygame.Surface, state: GameState) -> None:
    for s in state.spells:
        sprite = get_spell_sprite(s.kind, s.color, int(s.r))
        screen.blit(sprite, sprite.get_rect(center=(s.x, s.y)))
        # fade trail
        for i, (tx, ty) in enumerate(s.trail):
            alpha = int(180 * (i / len(s.trail)))
            pygame.draw.circle(screen, (*s.color, alpha), (int(tx), int(ty)), max(2, int(s.r * 0.35)))


# --- main render ------------------------------------------------------------
def draw_world(screen: pygame.Surface, state: GameState,
               font: pygame.font.Font, big_font: pygame.font.Font,
               hud_font: pygame.font.Font) -> None:
    shake_x, shake_y = _shake_offset(state)
    if shake_x or shake_y:
        playfield = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        playfield.blit(get_background(), (0, 0))
        draw_ambient(playfield)
        draw_enemies(playfield, state)
        draw_spells(playfield, state)
        for p in state.particles:
            alpha = max(p.life / PARTICLE_LIFE, 0.0)
            bucket = max(0, min(3, int(alpha * 4)))
            spr = pygame.Surface((p.size * 2 + 2, p.size * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(spr, (*p.color, int(220 * alpha)),
                               (int(p.size + 1), int(p.size + 1)), int(p.size))
            playfield.blit(spr, (int(p.x) - int(p.size + 1), int(p.y) - int(p.size + 1)))
        for popup in state.popups:
            alpha = max(popup.life / POPUP_LIFE, 0.0)
            surf = font.render(popup.text, True, popup.color)
            surf.set_alpha(int(255 * alpha))
            if popup.scale != 1.0:
                w, h = surf.get_size()
                surf = pygame.transform.smoothscale(
                    surf, (max(1, int(w * popup.scale)), max(1, int(h * popup.scale))))
            _blit_centered(playfield, surf, (popup.x, popup.y))
        draw_blade(playfield, state)
        screen.blit(playfield, (shake_x, shake_y))
    else:
        screen.blit(get_background(), (0, 0))
        draw_ambient(screen)
        draw_enemies(screen, state)
        draw_spells(screen, state)
        for p in state.particles:
            alpha = max(p.life / PARTICLE_LIFE, 0.0)
            spr = pygame.Surface((p.size * 2 + 2, p.size * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(spr, (*p.color, int(220 * alpha)),
                               (int(p.size + 1), int(p.size + 1)), int(p.size))
            screen.blit(spr, (int(p.x) - int(p.size + 1), int(p.y) - int(p.size + 1)))
        for popup in state.popups:
            alpha = max(popup.life / POPUP_LIFE, 0.0)
            surf = font.render(popup.text, True, popup.color)
            surf.set_alpha(int(255 * alpha))
            if popup.scale != 1.0:
                w, h = surf.get_size()
                surf = pygame.transform.smoothscale(
                    surf, (max(1, int(w * popup.scale)), max(1, int(h * popup.scale))))
            _blit_centered(screen, surf, (popup.x, popup.y))
        draw_blade(screen, state)

    draw_hud(screen, state, hud_font, font)
    draw_spell_bar(screen, hud_font)
    draw_overlays(screen, state, font, big_font)


def draw_overlays(screen: pygame.Surface, state: GameState,
                  font: pygame.font.Font, big_font: pygame.font.Font) -> None:
    if state.cursor is None and not state.game_over:
        if camera_alive():
            hint = get_static_text(font, "Show your hand to the camera...",
                                   (255, 220, 120))
        else:
            hint = get_static_text(
                font, "CAMERA DISCONNECTED — will reconnect automatically",
                (255, 90, 90))
        _blit_centered(screen, hint, (WIDTH // 2, HEIGHT // 2))

    if state.game_over:
        screen.blit(get_dim_overlay(), (0, 0))
        cx, cy = WIDTH // 2, HEIGHT // 2
        _blit_centered(screen,
                       get_static_text(big_font, "GAME OVER", (231, 76, 60)),
                       (cx, cy - 50))
        _blit_centered(screen,
                       _FINAL_SCORE_TEXT.render(font,
                                                f"Final score: {state.score}",
                                                HUD_COLOR),
                       (cx, cy + 5))
        _blit_centered(screen,
                       get_static_text(font, "Press any key to restart",
                                       (255, 220, 120)),
                       (cx, cy + 45))


# --- lifecycle --------------------------------------------------------------
def reset_game(state: GameState) -> None:
    state.score = 0
    state.lives = START_LIVES
    state.elapsed = 0.0
    state.spawn_timer = 0.0
    state.enemies.clear()
    state.spells.clear()
    state.particles.empty()
    Particle._pool = []
    state.popups.clear()
    state.trail.clear()
    state.score_pop_timer = 0.0
    state.shake_time = 0.0
    state.game_over = False
    state.paused = False
    state.combo_timer = 0.0
    state.combo_count = 0
    state.slow_mo_timer = 0.0
    state.cast_times = {}


def handle_events(state: GameState) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                state.paused = not state.paused
                return True
            if event.key == pygame.K_p:
                state.paused = not state.paused
                return True
            if state.game_over:
                reset_game(state)
    return True


def parse_args(argv: list[str]) -> dict:
    return {
        "mute": "--mute" in argv,
        "show_fps": "--show-fps" in argv,
    }


def main(mute: bool = False, show_fps: bool = False) -> None:
    if not mute:
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=256)
    pygame.init()

    try:
        read_cursor_px()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Arcane Swipe")
        clock = pygame.time.Clock()
        sounds = init_audio(mute) if not mute else None
        font = pygame.font.SysFont(None, 32)
        big_font = pygame.font.SysFont(None, 72)
        hud_font = pygame.font.SysFont(None, SCORE_FONT_SIZE)
        fps_font = pygame.font.SysFont(None, 22)

        state = new_game_state()
        running = True
        while running:
            dt = min(clock.tick(FPS) / 1000.0, MAX_FRAME_DT)
            running = handle_events(state)
            if not state.paused and not state.game_over:
                update_state(state, dt, sounds[0] if sounds else None)
            draw_world(screen, state, font, big_font, hud_font)
            if show_fps:
                fps_surf = fps_font.render(f"{clock.get_fps():.0f} fps",
                                          True, (140, 140, 140))
                screen.blit(fps_surf, (WIDTH - 70, 10))
            pygame.display.flip()
    finally:
        release_camera()
        pygame.quit()


if __name__ == "__main__":
    main(**parse_args(sys.argv[1:]))
