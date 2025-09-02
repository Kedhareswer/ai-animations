# input.py
# Matplotlib 60 FPS MP4 animation (dark, modern, center-focused, no on-screen text)

import os
import sys
import platform
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
from matplotlib.colors import hsv_to_rgb
import static_ffmpeg

# ---------- CONFIG ----------
FPS = 60
DURATION_SEC = 12  # adjust for pacing
FRAMES = FPS * DURATION_SEC
OUTFILE = "animation.mp4"
POINTS = 1400  # particle count (balance detail vs. render time)

# ---------- STYLE ----------
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0a0a0f",
    "axes.facecolor":   "#0a0a0f",
    "axes.edgecolor":   "#0a0a0f",
    "savefig.facecolor":"#0a0a0f",
    "savefig.edgecolor":"#0a0a0f",
    "lines.antialiased": True,
})

# 16:9 canvas for YouTube; high DPI for crisp visuals
fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)

# ---------- PARTICLE FIELD (spiral vortex) ----------
rng = np.random.default_rng(42)
golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.39996

i = np.arange(1, POINTS + 1)
r0 = np.sqrt(i) / np.sqrt(POINTS)  # base radial position 0..1
theta0 = i * golden_angle

# Per-particle modulation for organic motion
phase = rng.uniform(0, 2*np.pi, size=POINTS)
twist = rng.normal(0.0, 0.18, size=POINTS)  # slight theta variation

# Size: larger near center, smaller outward (keeps focus)
sizes = 6 + 38 * (1 - r0) ** 1.5

# Initialize at base positions (they’ll be animated)
x_init = r0 * np.cos(theta0)
y_init = r0 * np.sin(theta0)
scat = ax.scatter(
    x_init, y_init,
    s=sizes,
    c=np.zeros((POINTS, 3)),
    alpha=0.92,
    edgecolors="none"
)

# ---------- CENTER SHAPE (smooth Lissajous pulse) ----------
t_curve = np.linspace(0, 2*np.pi, 900)
a, b = 3, 2  # Lissajous frequencies
amp_base = 0.22
x_curve = amp_base * np.sin(a * t_curve)
y_curve = amp_base * np.sin(b * t_curve)
line, = ax.plot(x_curve, y_curve, lw=2.5, alpha=0.95)

# ---------- PULSE RINGS (neon glow, layered) ----------
ring_groups = []
num_rings = 3
ring_base = 0.24
ring_gap = 0.18
glow_widths = [4, 8, 14]
glow_alphas = [0.38, 0.18, 0.08]

for k in range(num_rings):
    layers = []
    for j, lw in enumerate(glow_widths):
        c = Circle(
            (0, 0),
            radius=ring_base + k * ring_gap,
            fill=False, lw=lw,
            alpha=glow_alphas[j]
        )
        ax.add_patch(c)
        layers.append(c)
    ring_groups.append(layers)

# ---------- ANIMATION UPDATE ----------
def update(frame):
    t = frame / FPS  # seconds elapsed

    # --- Particle motion ---
    # Gentle breathing, swirl, and radial waves for hypnotic flow
    breathe = 0.80 + 0.18 * np.sin(2*np.pi*(0.45*t) + phase)
    radial = r0 * breathe

    rotation = 2*np.pi * (0.07 * t)  # slow global rotation
    theta = theta0 + rotation + 0.33 * np.sin(2*np.pi*(0.31*t) + r0*4) + twist

    x = radial * np.cos(theta)
    y = radial * np.sin(theta)
    offsets = np.column_stack([x, y])
    scat.set_offsets(offsets)

    # Neon hues: cycle smoothly through the spectrum with angle/time coupling
    hue = (theta0/(2*np.pi) + 0.12*np.sin(2*np.pi*0.25*t) + 0.08*t) % 1.0
    sat = 0.85 + 0.15 * (1 - r0)         # more saturated near center
    val = 0.80 + 0.20 * np.sin(2*np.pi*0.5*t + r0*6 + phase)
    val = np.clip(val, 0.6, 1.0)
    colors = hsv_to_rgb(np.column_stack([hue, sat, val]))
    scat.set_facecolors(colors)

    # --- Center Lissajous ---
    amp = amp_base * (1.0 + 0.15 * np.sin(2*np.pi*0.5*t))
    phi = 2*np.pi * (0.25 * t)
    x_c = amp * np.sin(a * t_curve + phi)
    y_c = amp * np.sin(b * t_curve)
    line.set_data(x_c, y_c)
    line.set_linewidth(2.5 + 1.2 * np.sin(2*np.pi*0.5*t + 0.3))
    line_col = hsv_to_rgb([(0.85 + 0.20*np.sin(2*np.pi*0.12*t)) % 1.0, 1.0, 1.0])
    line.set_color(line_col)

    # --- Pulse rings (neon) ---
    for k, layers in enumerate(ring_groups):
        base_r = ring_base + k * ring_gap
        R = base_r * (1.0 + 0.12 * np.sin(2*np.pi*(0.5*t) + k))
        ring_hue = (0.60 + 0.25 * np.sin(2*np.pi*(0.10*t) + k)) % 1.0
        ring_col = hsv_to_rgb([ring_hue, 1.0, 1.0])
        for j, circ in enumerate(layers):
            # slight expansion per glow layer for a soft aura
            circ.set_radius(R + j * 0.004)
            circ.set_edgecolor(ring_col)

    # Return artists for blitting
    artists = [scat, line]
    for grp in ring_groups:
        artists.extend(grp)
    return artists

def init():
    return update(0)

# ---------- RENDER ----------
import matplotlib.pyplot as plt

static_ffmpeg.add_paths()

# Add FFmpeg path configuration
plt.rcParams['animation.ffmpeg_path'] = shutil.which('ffmpeg')

ani = FuncAnimation(
    fig, update, frames=FRAMES, init_func=init,
    blit=True, interval=1000 / FPS
)

metadata = {"artist": "Matplotlib", "title": "Center-Focused Neon Vortex"}
writer = FFMpegWriter(fps=FPS, metadata=metadata, codec="libx264")

print(f"Rendering {FRAMES} frames at {FPS} fps...")
ani.save(OUTFILE, writer=writer)
plt.close(fig)
print(f"Saved: {OUTFILE}")

# Auto-open the rendered video for quick iteration
def open_file(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception as e:
        print(f"Could not auto-open the video: {e}")

open_file(OUTFILE)
