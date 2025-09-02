# input.py
# New Matplotlib 60 FPS MP4 animation (dark, modern, center-focused, no on-screen text)
# Entertainment-first: smooth pacing, vibrant neon palette, center composition.
# Exports animation.mp4 and attempts to open it automatically.

import os
import sys
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import hsv_to_rgb
import static_ffmpeg

# Configure FFmpeg path
static_ffmpeg.add_paths()
plt.rcParams['animation.ffmpeg_path'] = shutil.which('ffmpeg')

# ---------- CONFIG ----------
FPS = 60
DURATION_SEC = 10        # 10 seconds (600 frames)
FRAMES = FPS * DURATION_SEC
OUTFILE = "animation_01.mp4"
SEED = 1337

# ---------- STYLE ----------
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#07060a",
    "axes.facecolor":   "#07060a",
    "savefig.facecolor":"#07060a",
    "savefig.edgecolor":"#07060a",
    "lines.antialiased": True,
})

# Create figure with centered composition (16:9)
fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-1.12, 1.12)
ax.set_ylim(-0.63, 0.63)  # narrower vertically to visually center for 16:9

rng = np.random.default_rng(SEED)

# ---------- BACKGROUND SUBTLE VIGNETTE IMAGE ----------
# Create a smooth radial gradient background as an RGBA image
res = 1200  # image resolution for gradient (keeps memory reasonable)
xx = np.linspace(-1.12, 1.12, res)
yy = np.linspace(-0.63, 0.63, res)
X, Y = np.meshgrid(xx, yy)
R = np.sqrt((X/1.12)**2 + (Y/0.63)**2)
bg_val = 0.06 + 0.28 * (1 - np.clip(R, 0, 1))**1.8
bg_rgb = np.dstack([bg_val * 0.06, bg_val * 0.09, bg_val * 0.14])  # cool bluish
ax.imshow(bg_rgb, extent=[-1.12,1.12,-0.63,0.63], origin='lower', interpolation='bilinear', zorder=0)

# ---------- ORBITING POLYGONS (central cluster) ----------
NUM_POLYGONS = 22
polygons = []
base_radii = np.linspace(0.06, 0.40, NUM_POLYGONS)
sides = rng.integers(3, 9, size=NUM_POLYGONS)  # triangles to octagons
angles0 = rng.uniform(0, 2*np.pi, size=NUM_POLYGONS)
rot_speed = rng.uniform(-0.9, 0.9, size=NUM_POLYGONS)  # rotation per second
orbit_speed = rng.uniform(0.05, 0.45, size=NUM_POLYGONS)  # revolutions per second
scale_jitter = rng.uniform(0.85, 1.25, size=NUM_POLYGONS)

for i in range(NUM_POLYGONS):
    poly = RegularPolygon((0,0), numVertices=int(sides[i]),
                          radius=base_radii[i]*scale_jitter[i],
                          orientation=angles0[i],
                          fill=True, ec='none', alpha=0.0, zorder=4)
    polygons.append(poly)
    ax.add_patch(poly)

poly_collection = PatchCollection(polygons, match_original=True)

# ---------- RADIATING PARTICLES (soft motion blur feel) ----------
PARTICLES = 900
theta = rng.uniform(0, 2*np.pi, size=PARTICLES)
r = rng.random(PARTICLES) ** 1.8  # denser near center
x = r * np.cos(theta)
y = r * np.sin(theta) * (0.63/1.12)  # correct aspect distortion
sizes = 3.5 + 20.0 * (1 - r)  # bigger near center
hue_base = rng.uniform(0, 1, size=PARTICLES)
sat = 0.9 - 0.25 * r
val = 0.85 - 0.25 * r
colors = hsv_to_rgb(np.column_stack([hue_base, sat, val]))
scat = ax.scatter(x, y, s=sizes, c=colors, alpha=0.9, linewidths=0, zorder=2)

# ---------- CONCENTRIC NEON ARC GROUP (pulsing rings) ----------
ring_patches = []
NUM_ARCS = 5
for k in range(NUM_ARCS):
    c = Circle((0,0), radius=0.08 + k*0.07, fill=False, lw=2.0 + k*1.6, alpha=0.0, zorder=3)
    ax.add_patch(c)
    ring_patches.append(c)

# ---------- SOFT CENTER GLOW (layered circles) ----------
glow_layers = []
glow_radii = [0.02, 0.045, 0.085]
for r0 in glow_radii:
    c = Circle((0,0), radius=r0, fill=True, ec='none', alpha=0.0, zorder=5)
    ax.add_patch(c)
    glow_layers.append(c)

# ---------- CAMERA ZOOM / SUBTLE PANNING SETTINGS ----------
# We'll create a smooth zoom-in and slight vertical float to mimic cinematic framing.
zoom_base = 1.0
zoom_strength = 0.06  # max zoom
float_amp = 0.03      # vertical float amplitude

# ---------- UPDATE FUNCTION ----------
def update(frame):
    t = frame / FPS  # seconds elapsed
    # Time-based easing for intro/outro presence (soft in/out)
    # Using a smoothstep for first and last 1.25 seconds to avoid abruptness
    fade_in = np.clip((t / 1.25), 0, 1)
    fade_in = fade_in * fade_in * (3 - 2 * fade_in)  # smoothstep
    fade_out = np.clip(((DURATION_SEC - t) / 1.25), 0, 1)
    fade_out = fade_out * fade_out * (3 - 2 * fade_out)
    presence = fade_in * fade_out

    # --- Update polygons: orbit, spin, color, alpha ---
    poly_colors = []
    for i, poly in enumerate(polygons):
        # orbit radius with gentle breathing
        orbit_r = base_radii[i] * (1.0 + 0.06 * np.sin(2*np.pi*(0.6*t + i*0.07)))
        ang = angles0[i] + 2*np.pi*orbit_speed[i]*t  # revolution
        px = orbit_r * np.cos(ang)
        py = orbit_r * np.sin(ang) * (0.63/1.12)
        # Update polygon center - fixed to avoid TypeError
        # poly.xy = None  # This line causes the error
        poly._xy = (px, py)  # set center (works with RegularPolygon internals)
        # rotation animation (orientation)
        ori = angles0[i] + rot_speed[i] * t * 0.8 + 0.25 * np.sin(2*np.pi*0.35*t + i)
        poly._orientation = ori

        # color hue shifts, more vivid near center (lower i)
        hue = (0.62 + 0.30 * np.sin(2*np.pi*(0.12*t + i*0.06))) % 1.0
        sat_local = 0.9 - 0.02 * i
        val_local = 0.6 + 0.45 * (1 - base_radii[i]/0.5)
        col = hsv_to_rgb([hue, sat_local, val_local])
        poly.set_facecolor(col)
        # alpha depends on presence and distance to center for cinematic focus
        alpha = presence * (0.28 + 0.72 * (1 - base_radii[i]/0.5)**1.6)
        # slight strobe on select shapes (very subtle) to add life
        alpha *= 0.88 + 0.12 * np.sin(2*np.pi*(0.9*t + i*0.15))
        poly.set_alpha(np.clip(alpha, 0, 1))
        poly_colors.append(col)

    # --- Particles: subtle radial flow outward with curl noise ---
    # moving particles by combining circular flow + radial drift
    curl = 0.55 * np.sin(2*np.pi*(0.22*t) + theta*3.1)
    radial_flow = 0.14 * np.sin(2*np.pi*(0.35*t) + 2*theta) * (1 - r)
    # compute new positions using complex numbers for smooth rotation-like flow
    x_new = x * (1 + radial_flow) - y * (0.0025*curl)
    y_new = y * (1 + radial_flow) + x * (0.0025*curl)
    # tiny jitter for organic feel
    jitter = 0.0008 * np.sin(2*np.pi*(0.9*t) + theta*5.3)
    x_disp = x_new + jitter * np.cos(theta*2.7 + t*1.2)
    y_disp = y_new + jitter * np.sin(theta*1.9 + t*0.8)
    offsets = np.column_stack([x_disp, y_disp])
    scat.set_offsets(offsets)

    # update particle colors slowly cycling hue
    hue_variation = (hue_base + 0.08*np.sin(2*np.pi*(0.07*t) + r*8.0) + 0.02*t) % 1.0
    colors = hsv_to_rgb(np.column_stack([hue_variation, sat, np.clip(val + 0.12*np.sin(2*np.pi*(0.35*t) + r*6.0), 0.45, 1.0)]))
    # fade particle alpha towards edges for focus
    alphas = np.clip(0.95 * presence * (0.45 + 0.55*(1 - r)**1.2), 0, 1)
    scat.set_facecolors(colors)
    scat.set_alpha(alphas)

    # --- Rings (neon arcs): pulse and rotate slightly ---
    for k, ring in enumerate(ring_patches):
        base = 0.08 + k*0.07
        pulse = 1.0 + 0.18 * np.sin(2*np.pi*(0.6*t + k*0.15))
        ring.set_radius(base * pulse)
        hue_ring = (0.02 + 0.18*k + 0.15*np.sin(2*np.pi*(0.08*t + k*0.12))) % 1.0
        col = hsv_to_rgb([hue_ring, 0.98, 1.0])
        ring.set_edgecolor(col)
        # more outer rings are fainter
        ring.set_alpha(np.clip(presence * (0.48 - 0.08*k), 0.0, 0.9))
        ring.set_linewidth(1.6 + 1.6*(NUM_ARCS - k))

    # --- Center Glow layers ---
    for j, g in enumerate(glow_layers):
        baseg = glow_radii[j]
        g_hue = (0.72 + 0.08 * np.sin(2*np.pi*(0.2*t + j*0.3))) % 1.0
        g_col = hsv_to_rgb([g_hue, 0.95, 1.0])
        g.set_facecolor(g_col)
        g_alpha = presence * (0.28 if j==2 else 0.52) * (1.0 - 0.22*j) * (1.0 + 0.12*np.sin(2*np.pi*(0.9*t + j)))
        g.set_alpha(np.clip(g_alpha, 0, 0.9))
        # breathing radius change
        g.set_radius(baseg * (1.0 + 0.10 * np.sin(2*np.pi*(0.5*t + j*0.4))))

    # --- Camera: subtle zoom + vertical float (applied by scaling artist transforms) ---
    # We simulate camera by scaling the entire axes limits around center
    zoom = 1.0 - zoom_strength * np.sin(np.pi * (0.5 * t / DURATION_SEC)) * (0.5 + 0.5*np.sin(2*np.pi*0.08*t))
    # gentle ease in/out for zoom (using presence)
    zoom_effect = 1.0 - (zoom_strength * (1 - presence) * 0.9) + (zoom_strength * (1 - presence) * 0.1)
    total_zoom = zoom * zoom_effect
    # compute new limits so center stays centered (with slight vertical float)
    v_offset = float_amp * np.sin(2*np.pi*(0.12*t)) * (1 - 0.5*(1-presence))
    xlim = (-1.12*total_zoom, 1.12*total_zoom)
    ylim = (-0.63*total_zoom + v_offset, 0.63*total_zoom + v_offset)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Gather artists to return (for blitting)
    artists = [scat]
    artists.extend(polygons)
    artists.extend(ring_patches)
    artists.extend(glow_layers)
    return artists

def init():
    # initialize alpha to zero so fade-in is smooth via update
    scat.set_alpha(0.0)
    for p in polygons:
        p.set_alpha(0.0)
    for r in ring_patches:
        r.set_alpha(0.0)
    for g in glow_layers:
        g.set_alpha(0.0)
    return [scat] + polygons + ring_patches + glow_layers

# ---------- ANIMATION ----------
ani = FuncAnimation(fig, update, frames=FRAMES, init_func=init, blit=True, interval=1000/FPS)

# ---------- SAVE ----------
metadata = {"artist": "Matplotlib", "title": "Neon Orbit Cluster"}
writer = FFMpegWriter(fps=FPS, metadata=metadata, codec="libx264", bitrate=18000)

print(f"Rendering {FRAMES} frames at {FPS} fps to {OUTFILE} ... (this may take a while)")
ani.save(OUTFILE, writer=writer)
plt.close(fig)
print(f"Saved: {OUTFILE}")

# ---------- Auto-open the rendered file ----------
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
