import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

# Style setup
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.linewidth": 0.5,
})

# Parameters
n_points = 1200
golden_angle = np.pi * (3 - np.sqrt(5))  # phyllotaxis
frames = 600
fps = 60

# Spiral coordinates (phyllotaxis)
indices = np.arange(0, n_points)
theta = indices * golden_angle
r = np.sqrt(indices)

x = r * np.cos(theta)
y = r * np.sin(theta)

# Normalize for better layout
x /= max(abs(x))
y /= max(abs(y))

# Figure setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal")
ax.axis("off")

# Scatter plot
scatter = ax.scatter([], [], s=2, alpha=0.85)

# Shockwave circles
shock_circles = [Circle((0, 0), radius=0.0, fill=False, lw=1.5, alpha=0.0, ec="cyan") for _ in range(3)]
for c in shock_circles:
    ax.add_patch(c)

def hsv_cycle(t, base_hue=0.6):
    """Return RGB color cycling through HSV (magenta–cyan–blue)."""
    return plt.cm.hsv((base_hue + 0.2 * np.sin(t)) % 1.0)

def init():
    scatter.set_offsets(np.empty((0, 2)))
    scatter.set_facecolors([])
    for c in shock_circles:
        c.set_radius(0)
        c.set_alpha(0.0)
    return [scatter] + shock_circles

def animate(frame):
    # Spiral breathing (scale with sinusoidal pulse)
    scale = 0.95 + 0.05 * np.sin(2 * np.pi * frame / 120)
    angle_offset = 0.002 * frame
    xs = scale * (x * np.cos(angle_offset) - y * np.sin(angle_offset))
    ys = scale * (x * np.sin(angle_offset) + y * np.cos(angle_offset))

    # Color cycling
    colors = [hsv_cycle(0.002 * frame + i * 0.002) for i in indices]
    
    scatter.set_offsets(np.c_[xs, ys])
    scatter.set_facecolors(colors)

    # Shock rings: pulsing from center
    for i, c in enumerate(shock_circles):
        t = (frame + i * 200) % 600 / 600
        radius = t * 1.2
        alpha = 1.0 - t
        c.set_radius(radius)
        c.set_alpha(alpha * 0.5)
        c.set_edgecolor(hsv_cycle(t + 0.5))

    return [scatter] + shock_circles

anim = FuncAnimation(fig, animate, frames=frames, init_func=init, interval=1000/fps, blit=True)

# Save to GIF
writer = PillowWriter(fps=fps, metadata={"artist": "Matplotlib"})
anim.save("neon_vortex_pulse_hook.gif", writer=writer, dpi=100)

plt.close(fig)
