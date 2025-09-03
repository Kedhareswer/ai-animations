import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import colorsys

# Set up the dark theme and parameters
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['savefig.facecolor'] = 'black'

# Animation parameters
FPS = 60
DURATION = 8  # seconds
TOTAL_FRAMES = FPS * DURATION

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Phyllotaxis spiral parameters
GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))  # Golden angle in radians
N_PARTICLES = 800

# Generate phyllotaxis positions
def generate_phyllotaxis(n_points, scale=1.0):
    """Generate phyllotaxis spiral points"""
    indices = np.arange(n_points)
    angles = indices * GOLDEN_ANGLE
    radii = np.sqrt(indices) * scale * 0.03
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y, angles, radii

# Color cycling function
def get_color_cycle(t, base_hue_offset=0):
    """Generate cycling HSV colors"""
    hue = (t * 0.3 + base_hue_offset) % 1.0  # Cycle through hues
    saturation = 0.9
    value = 0.9
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return rgb

# Easing functions
def ease_in_out_cubic(t):
    """Smooth easing function"""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def pulse_function(t, frequency=1.0, phase=0):
    """Breathing pulse function"""
    return 0.5 + 0.5 * np.sin(2 * np.pi * frequency * t + phase)

# Initialize particles
base_x, base_y, base_angles, base_radii = generate_phyllotaxis(N_PARTICLES)
particles = ax.scatter([], [], s=[], c=[], alpha=0.8, edgecolors='none')

# Initialize shock rings (multiple rings for layered effect)
N_RINGS = 5
shock_rings = []
for i in range(N_RINGS):
    ring = Circle((0, 0), 0, fill=False, linewidth=3, alpha=0)
    ax.add_patch(ring)
    shock_rings.append(ring)

def animate(frame):
    """Animation function"""
    t = frame / TOTAL_FRAMES
    
    # Rotation and breathing parameters
    rotation = t * 4 * np.pi  # Multiple rotations over duration
    breath_scale = 0.8 + 0.4 * pulse_function(t, frequency=2.0)  # Breathing effect
    spiral_speed = t * 3.0  # Spiral evolution
    
    # Transform particle positions
    # Add spiral evolution and rotation
    evolved_angles = base_angles + spiral_speed
    evolved_radii = base_radii * breath_scale
    
    # Apply rotation
    rotated_angles = evolved_angles + rotation
    x = evolved_radii * np.cos(rotated_angles)
    y = evolved_radii * np.sin(rotated_angles)
    
    # Particle sizes with pulsing effect
    base_sizes = 20 + 15 * np.sin(base_radii * 10 + t * 8 * np.pi)
    sizes = base_sizes * (0.7 + 0.3 * pulse_function(t, frequency=3.0))
    
    # Color cycling for particles
    colors = []
    for i in range(len(x)):
        # Each particle has slight hue offset based on its position
        hue_offset = base_radii[i] * 5 + base_angles[i] * 0.1
        color = get_color_cycle(t, hue_offset)
        colors.append(color)
    
    # Update particles
    particles.set_offsets(np.column_stack((x, y)))
    particles.set_sizes(sizes)
    particles.set_color(colors)
    
    # Animate shock rings
    for i, ring in enumerate(shock_rings):
        # Stagger ring timing
        ring_phase = i * 0.2
        ring_t = (t + ring_phase) % 1.0
        
        # Ring expansion with easing
        max_radius = 1.8
        ring_progress = ease_in_out_cubic(ring_t)
        radius = ring_progress * max_radius
        
        # Ring opacity (fade in, peak, fade out)
        if ring_t < 0.3:
            alpha = ring_t / 0.3
        elif ring_t > 0.8:
            alpha = (1.0 - ring_t) / 0.2
        else:
            alpha = 1.0
        
        # Ring color cycling
        ring_color = get_color_cycle(t + ring_phase * 0.5, 0.3)
        
        # Ring thickness variation
        thickness = 4 - i * 0.5
        
        ring.set_radius(radius)
        ring.set_alpha(alpha * 0.6)  # Overall transparency
        ring.set_edgecolor(ring_color)
        ring.set_linewidth(thickness)
    
    # Add central glow effect
    center_glow_intensity = pulse_function(t, frequency=4.0)
    center_color = get_color_cycle(t, 0.5)
    
    # Create a bright center point
    center_scatter = ax.scatter([0], [0], 
                              s=[100 + 50 * center_glow_intensity], 
                              c=[center_color], 
                              alpha=0.9,
                              edgecolors='white',
                              linewidths=2)
    
    return [particles] + shock_rings + [center_scatter]

# Create and run animation
print("Creating Neon Vortex Pulse Hook animation...")
print(f"Duration: {DURATION} seconds at {FPS} FPS")
print(f"Total frames: {TOTAL_FRAMES}")

anim = animation.FuncAnimation(
    fig, animate, frames=TOTAL_FRAMES, 
    interval=1000/FPS, blit=False, repeat=True
)

# Set up Pillow writer for GIF
writer = animation.PillowWriter(fps=FPS)

# Save animation
filename = "neon_vortex_pulse_hook.gif"
print(f"Saving animation as {filename}...")
anim.save(filename, writer=writer, dpi=5)
print("Animation saved successfully!")

# Display the animation
plt.tight_layout()
plt.show()
