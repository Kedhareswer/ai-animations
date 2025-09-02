import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os
import shutil
import static_ffmpeg

# Configure FFmpeg
static_ffmpeg.add_paths()
plt.rcParams['animation.ffmpeg_path'] = shutil.which('ffmpeg')

# Set up the dark theme and high-quality rendering
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#000000'
plt.rcParams['axes.facecolor'] = '#000000'
plt.rcParams['savefig.facecolor'] = '#000000'
plt.rcParams['figure.dpi'] = 100

class AbstractArtAnimation:
    def __init__(self, width=1920, height=1080):
        self.fig, self.ax = plt.subplots(figsize=(width/100, height/100))
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Animation parameters
        self.n_particles = 150
        self.n_waves = 8
        self.frame_count = 0
        self.total_frames = 900  # 15 seconds at 60fps
        
        # Initialize particles
        self.particles_x = np.random.uniform(-8, 8, self.n_particles)
        self.particles_y = np.random.uniform(-8, 8, self.n_particles)
        self.particle_velocities_x = np.random.uniform(-0.1, 0.1, self.n_particles)
        self.particle_velocities_y = np.random.uniform(-0.1, 0.1, self.n_particles)
        self.particle_phases = np.random.uniform(0, 2*np.pi, self.n_particles)
        
        # Wave parameters
        self.wave_phases = np.linspace(0, 2*np.pi, self.n_waves)
        
        # Color palettes for different phases
        self.color_palettes = [
            ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],  # Warm sunset
            ['#A8E6CF', '#FFD93D', '#6BCF7F', '#4D96FF', '#9B59B6'],  # Electric
            ['#FF8A80', '#82B1FF', '#B388FF', '#80CBC4', '#C5E1A5'],  # Pastel neon
            ['#FF5722', '#FF9800', '#FFC107', '#8BC34A', '#00BCD4']   # Fire gradient
        ]
        
    def get_dynamic_colors(self, progress):
        """Generate colors based on animation progress"""
        palette_idx = int(progress * len(self.color_palettes)) % len(self.color_palettes)
        return self.color_palettes[palette_idx]
    
    def create_flowing_lines(self, t):
        """Create flowing curved lines"""
        lines = []
        colors = []
        
        progress = t / self.total_frames
        current_colors = self.get_dynamic_colors(progress)
        
        for i in range(5):
            # Create flowing sine waves with varying parameters
            x = np.linspace(-10, 10, 200)
            frequency = 0.8 + 0.4 * np.sin(t * 0.02 + i)
            amplitude = 3 + 2 * np.sin(t * 0.015 + i * 0.7)
            phase = t * 0.03 + i * np.pi / 3
            
            y = amplitude * np.sin(frequency * x + phase) + 2 * np.sin(x * 0.3 + t * 0.01 + i)
            
            # Add some vertical offset variation
            y_offset = 3 * np.sin(t * 0.008 + i * 1.2)
            y += y_offset
            
            # Create line segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lines.append(segments)
            colors.append(current_colors[i % len(current_colors)])
            
        return lines, colors
    
    def update_particles(self, t):
        """Update particle positions with complex motion"""
        dt = 0.016  # 60fps
        
        # Add orbital motion around invisible attractors
        attractor_x = 4 * np.sin(t * 0.005)
        attractor_y = 3 * np.cos(t * 0.007)
        
        # Calculate forces from attractor
        dx = attractor_x - self.particles_x
        dy = attractor_y - self.particles_y
        distance = np.sqrt(dx**2 + dy**2) + 0.1  # Avoid division by zero
        
        force_strength = 0.3
        force_x = force_strength * dx / distance**2
        force_y = force_strength * dy / distance**2
        
        # Add some turbulence
        turbulence_x = 0.2 * np.sin(self.particles_x * 0.5 + t * 0.01 + self.particle_phases)
        turbulence_y = 0.2 * np.cos(self.particles_y * 0.5 + t * 0.01 + self.particle_phases)
        
        # Update velocities and positions
        self.particle_velocities_x += (force_x + turbulence_x) * dt
        self.particle_velocities_y += (force_y + turbulence_y) * dt
        
        # Add damping
        self.particle_velocities_x *= 0.98
        self.particle_velocities_y *= 0.98
        
        self.particles_x += self.particle_velocities_x * dt
        self.particles_y += self.particle_velocities_y * dt
        
        # Boundary conditions - wrap around
        self.particles_x = np.where(self.particles_x > 10, -10, self.particles_x)
        self.particles_x = np.where(self.particles_x < -10, 10, self.particles_x)
        self.particles_y = np.where(self.particles_y > 10, -10, self.particles_y)
        self.particles_y = np.where(self.particles_y < -10, 10, self.particles_y)
    
    def create_geometric_shapes(self, t):
        """Create rotating geometric patterns"""
        shapes = []
        progress = t / self.total_frames
        current_colors = self.get_dynamic_colors(progress)
        
        # Create rotating polygons
        for i in range(3):
            n_sides = 6 + i * 2
            radius = 2 + i * 0.5
            rotation = t * 0.01 * (i + 1)
            center_x = 5 * np.sin(t * 0.003 + i * 2)
            center_y = 4 * np.cos(t * 0.004 + i * 1.5)
            
            angles = np.linspace(0, 2*np.pi, n_sides + 1) + rotation
            x = center_x + radius * np.cos(angles)
            y = center_y + radius * np.sin(angles)
            
            shapes.append((x, y, current_colors[i % len(current_colors)]))
        
        return shapes
    
    def animate(self, frame):
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_facecolor('#000000')
        
        t = frame
        self.frame_count = frame
        
        # Update particles
        self.update_particles(t)
        
        # Draw flowing lines
        lines, line_colors = self.create_flowing_lines(t)
        for segments, color in zip(lines, line_colors):
            lc = LineCollection(segments, colors=color, linewidths=2, alpha=0.7)
            self.ax.add_collection(lc)
        
        # Draw particles with trails
        progress = t / self.total_frames
        current_colors = self.get_dynamic_colors(progress)
        
        # Create particle size variation
        sizes = 20 + 30 * np.sin(t * 0.05 + self.particle_phases)
        
        # Color particles based on position and time
        particle_colors = []
        for i in range(self.n_particles):
            color_idx = int((self.particles_x[i] + self.particles_y[i] + t * 0.1) * 2) % len(current_colors)
            particle_colors.append(current_colors[color_idx])
        
        self.ax.scatter(self.particles_x, self.particles_y, 
                       s=sizes, c=particle_colors, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Draw geometric shapes
        shapes = self.create_geometric_shapes(t)
        for x, y, color in shapes:
            self.ax.plot(x, y, color=color, linewidth=3, alpha=0.6)
        
        # Add central focal point with pulsing effect
        center_radius = 1 + 0.5 * np.sin(t * 0.1)
        center_alpha = 0.3 + 0.2 * np.sin(t * 0.08)
        circle = plt.Circle((0, 0), center_radius, fill=False, 
                          color=current_colors[0], linewidth=4, alpha=center_alpha)
        self.ax.add_patch(circle)
        
        return []

def main():
    # Create animation
    art_anim = AbstractArtAnimation()
    
    print("Creating abstract artistic animation...")
    print("This will take a few minutes to render at 60fps...")
    
    # Create animation
    anim = animation.FuncAnimation(
        art_anim.fig, art_anim.animate, 
        frames=art_anim.total_frames,
        interval=16.67,  # 60fps
        blit=False,
        repeat=True
    )
    
    # Set up the writer
    writer = animation.FFMpegWriter(
        fps=60,
        metadata=dict(artist='AbstractArt'),
        bitrate=5000
    )
    
    # Save the animation
    output_file = 'abstract_art_animation.mp4'
    print(f"Saving animation to {output_file}...")
    
    anim.save(output_file, writer=writer, dpi=100)
    print(f"Animation saved successfully!")
    
    # Automatically open the video
    if os.name == 'nt':  # Windows
        os.startfile(output_file)
    elif os.name == 'posix':  # macOS and Linux
        # Fix for Windows which doesn't have os.uname()
        import platform
        os.system(f'open "{output_file}"' if platform.system() == 'Darwin' else f'xdg-open "{output_file}"')
    
    plt.close()

if __name__ == "__main__":
    main()