import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection, EllipseCollection
from matplotlib.patches import Circle
import os
import shutil
import platform
import static_ffmpeg
from scipy.spatial.distance import cdist

# Set up the dark theme and highest quality rendering
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#000000'
plt.rcParams['axes.facecolor'] = '#000000'
plt.rcParams['savefig.facecolor'] = '#000000'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['animation.html'] = 'html5'

class NaturalAbstractAnimation:
    def __init__(self, width=1920, height=1080):
        self.fig, self.ax = plt.subplots(figsize=(width/100, height/100))
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-9, 9)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Animation parameters
        self.frame_count = 0
        self.total_frames = 1200  # 20 seconds at 60fps
        
        # Complex particle systems
        self.init_particle_systems()
        self.init_flow_fields()
        self.init_organic_structures()
        
        # Natural color palettes inspired by nature
        self.color_palettes = [
            # Aurora Borealis
            ['#00FF87', '#60EFFF', '#6B73FF', '#9F40FF', '#C740CC'],
            # Deep Ocean
            ['#0077BE', '#00A8CC', '#00CED1', '#40E0D0', '#7FFFD4'],
            # Sunset Sky
            ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#4D9DE0'],
            # Forest Mystique
            ['#355E3B', '#50C878', '#90EE90', '#00FF7F', '#ADFF2F'],
            # Cosmic Nebula
            ['#8A2BE2', '#DA70D6', '#FF1493', '#FF69B4', '#FFC0CB'],
            # Fire and Ice
            ['#FF4500', '#FF6347', '#00CED1', '#48CAE4', '#ADE8F4']
        ]
        
    def init_particle_systems(self):
        """Initialize multiple complex particle systems"""
        # Main particle swarm (boids-like behavior)
        self.n_main_particles = 200
        self.main_particles = np.random.uniform(-10, 10, (self.n_main_particles, 2))
        self.main_velocities = np.random.uniform(-0.5, 0.5, (self.n_main_particles, 2))
        self.main_accelerations = np.zeros((self.n_main_particles, 2))
        self.main_phases = np.random.uniform(0, 2*np.pi, self.n_main_particles)
        self.main_lifetimes = np.random.uniform(100, 400, self.n_main_particles)
        self.main_ages = np.random.uniform(0, 400, self.n_main_particles)
        
        # Secondary particle trails
        self.n_trail_particles = 500
        self.trail_particles = np.random.uniform(-8, 8, (self.n_trail_particles, 2))
        self.trail_velocities = np.random.uniform(-0.2, 0.2, (self.n_trail_particles, 2))
        self.trail_ages = np.random.uniform(0, 200, self.n_trail_particles)
        
        # Micro particles for texture
        self.n_micro_particles = 800
        self.micro_particles = np.random.uniform(-12, 12, (self.n_micro_particles, 2))
        self.micro_phases = np.random.uniform(0, 2*np.pi, self.n_micro_particles)
        
    def init_flow_fields(self):
        """Initialize complex flow field systems"""
        # Create multiple overlapping flow fields
        self.flow_resolution = 40
        x = np.linspace(-12, 12, self.flow_resolution)
        y = np.linspace(-9, 9, int(self.flow_resolution * 0.75))
        self.flow_x, self.flow_y = np.meshgrid(x, y)
        
        # Flow field parameters
        self.flow_scales = [0.8, 1.2, 0.5, 1.5]
        self.flow_speeds = [0.02, 0.015, 0.025, 0.01]
        self.flow_offsets = [0, np.pi/3, 2*np.pi/3, np.pi]
        
    def init_organic_structures(self):
        """Initialize organic, plant-like structures"""
        self.n_branches = 12
        self.branch_points = []
        self.branch_generations = []
        
        for i in range(self.n_branches):
            # Create fractal-like branching structures
            root_angle = i * 2 * np.pi / self.n_branches
            root_x = 6 * np.cos(root_angle)
            root_y = 4 * np.sin(root_angle)
            
            branch_system = self.generate_branch_system(root_x, root_y, root_angle, 0, 4)
            self.branch_points.append(branch_system)
    
    def generate_branch_system(self, x, y, angle, generation, max_gen):
        """Generate fractal branching patterns"""
        if generation > max_gen:
            return []
        
        branch_length = 2.0 * (0.7 ** generation)
        points = [(x, y)]
        
        # Create main branch
        for i in range(8):
            t = i / 7.0
            new_x = x + branch_length * t * np.cos(angle)
            new_y = y + branch_length * t * np.sin(angle)
            points.append((new_x, new_y))
        
        # Add sub-branches
        if generation < max_gen:
            branch_x = x + branch_length * 0.6 * np.cos(angle)
            branch_y = y + branch_length * 0.6 * np.sin(angle)
            
            # Left branch
            left_angle = angle + np.pi/4 + np.random.uniform(-0.2, 0.2)
            points.extend(self.generate_branch_system(branch_x, branch_y, left_angle, generation + 1, max_gen))
            
            # Right branch
            right_angle = angle - np.pi/4 + np.random.uniform(-0.2, 0.2)
            points.extend(self.generate_branch_system(branch_x, branch_y, right_angle, generation + 1, max_gen))
        
        return points
    
    def calculate_flow_field(self, t):
        """Calculate complex, multi-layered flow field"""
        combined_flow_x = np.zeros_like(self.flow_x)
        combined_flow_y = np.zeros_like(self.flow_y)
        
        for scale, speed, offset in zip(self.flow_scales, self.flow_speeds, self.flow_offsets):
            # Create multiple overlapping noise patterns
            noise_x = np.sin(scale * self.flow_x + t * speed + offset) * np.cos(scale * self.flow_y + t * speed * 0.7)
            noise_y = np.cos(scale * self.flow_x + t * speed + offset) * np.sin(scale * self.flow_y + t * speed * 0.8)
            
            # Add spiral components
            center_dist = np.sqrt(self.flow_x**2 + self.flow_y**2)
            spiral_strength = np.exp(-center_dist / 8)
            spiral_angle = np.arctan2(self.flow_y, self.flow_x) + t * 0.005
            
            spiral_x = -spiral_strength * np.sin(spiral_angle) * 0.5
            spiral_y = spiral_strength * np.cos(spiral_angle) * 0.5
            
            combined_flow_x += noise_x + spiral_x
            combined_flow_y += noise_y + spiral_y
        
        return combined_flow_x, combined_flow_y
    
    def update_main_particles(self, t, flow_x, flow_y):
        """Update main particles with complex flocking behavior"""
        # Reset accelerations
        self.main_accelerations.fill(0)
        
        # Calculate distances between particles
        distances = cdist(self.main_particles, self.main_particles)
        
        for i in range(self.n_main_particles):
            # Age particles
            self.main_ages[i] += 1
            if self.main_ages[i] > self.main_lifetimes[i]:
                # Respawn particle
                self.main_particles[i] = np.random.uniform(-10, 10, 2)
                self.main_velocities[i] = np.random.uniform(-0.5, 0.5, 2)
                self.main_ages[i] = 0
                self.main_lifetimes[i] = np.random.uniform(100, 400)
            
            # Flocking forces
            neighbors = distances[i] < 2.0
            neighbor_count = np.sum(neighbors) - 1  # Exclude self
            
            if neighbor_count > 0:
                # Separation
                close_neighbors = distances[i] < 1.0
                if np.sum(close_neighbors) > 1:
                    separation = self.main_particles[i] - np.mean(self.main_particles[close_neighbors], axis=0)
                    separation_norm = np.linalg.norm(separation)
                    if separation_norm > 0:
                        self.main_accelerations[i] += 0.3 * separation / separation_norm
                
                # Alignment
                neighbor_velocities = self.main_velocities[neighbors]
                avg_velocity = np.mean(neighbor_velocities, axis=0)
                self.main_accelerations[i] += 0.1 * (avg_velocity - self.main_velocities[i])
                
                # Cohesion
                neighbor_positions = self.main_particles[neighbors]
                center_of_mass = np.mean(neighbor_positions, axis=0)
                self.main_accelerations[i] += 0.05 * (center_of_mass - self.main_particles[i])
            
            # Flow field influence
            grid_x = int(np.clip((self.main_particles[i, 0] + 12) / 24 * (self.flow_resolution - 1), 0, self.flow_resolution - 1))
            grid_y = int(np.clip((self.main_particles[i, 1] + 9) / 18 * (len(flow_y) - 1), 0, len(flow_y) - 1))
            
            flow_force_x = flow_x[grid_y, grid_x] * 0.2
            flow_force_y = flow_y[grid_y, grid_x] * 0.2
            self.main_accelerations[i] += [flow_force_x, flow_force_y]
            
            # Attractor points (like planets)
            attractor1 = np.array([6 * np.sin(t * 0.003), 4 * np.cos(t * 0.002)])
            attractor2 = np.array([-5 * np.cos(t * 0.004), 3 * np.sin(t * 0.003)])
            
            for attractor in [attractor1, attractor2]:
                to_attractor = attractor - self.main_particles[i]
                dist_to_attractor = np.linalg.norm(to_attractor)
                if dist_to_attractor > 0:
                    force_strength = 0.8 / (dist_to_attractor + 1)**2
                    self.main_accelerations[i] += force_strength * to_attractor / dist_to_attractor
        
        # Update velocities and positions
        self.main_velocities += self.main_accelerations * 0.016  # 60fps timestep
        self.main_velocities *= 0.95  # Damping
        
        # Limit velocity
        speeds = np.linalg.norm(self.main_velocities, axis=1)
        max_speed = 1.5
        too_fast = speeds > max_speed
        self.main_velocities[too_fast] = (self.main_velocities[too_fast].T * max_speed / speeds[too_fast]).T
        
        self.main_particles += self.main_velocities * 0.016
        
        # Boundary conditions - soft wrap
        for i in range(self.n_main_particles):
            if abs(self.main_particles[i, 0]) > 11:
                self.main_particles[i, 0] = np.sign(self.main_particles[i, 0]) * -10.5
            if abs(self.main_particles[i, 1]) > 8:
                self.main_particles[i, 1] = np.sign(self.main_particles[i, 1]) * -7.5
    
    def update_trail_particles(self, t):
        """Update trailing particles with organic movement"""
        # Organic wave motion
        wave1 = 0.3 * np.sin(0.5 * self.trail_particles[:, 0] + t * 0.02)
        wave2 = 0.2 * np.cos(0.3 * self.trail_particles[:, 1] + t * 0.015)
        
        self.trail_velocities[:, 0] = wave1 + 0.1 * np.sin(t * 0.01 + self.trail_particles[:, 1] * 0.2)
        self.trail_velocities[:, 1] = wave2 + 0.1 * np.cos(t * 0.008 + self.trail_particles[:, 0] * 0.3)
        
        self.trail_particles += self.trail_velocities * 0.016
        
        # Age and respawn
        self.trail_ages += 1
        expired = self.trail_ages > 200
        self.trail_particles[expired] = np.random.uniform(-8, 8, (np.sum(expired), 2))
        self.trail_ages[expired] = 0
    
    def get_natural_colors(self, t):
        """Get evolving natural color palette"""
        progress = (t / self.total_frames) % 1.0
        palette_transition = progress * len(self.color_palettes)
        palette_idx = int(palette_transition) % len(self.color_palettes)
        next_palette_idx = (palette_idx + 1) % len(self.color_palettes)
        
        # Smooth transition between palettes
        blend_factor = palette_transition - int(palette_transition)
        
        current_palette = self.color_palettes[palette_idx]
        next_palette = self.color_palettes[next_palette_idx]
        
        return current_palette, next_palette, blend_factor
    
    def draw_organic_branches(self, t):
        """Draw evolving organic branch structures"""
        current_palette, _, _ = self.get_natural_colors(t)
        
        for i, branch_points in enumerate(self.branch_points):
            if len(branch_points) < 2:
                continue
                
            # Animate branch growth and sway
            growth_phase = (t * 0.005 + i * 0.3) % (2 * np.pi)
            sway_strength = 0.3 * np.sin(growth_phase)
            
            x_coords = []
            y_coords = []
            
            for j, (x, y) in enumerate(branch_points):
                # Add organic sway
                sway_x = sway_strength * np.sin(t * 0.01 + j * 0.1) * (j / len(branch_points))
                sway_y = sway_strength * np.cos(t * 0.008 + j * 0.15) * (j / len(branch_points))
                
                x_coords.append(x + sway_x)
                y_coords.append(y + sway_y)
            
            # Draw with varying thickness and color
            if len(x_coords) > 1:
                color = current_palette[i % len(current_palette)]
                alpha = 0.4 + 0.3 * np.sin(t * 0.02 + i)
                
                self.ax.plot(x_coords, y_coords, color=color, 
                           linewidth=2 + np.sin(t * 0.01 + i), alpha=alpha)
                
                # Add glowing endpoints
                if x_coords:
                    self.ax.scatter([x_coords[-1]], [y_coords[-1]], 
                                  s=30 + 20 * np.sin(t * 0.05 + i), 
                                  c=[color], alpha=0.8, edgecolors='white', linewidth=0.5)
    
    def animate(self, frame):
        self.ax.clear()
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-9, 9)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_facecolor('#000000')
        
        t = frame
        self.frame_count = frame
        
        # Calculate complex flow field
        flow_x, flow_y = self.calculate_flow_field(t)
        
        # Update all particle systems
        self.update_main_particles(t, flow_x, flow_y)
        self.update_trail_particles(t)
        
        # Get current color scheme
        current_palette, next_palette, blend_factor = self.get_natural_colors(t)
        
        # Draw organic branches
        self.draw_organic_branches(t)
        
        # Draw micro particles (background texture)
        micro_x = self.micro_particles[:, 0] + 0.5 * np.sin(t * 0.005 + self.micro_phases)
        micro_y = self.micro_particles[:, 1] + 0.3 * np.cos(t * 0.007 + self.micro_phases * 1.3)
        
        micro_sizes = 5 + 3 * np.sin(t * 0.03 + self.micro_phases)
        micro_alphas = 0.1 + 0.1 * np.sin(t * 0.04 + self.micro_phases)
        
        self.ax.scatter(micro_x, micro_y, s=micro_sizes, 
                       c=[current_palette[int(phase * len(current_palette)) % len(current_palette)] 
                          for phase in self.micro_phases],
                       alpha=0.2)
        
        # Draw trail particles with organic connections
        trail_ages_normalized = self.trail_ages / 200.0
        trail_sizes = 15 * (1 - trail_ages_normalized) + 5
        trail_alphas = 0.6 * (1 - trail_ages_normalized) + 0.1
        
        # Create organic connections between nearby trail particles
        trail_distances = cdist(self.trail_particles, self.trail_particles)
        connection_threshold = 2.5
        
        for i in range(min(100, self.n_trail_particles)):  # Limit for performance
            nearby = np.where((trail_distances[i] < connection_threshold) & (trail_distances[i] > 0))[0]
            for j in nearby[:3]:  # Max 3 connections per particle
                if i < j:  # Avoid duplicate lines
                    x_line = [self.trail_particles[i, 0], self.trail_particles[j, 0]]
                    y_line = [self.trail_particles[i, 1], self.trail_particles[j, 1]]
                    
                    connection_strength = 1 - (trail_distances[i, j] / connection_threshold)
                    alpha = 0.3 * connection_strength * (1 - trail_ages_normalized[i]) * (1 - trail_ages_normalized[j])
                    
                    self.ax.plot(x_line, y_line, color=current_palette[2], alpha=alpha, linewidth=1)
        
        self.ax.scatter(self.trail_particles[:, 0], self.trail_particles[:, 1], 
                       s=trail_sizes, 
                       c=[current_palette[1]] * len(self.trail_particles),
                       alpha=trail_alphas, edgecolors='white', linewidth=0.3)
        
        # Draw main particles with complex interactions
        particle_ages_normalized = self.main_ages / self.main_lifetimes
        particle_sizes = 25 + 15 * np.sin(t * 0.1 + self.main_phases) * (1 - particle_ages_normalized)
        particle_alphas = 0.8 * (1 - particle_ages_normalized * 0.5)
        
        # Create dynamic particle colors based on velocity and position
        velocities_mag = np.linalg.norm(self.main_velocities, axis=1)
        max_vel = np.max(velocities_mag) if np.max(velocities_mag) > 0 else 1
        
        particle_colors = []
        for i in range(self.n_main_particles):
            vel_factor = velocities_mag[i] / max_vel
            color_idx = int((vel_factor + t * 0.01) * len(current_palette)) % len(current_palette)
            particle_colors.append(current_palette[color_idx])
        
        self.ax.scatter(self.main_particles[:, 0], self.main_particles[:, 1], 
                       s=particle_sizes, c=particle_colors, alpha=particle_alphas,
                       edgecolors='white', linewidth=0.8)
        
        # Draw velocity trails for main particles
        for i in range(0, self.n_main_particles, 3):  # Sample for performance
            if velocities_mag[i] > 0.1:  # Only for moving particles
                trail_length = min(3, velocities_mag[i] * 5)
                trail_x = [self.main_particles[i, 0] - self.main_velocities[i, 0] * trail_length,
                          self.main_particles[i, 0]]
                trail_y = [self.main_particles[i, 1] - self.main_velocities[i, 1] * trail_length,
                          self.main_particles[i, 1]]
                
                self.ax.plot(trail_x, trail_y, color=particle_colors[i], 
                           alpha=0.4 * particle_alphas[i], linewidth=2)
        
        # Add central energy vortex
        vortex_radius = 1.5 + 0.5 * np.sin(t * 0.05)
        vortex_alpha = 0.2 + 0.1 * np.sin(t * 0.03)
        
        theta = np.linspace(0, 4*np.pi, 100)
        spiral_r = vortex_radius * theta / (4*np.pi)
        spiral_x = spiral_r * np.cos(theta + t * 0.02)
        spiral_y = spiral_r * np.sin(theta + t * 0.02)
        
        self.ax.plot(spiral_x, spiral_y, color=current_palette[0], 
                    alpha=vortex_alpha, linewidth=3)
        
        return []

def main():
    # Create animation
    art_anim = NaturalAbstractAnimation()
    
    print("Creating complex natural abstract animation...")
    print("This will take several minutes to render at 60fps with high complexity...")
    print("Features: Flocking particles, organic branches, flow fields, natural color transitions")
    
    # Configure FFmpeg path
    static_ffmpeg.add_paths()
    plt.rcParams['animation.ffmpeg_path'] = shutil.which('ffmpeg')
    
    # Create animation
    anim = animation.FuncAnimation(
        art_anim.fig, art_anim.animate, 
        frames=art_anim.total_frames,
        interval=16.67,  # 60fps
        blit=False,
        repeat=True
    )
    
    # Set up the writer with high quality settings
    writer = animation.FFMpegWriter(
        fps=60,
        metadata=dict(artist='NaturalAbstractArt'),
        bitrate=8000,
        extra_args=['-pix_fmt', 'yuv420p']
    )
    
    # Save the animation
    output_file = 'natural_abstract_animation.mp4'
    print(f"Saving high-quality animation to {output_file}...")
    
    anim.save(output_file, writer=writer, dpi=150)
    print(f"Animation saved successfully!")
    
    # Automatically open the video
    if os.name == 'nt':  # Windows
        os.startfile(output_file)
    elif os.name == 'posix':  # macOS and Linux
        os.system(f'open "{output_file}"' if platform.system() == 'Darwin' else f'xdg-open "{output_file}"')
    
    plt.close()

if __name__ == "__main__":
    main()