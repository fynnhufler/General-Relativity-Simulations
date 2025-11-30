import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import os
from matplotlib.collections import LineCollection
from IPython.display import HTML

class SchwarzschildMetric:
    """
    Represents the Schwarzschild metric with precomputed Christoffel symbols.
    Uses geometric units where G = c = 1.
    """
    
    def __init__(self, mass: float = 1.0):
        """
        Initialize the Schwarzschild metric.
        
        Parameters:
        -----------
        mass : float
            Mass of the black hole (in geometric units where G = c = 1)
        """
        self.M = mass
        self.r_s = 2 * self.M  # Schwarzschild radius (event horizon)
        self.r_photon = 3 * self.M  # Photon sphere radius
    
    def Gamma_t_tr(self, r: float) -> float:
        """Γ^t_tr = M / (r² (1 - 2M/r))"""
        return self.M / (r**2 * (1 - 2*self.M/r))
    
    def Gamma_r_tt(self, r: float) -> float:
        """Γ^r_tt = M(1 - 2M/r) / r²"""
        return self.M * (1 - 2*self.M/r) / r**2
    
    def Gamma_r_rr(self, r: float) -> float:
        """Γ^r_rr = -M / (r² (1 - 2M/r))"""
        return -self.M / (r**2 * (1 - 2*self.M/r))
    
    def Gamma_r_pp(self, r: float) -> float:
        """Γ^r_φφ = -(r - 2M)"""
        return -(r - 2*self.M)
    
    def Gamma_p_rp(self, r: float) -> float:
        """Γ^φ_rφ = 1/r"""
        return 1.0 / r
    
    def metric_factor(self, r: float) -> float:
        """Compute f(r) = 1 - 2M/r"""
        return 1 - 2*self.M/r if r > self.r_s else 0.0


class GeodesicIntegrator:
    """
    Integrates the geodesic equation for photon trajectories.
    """
    
    def __init__(self, metric: SchwarzschildMetric):
        """
        Initialize the integrator with a metric.
        
        Parameters:
        -----------
        metric : SchwarzschildMetric
            The metric defining the spacetime geometry
        """
        self.metric = metric
    
    def geodesic_equations(self, tau: float, state: np.ndarray) -> np.ndarray:
        """
        Geodesic equations for photons in equatorial plane (θ = π/2).
        
        Parameters:
        -----------
        tau : float
            Affine parameter
        state : np.ndarray
            [t, r, φ, dt/dτ, dr/dτ, dφ/dτ]
        
        Returns:
        --------
        derivatives : np.ndarray
            Time derivatives of state variables
        """
        t, r, phi, ut, ur, uphi = state
        M = self.metric.M
        
        # Prevent integration inside event horizon
        if r <= self.metric.r_s * 1.01:
            return np.zeros(6)
        
        # Compute second derivatives using Christoffel symbols
        # d²t/dτ² = -2 Γ^t_tr (dt/dτ)(dr/dτ)
        d2t = -2 * self.metric.Gamma_t_tr(r) * ut * ur
        
        # d²r/dτ² = -Γ^r_tt (dt/dτ)² - Γ^r_rr (dr/dτ)² - Γ^r_φφ (dφ/dτ)²
        d2r = (-self.metric.Gamma_r_tt(r) * ut**2 
               - self.metric.Gamma_r_rr(r) * ur**2 
               - self.metric.Gamma_r_pp(r) * uphi**2)
        
        # d²φ/dτ² = -2 Γ^φ_rφ (dr/dτ)(dφ/dτ)
        d2phi = -2 * self.metric.Gamma_p_rp(r) * ur * uphi
        
        return np.array([ut, ur, uphi, d2t, d2r, d2phi])
    
    def integrate(self, initial_state: np.ndarray, tau_span: Tuple[float, float], 
                  max_step: float = 0.1) -> dict:
        """
        Integrate the geodesic equation.
        
        Parameters:
        -----------
        initial_state : np.ndarray
            Initial conditions [t, r, φ, dt/dτ, dr/dτ, dφ/dτ]
        tau_span : tuple
            (tau_start, tau_end) integration range
        max_step : float
            Maximum integration step size
        
        Returns:
        --------
        solution : dict
            Dictionary with solution arrays
        """
        # Event function to stop at event horizon
        def hit_horizon(tau, state):
            return state[1] - self.metric.r_s * 1.05
        hit_horizon.terminal = True
        hit_horizon.direction = -1
        
        # Event function to stop if photon escapes too far
        def escape(tau, state):
            return state[1] - 100  # Stop at r = 100M
        escape.terminal = True
        escape.direction = 1
        
        solution = solve_ivp(
            self.geodesic_equations,
            tau_span,
            initial_state,
            method='RK45',
            max_step=max_step,
            events=[hit_horizon, escape],
            dense_output=True
        )
        
        return solution


class Trajectory:
    """
    Stores and visualizes a photon trajectory.
    """
    
    def __init__(self, solution: dict, label: str = ""):
        """
        Initialize trajectory from integration solution.
        
        Parameters:
        -----------
        solution : dict
            Solution dictionary from scipy integrate
        label : str
            Label for this trajectory
        """
        self.tau = solution.t  # Affine parameter
        self.t = solution.y[0]  # Coordinate time
        self.r = solution.y[1]  # Radial coordinate
        self.phi = solution.y[2]  # Angular coordinate
        self.label = label
        
        # Velocities
        self.ut = solution.y[3]  # dt/dτ
        self.ur = solution.y[4]  # dr/dτ
        self.uphi = solution.y[5]  # dφ/dτ
        
        # Convert to Cartesian coordinates for plotting
        self.x = self.r * np.cos(self.phi)
        self.y = self.r * np.sin(self.phi)
        
        # Compute coordinate velocities (what a distant observer sees)
        self.vr_coord = np.gradient(self.r, self.t)  # dr/dt
        self.vphi_coord = np.gradient(self.phi, self.t)  # dφ/dt
        
        # Compute "speed" in coordinate space
        self.speed_coord = np.sqrt(self.vr_coord**2 + (self.r * self.vphi_coord)**2)
    
    def __len__(self):
        """Return number of points in trajectory"""
        return len(self.tau)

class PhotonSimulation:
    """
    Manages the simulation of multiple photon trajectories.
    """
    
    def __init__(self, mass: float = 1.0):
        """
        Initialize the photon simulation.
        
        Parameters:
        -----------
        mass : float
            Black hole mass in geometric units
        """
        self.metric = SchwarzschildMetric(mass)
        self.integrator = GeodesicIntegrator(self.metric)
        self.trajectories: List[Trajectory] = []
    
    def create_initial_conditions(self, r0: float, phi_dot: float) -> np.ndarray:
        """
        Create initial conditions for a photon in the equatorial plane.
        
        Parameters:
        -----------
        r0 : float
            Initial radial coordinate
        phi_dot : float
            Initial dφ/dτ (angular velocity)
        
        Returns:
        --------
        initial_state : np.ndarray
            [t, r, φ, dt/dτ, dr/dτ, dφ/dτ]
        """
        t0 = 0.0
        phi0 = 0.0
        
        # For photons, use null geodesic condition: g_μν u^μ u^ν = 0
        # In Schwarzschild: -(1-2M/r)(dt/dτ)² + (1-2M/r)^(-1)(dr/dτ)² + r²(dφ/dτ)² = 0
        
        # Assume photon starts moving radially inward/outward
        # For simplicity, we set dr/dτ based on the null condition
        f = self.metric.metric_factor(r0)
        
        # Solve for dt/dτ and dr/dτ from the null condition:
        # (1-2M/r)(dt/dτ)² = r²(dφ/dτ)²
        # We can choose a small dφ/dτ (phi_dot) and solve for the other velocities.
        
        # Assume the photon has no radial velocity at first (dr/dτ = 0)
        # And solve for the proper angular velocity (phi_dot).
        dt_dtau = r0 * phi_dot / np.sqrt(f) if f > 0 else 1.0  # Solving for dt/dτ based on phi_dot
        
        # Radial velocity (dr/dτ) is initially set to 0
        dr_dtau = 0.0
        
        return np.array([t0, r0, phi0, dt_dtau, dr_dtau, phi_dot])
    
    def simulate_photon(self, r0: float, phi_dot: float, 
                       tau_span: Tuple[float, float]) -> Trajectory:
        """
        Simulate a single photon trajectory.
        
        Parameters:
        -----------
        r0 : float
            Initial radius
        phi_dot : float
            Initial angular velocity
        tau_span : tuple
            Integration time range
        
        Returns:
        --------
        trajectory : Trajectory
            Computed photon trajectory
        """
        initial_state = self.create_initial_conditions(r0, phi_dot)
        solution = self.integrator.integrate(initial_state, tau_span)
        trajectory = Trajectory(solution, label=f"φ̇={phi_dot:.4f}")
        return trajectory
    
    def simulate_bundle(self, r0_values: np.ndarray, phi_dot: float,
                       tau_span: Tuple[float, float]) -> List[Trajectory]:
        """
        Simulate multiple photons at different initial radii.
        
        Parameters:
        -----------
        r0_values : np.ndarray
            Array of initial radii to simulate
        phi_dot : float
            Fixed angular velocity for all photons
        tau_span : tuple
            Integration time range
        
        Returns:
        --------
        trajectories : List[Trajectory]
            List of computed trajectories
        """
        self.trajectories = []
        for r0 in r0_values:
            traj = self.simulate_photon(r0, phi_dot, tau_span)
            self.trajectories.append(traj)
            print(f"Simulated photon with r0={r0:.4f}, points: {len(traj)}")
        
        return self.trajectories
    
    def plot_trajectories(self, figsize: Tuple[float, float] = (10, 10)):
        """
        Plot all simulated trajectories.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trajectories
        for traj in self.trajectories:
            ax.plot(traj.x, traj.y, alpha=0.7, linewidth=1.5)
        
        # Draw event horizon
        horizon = plt.Circle((0, 0), self.metric.r_s, color='black', 
                            label='Event Horizon')
        ax.add_patch(horizon)
        
        # Draw photon sphere
        photon_sphere = plt.Circle((0, 0), self.metric.r_photon, 
                                   color='orange', fill=False, 
                                   linestyle='--', linewidth=2,
                                   label='Photon Sphere')
        ax.add_patch(photon_sphere)
        
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (M)', fontsize=12)
        ax.set_ylabel('y (M)', fontsize=12)
        ax.set_title('Photon Trajectories Around Schwarzschild Black Hole', 
                    fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def animate_trajectories(self, figsize: Tuple[float, float] = (10, 10), 
                            video_path: str = 'results/videos/photon_trajectories.mp4',
                            speed_factor: float = 1.0):
        """
        Animate photon trajectories using FuncAnimation and save to a video file.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        video_path : str
            Path where the video will be saved
        speed_factor : float
            Animation speed multiplier (higher = faster)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up plot limits
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (M)', fontsize=12)
        ax.set_ylabel('y (M)', fontsize=12)
        ax.set_title('Photon Trajectories Around Schwarzschild Black Hole', fontsize=14)
        
        # Draw event horizon and photon sphere
        horizon = plt.Circle((0, 0), self.metric.r_s, color='black', label='Event Horizon')
        ax.add_patch(horizon)
        photon_sphere = plt.Circle((0, 0), self.metric.r_photon, color='orange', 
                                fill=False, linestyle='--', linewidth=2, 
                                label='Photon Sphere')
        ax.add_patch(photon_sphere)
        ax.legend()
        
        # Initialize lines with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.trajectories)))
        lines = [ax.plot([], [], alpha=0.7, linewidth=2, color=colors[i])[0] 
                for i in range(len(self.trajectories))]
        
        # Find the maximum length among all trajectories
        max_length = max(len(traj) for traj in self.trajectories)
        
        # Create interpolated points for smooth animation
        # Use a common time grid
        n_frames = min(500, max_length)  # Limit to 500 frames for reasonable file size
        
        # Interpolate all trajectories to the same number of frames
        interpolated_trajs = []
        for traj in self.trajectories:
            if len(traj) > 1:
                # Interpolate x and y to n_frames points
                tau_interp = np.linspace(traj.tau[0], traj.tau[-1], n_frames)
                x_interp = np.interp(tau_interp, traj.tau, traj.x)
                y_interp = np.interp(tau_interp, traj.tau, traj.y)
                interpolated_trajs.append((x_interp, y_interp))
            else:
                # Handle edge case of very short trajectories
                interpolated_trajs.append((traj.x, traj.y))
        
        # Initialize function
        def init():
            for line in lines:
                line.set_data([], [])
            return lines
        
        # Update function for the animation
        def update(frame):
            for i, (x_data, y_data) in enumerate(interpolated_trajs):
                if frame < len(x_data):
                    # Show trajectory up to current frame
                    lines[i].set_data(x_data[:frame+1], y_data[:frame+1])
                else:
                    # Show complete trajectory
                    lines[i].set_data(x_data, y_data)
            return lines
        
        # Create the animation
        print(f"Creating animation with {n_frames} frames...")
        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, 
                        blit=True, interval=20)  # 20ms = 50 fps
        
        # Save the animation as a video
        print(f"Saving animation to {video_path}...")
        ani.save(video_path, writer='ffmpeg', fps=30, dpi=100)
        print(f"Video saved successfully!")
        plt.close(fig)

    def analyze_speeds(self):
        """
        Analyze and plot photon speeds in different reference frames.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.trajectories)))
        
        # Plot 1: Coordinate time vs affine parameter
        ax1 = axes[0, 0]
        for i, traj in enumerate(self.trajectories):
            ax1.plot(traj.tau, traj.t, color=colors[i], alpha=0.7, 
                    label=traj.label)
        ax1.set_xlabel('Affine Parameter τ', fontsize=12)
        ax1.set_ylabel('Coordinate Time t', fontsize=12)
        ax1.set_title('Coordinate Time vs Affine Parameter', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot 2: dt/dτ vs radius (time dilation)
        ax2 = axes[0, 1]
        for i, traj in enumerate(self.trajectories):
            ax2.plot(traj.r, traj.ut, color=colors[i], alpha=0.7)
        ax2.axvline(self.metric.r_s, color='black', linestyle='--', 
                   label='Event Horizon')
        ax2.axvline(self.metric.r_photon, color='orange', linestyle='--', 
                   label='Photon Sphere')
        ax2.set_xlabel('Radius r (M)', fontsize=12)
        ax2.set_ylabel('dt/dτ (Time Dilation Factor)', fontsize=12)
        ax2.set_title('Time Dilation Along Trajectory', fontsize=14)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Plot 3: Coordinate speed |dr/dt|
        ax3 = axes[1, 0]
        for i, traj in enumerate(self.trajectories):
            # Avoid division by zero
            valid = traj.ut > 0
            dr_dt = traj.ur[valid] / traj.ut[valid]
            ax3.plot(traj.r[valid], np.abs(dr_dt), color=colors[i], alpha=0.7)
        ax3.axvline(self.metric.r_s, color='black', linestyle='--', 
                   label='Event Horizon')
        ax3.axvline(self.metric.r_photon, color='orange', linestyle='--', 
                   label='Photon Sphere')
        ax3.set_xlabel('Radius r (M)', fontsize=12)
        ax3.set_ylabel('|dr/dt| (Coordinate Speed)', fontsize=12)
        ax3.set_title('Radial Coordinate Velocity', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        # Plot 4: Angular velocity dφ/dt
        ax4 = axes[1, 1]
        for i, traj in enumerate(self.trajectories):
            valid = traj.ut > 0
            dphi_dt = traj.uphi[valid] / traj.ut[valid]
            ax4.plot(traj.r[valid], dphi_dt, color=colors[i], alpha=0.7)
        ax4.axvline(self.metric.r_s, color='black', linestyle='--', 
                   label='Event Horizon')
        ax4.axvline(self.metric.r_photon, color='orange', linestyle='--', 
                   label='Photon Sphere')
        ax4.set_xlabel('Radius r (M)', fontsize=12)
        ax4.set_ylabel('dφ/dt (Angular Velocity)', fontsize=12)
        ax4.set_title('Angular Coordinate Velocity', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig('results/plots/speed_analysis.png', dpi=300)
        plt.show()

    def animate_trajectories_coordinate_time(self, figsize: Tuple[float, float] = (10, 10), 
                                        video_path: str = 'results/videos/photon_trajectories_coordtime.mp4'):
        """
        Animate photon trajectories using COORDINATE TIME (what a distant observer sees).
        This will show that photons slow down near the black hole in coordinate time.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up plot
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (M)', fontsize=12)
        ax.set_ylabel('y (M)', fontsize=12)
        ax.set_title('Photon Trajectories (Coordinate Time)', fontsize=14)
        
        # Draw circles
        horizon = plt.Circle((0, 0), self.metric.r_s, color='black', label='Event Horizon')
        ax.add_patch(horizon)
        photon_sphere = plt.Circle((0, 0), self.metric.r_photon, color='orange', 
                                fill=False, linestyle='--', linewidth=2, 
                                label='Photon Sphere')
        ax.add_patch(photon_sphere)
        
        # Add time display
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                        fontsize=14, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.trajectories)))
        lines = [ax.plot([], [], alpha=0.5, linewidth=1.5, color=colors[i])[0] 
                for i in range(len(self.trajectories))]
        particles = [ax.plot([], [], 'o', markersize=8, color=colors[i])[0] 
                    for i in range(len(self.trajectories))]
        
        # Find common coordinate time grid
        t_min = min(traj.t[0] for traj in self.trajectories)
        t_max = min(traj.t[-1] for traj in self.trajectories)  # Use minimum to avoid extrapolation
        n_frames = 500
        t_grid = np.linspace(t_min, t_max, n_frames)
        
        # Interpolate trajectories to coordinate time grid
        interpolated_trajs = []
        for traj in self.trajectories:
            # Interpolate x, y as functions of coordinate time t
            x_interp = np.interp(t_grid, traj.t, traj.x)
            y_interp = np.interp(t_grid, traj.t, traj.y)
            interpolated_trajs.append((x_interp, y_interp))
        
        def init():
            for line, particle in zip(lines, particles):
                line.set_data([], [])
                particle.set_data([], [])
            time_text.set_text('')
            return lines + particles + [time_text]
        
        def update(frame):
            for i, (x_data, y_data) in enumerate(interpolated_trajs):
                lines[i].set_data(x_data[:frame+1], y_data[:frame+1])
                particles[i].set_data([x_data[frame]], [y_data[frame]])
            
            time_text.set_text(f'Coordinate Time t = {t_grid[frame]:.2f} M')
            return lines + particles + [time_text]
        
        print(f"Creating animation with {n_frames} frames (coordinate time)...")
        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, 
                        blit=True, interval=20)
        
        print(f"Saving animation to {video_path}...")
        ani.save(video_path, writer='ffmpeg', fps=30, dpi=100)
        print(f"Video saved successfully!")
        plt.close(fig)

"""
if __name__ == "__main__":
    # Ensure the directories exist
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/videos', exist_ok=True)
    
    # Configure matplotlib for animations
    plt.rcParams["animation.writer"] = "ffmpeg"
    plt.rcParams["animation.bitrate"] = 800
    plt.rcParams["animation.ffmpeg_args"] = [
        "-crf", "24",
        "-preset", "slow",
        "-pix_fmt", "yuv420p"
    ]
    plt.rcParams["animation.embed_limit"] = 25
    
    print("=" * 70)
    print("PHOTON TRAJECTORIES AROUND SCHWARZSCHILD BLACK HOLE")
    print("=" * 70)
    
    # Create simulation
    sim = PhotonSimulation(mass=1.0)
    M = sim.metric.M
    b_crit = 3 * np.sqrt(3) * M
    
    print(f"\nBlack hole parameters:")
    print(f"  Mass M = {M}")
    print(f"  Event horizon r_s = {sim.metric.r_s:.4f} M")
    print(f"  Photon sphere r_photon = {sim.metric.r_photon:.4f} M")
    print(f"  Critical impact parameter b_crit = {b_crit:.4f} M")
    
    # Simulation parameters
    impact_parameter = 5.5  # Slightly above critical
    r0_values = np.linspace(3.1, 6, 20)  # Different starting radii
    tau_span = (0, 300)  # Affine parameter range
    
    print(f"\nSimulation parameters:")
    print(f"  Impact parameter b = {impact_parameter:.2f} M")
    print(f"  Number of photons = {len(r0_values)}")
    print(f"  Starting radii: {r0_values[0]:.1f} M to {r0_values[-1]:.1f} M")
    print(f"  Integration range τ ∈ [0, {tau_span[1]}]")
    
    # ========================================================================
    # SIMULATE PHOTON TRAJECTORIES
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("SIMULATING PHOTON BUNDLE...")
    print("-" * 70)
    
    trajectories = sim.simulate_bundle(r0_values, impact_parameter, tau_span)
    
    print(f"\n✓ Successfully computed {len(trajectories)}/{len(r0_values)} trajectories")
    
    if len(trajectories) == 0:
        print("\n✗ ERROR: No trajectories computed! Check initial conditions.")
        exit(1)
    
    # ========================================================================
    # PLOT 1: STATIC TRAJECTORY PLOT
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("CREATING STATIC TRAJECTORY PLOT...")
    print("-" * 70)
    
    plot_path = 'results/plots/photon_trajectories.png'
    sim.plot_trajectories(figsize=(10, 10))
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Static plot saved to: {plot_path}")
    plt.close()
    
    # ========================================================================
    # PLOT 2: SPEED ANALYSIS
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("ANALYZING PHOTON SPEEDS...")
    print("-" * 70)
    
    sim.analyze_speeds()
    print(f"✓ Speed analysis saved to: results/plots/speed_analysis.png")
    
    # ========================================================================
    # VIDEO 1: ANIMATION IN AFFINE PARAMETER
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("CREATING ANIMATION (AFFINE PARAMETER)...")
    print("-" * 70)
    print("This shows uniform progression along geodesics.")
    
    video_path_affine = 'results/videos/photon_affine_time.mp4'
    sim.animate_trajectories_with_particles(video_path=video_path_affine)
    print(f"✓ Affine parameter animation saved to: {video_path_affine}")
    
    # ========================================================================
    # VIDEO 2: ANIMATION IN COORDINATE TIME
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("CREATING ANIMATION (COORDINATE TIME)...")
    print("-" * 70)
    print("This shows what a distant observer sees - photons slow near horizon!")
    
    video_path_coord = 'results/videos/photon_coordinate_time.mp4'
    sim.animate_trajectories_coordinate_time(video_path=video_path_coord)
    print(f"✓ Coordinate time animation saved to: {video_path_coord}")
    
    # ========================================================================
    # BONUS: SIMULATION WITH VARYING IMPACT PARAMETERS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("BONUS: SIMULATING PHOTONS WITH VARYING IMPACT PARAMETERS")
    print("=" * 70)
    
    # Fixed starting radius, vary impact parameter around critical value
    r0_fixed = 15.0
    impact_parameters = np.linspace(4.5, 6.5, 10)  # Around b_crit ≈ 5.196
    
    print(f"\nStarting radius: r0 = {r0_fixed} M")
    print(f"Impact parameters: {impact_parameters[0]:.2f} to {impact_parameters[-1]:.2f}")
    
    sim_varying_b = PhotonSimulation(mass=1.0)
    sim_varying_b.trajectories = []
    
    print("\nSimulating...")
    for b in impact_parameters:
        traj = sim_varying_b.simulate_photon(r0_fixed, b, tau_span)
        if traj is not None:
            sim_varying_b.trajectories.append(traj)
            fate = "captured" if traj.r[-1] < 3*M else "escaped"
            print(f"  b={b:.3f}: {fate} (final r={traj.r[-1]:.2f} M)")
    
    # Plot varying impact parameters
    print("\nCreating plot for varying impact parameters...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(sim_varying_b.trajectories)))
    for i, traj in enumerate(sim_varying_b.trajectories):
        ax.plot(traj.x, traj.y, alpha=0.7, linewidth=1.5, 
               color=colors[i], label=traj.label)
    
    horizon = plt.Circle((0, 0), sim_varying_b.metric.r_s, color='black', 
                        label='Event Horizon')
    ax.add_patch(horizon)
    photon_sphere = plt.Circle((0, 0), sim_varying_b.metric.r_photon, 
                              color='orange', fill=False, linestyle='--', 
                              linewidth=2, label='Photon Sphere')
    ax.add_patch(photon_sphere)
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (M)', fontsize=12)
    ax.set_ylabel('y (M)', fontsize=12)
    ax.set_title(f'Photon Trajectories: Varying Impact Parameter (r0={r0_fixed} M)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plot_path_varying = 'results/plots/photon_trajectories_varying_b.png'
    plt.savefig(plot_path_varying, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path_varying}")
    plt.close()
    
    # Animate varying impact parameters
    print("\nCreating animation for varying impact parameters...")
    video_path_varying = 'results/videos/photon_varying_impact.mp4'
    sim_varying_b.animate_trajectories_with_particles(video_path=video_path_varying)
    print(f"✓ Animation saved to: {video_path_varying}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    
    print("\nGenerated files:")
    print("\nPlots:")
    print(f"  1. results/plots/photon_trajectories.png")
    print(f"  2. results/plots/speed_analysis.png")
    print(f"  3. results/plots/photon_trajectories_varying_b.png")
    
    print("\nVideos:")
    print(f"  1. results/videos/photon_affine_time.mp4")
    print(f"     → Shows uniform progression in affine parameter")
    print(f"  2. results/videos/photon_coordinate_time.mp4")
    print(f"     → Shows gravitational time dilation (distant observer view)")
    print(f"  3. results/videos/photon_varying_impact.mp4")
    print(f"     → Shows effect of impact parameter on trajectories")
    
    print("\nKey physics insights:")
    print(f"  • Event horizon at r = {sim.metric.r_s:.2f} M")
    print(f"  • Photon sphere at r = {sim.metric.r_photon:.2f} M")
    print(f"  • Critical impact parameter b_crit = {b_crit:.4f} M")
    print(f"  • Photons with b < b_crit are captured")
    print(f"  • Photons with b > b_crit escape or orbit")
    print(f"  • Time dilation becomes infinite at event horizon")
    
    print("\n" + "=" * 70)
"""