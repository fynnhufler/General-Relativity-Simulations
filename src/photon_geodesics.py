import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import os
from matplotlib.collections import LineCollection
from IPython.display import HTML

class SchwarzschildMetric:
    """Schwarzschild metric with Christoffel symbols (G = c = 1)."""

    def __init__(self, mass: float = 1.0):
        self.M = mass
        self.r_s = 2 * self.M
        self.r_photon = 3 * self.M

    def Gamma_t_tr(self, r: float) -> float:
        return self.M / (r**2 * (1 - 2*self.M/r))

    def Gamma_r_tt(self, r: float) -> float:
        return self.M * (1 - 2*self.M/r) / r**2

    def Gamma_r_rr(self, r: float) -> float:
        return -self.M / (r**2 * (1 - 2*self.M/r))

    def Gamma_r_pp(self, r: float) -> float:
        return -(r - 2*self.M)

    def Gamma_p_rp(self, r: float) -> float:
        return 1.0 / r

    def metric_factor(self, r: float) -> float:
        return 1 - 2*self.M/r if r > self.r_s else 0.0


class GeodesicIntegrator:
    """Integrates geodesic equations for photon trajectories."""

    def __init__(self, metric: SchwarzschildMetric):
        """metric: SchwarzschildMetric object defining spacetime geometry."""
        self.metric = metric

    def geodesic_equations(self, tau: float, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for geodesic equation in equatorial plane.

        state: [t, r, φ, dt/dτ, dr/dτ, dφ/dτ]
        returns: time derivatives of state variables
        """
        t, r, phi, ut, ur, uphi = state
        M = self.metric.M

        if r <= self.metric.r_s * 1.01:
            return np.zeros(6)

        d2t = -2 * self.metric.Gamma_t_tr(r) * ut * ur
        d2r = (-self.metric.Gamma_r_tt(r) * ut**2
               - self.metric.Gamma_r_rr(r) * ur**2
               - self.metric.Gamma_r_pp(r) * uphi**2)
        d2phi = -2 * self.metric.Gamma_p_rp(r) * ur * uphi

        return np.array([ut, ur, uphi, d2t, d2r, d2phi])

    def integrate(self, initial_state: np.ndarray, tau_span: Tuple[float, float],
                  max_step: float = 0.1) -> dict:
        """Integrate geodesic equations with event stopping.

        initial_state: [t, r, φ, dt/dτ, dr/dτ, dφ/dτ]
        tau_span: (start, end) integration range
        max_step: maximum step size
        returns: solution object from scipy.integrate.solve_ivp
        """
        def hit_horizon(tau, state):
            return state[1] - self.metric.r_s * 1.05
        hit_horizon.terminal = True
        hit_horizon.direction = -1

        def escape(tau, state):
            return state[1] - 100
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
    """Stores photon trajectory data and derivatives."""

    def __init__(self, solution: dict, label: str = ""):
        """solution: scipy integrate result object, label: trajectory label."""
        self.tau = solution.t
        self.t = solution.y[0]
        self.r = solution.y[1]
        self.phi = solution.y[2]
        self.label = label

        self.ut = solution.y[3]
        self.ur = solution.y[4]
        self.uphi = solution.y[5]

        self.x = self.r * np.cos(self.phi)
        self.y = self.r * np.sin(self.phi)

        self.vr_coord = np.gradient(self.r, self.t)
        self.vphi_coord = np.gradient(self.phi, self.t)
        self.speed_coord = np.sqrt(self.vr_coord**2 + (self.r * self.vphi_coord)**2)

    def __len__(self):
        return len(self.tau)

class PhotonSimulation:
    """Manages simulation of photon trajectories around a black hole."""

    def __init__(self, mass: float = 1.0):
        """mass: black hole mass in geometric units (G=c=1)."""
        self.metric = SchwarzschildMetric(mass)
        self.integrator = GeodesicIntegrator(self.metric)
        self.trajectories: List[Trajectory] = []

    def create_initial_conditions(self, r0: float, phi_dot: float) -> np.ndarray:
        """Create initial conditions for a photon.

        r0: initial radius
        phi_dot: initial angular velocity (dφ/dτ)
        returns: [t, r, φ, dt/dτ, dr/dτ, dφ/dτ]
        """
        t0 = 0.0
        phi0 = 0.0

        f = self.metric.metric_factor(r0)
        dt_dtau = r0 * phi_dot / np.sqrt(f) if f > 0 else 1.0
        dr_dtau = 0.0

        return np.array([t0, r0, phi0, dt_dtau, dr_dtau, phi_dot])
    
    def simulate_photon(self, r0: float, phi_dot: float,
                       tau_span: Tuple[float, float]) -> Trajectory:
        """Simulate a single photon trajectory."""
        initial_state = self.create_initial_conditions(r0, phi_dot)
        solution = self.integrator.integrate(initial_state, tau_span)
        return Trajectory(solution, label=f"φ̇={phi_dot:.4f}")

    def simulate_bundle(self, r0_values: np.ndarray, phi_dot: float,
                       tau_span: Tuple[float, float]) -> List[Trajectory]:
        """Simulate multiple photons at different initial radii.

        r0_values: array of initial radii
        phi_dot: fixed angular velocity for all photons
        tau_span: integration range
        returns: list of Trajectory objects
        """
        self.trajectories = []
        for r0 in r0_values:
            traj = self.simulate_photon(r0, phi_dot, tau_span)
            self.trajectories.append(traj)

        return self.trajectories
    
    def plot_trajectories(self, figsize: Tuple[float, float] = (10, 10)):
        """Plot all simulated trajectories.

        figsize: figure dimensions (width, height)
        """
        fig, ax = plt.subplots(figsize=figsize)

        for traj in self.trajectories:
            ax.plot(traj.x, traj.y, alpha=0.7, linewidth=1.5)

        horizon = plt.Circle((0, 0), self.metric.r_s, color='black',
                            label='Event Horizon')
        ax.add_patch(horizon)

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
        """Animate photon trajectories and save to video.

        figsize: figure dimensions (width, height)
        video_path: output video file path
        speed_factor: animation speed multiplier
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (M)', fontsize=12)
        ax.set_ylabel('y (M)', fontsize=12)
        ax.set_title('Photon Trajectories Around Schwarzschild Black Hole', fontsize=14)

        horizon = plt.Circle((0, 0), self.metric.r_s, color='black', label='Event Horizon')
        ax.add_patch(horizon)
        photon_sphere = plt.Circle((0, 0), self.metric.r_photon, color='orange',
                                fill=False, linestyle='--', linewidth=2,
                                label='Photon Sphere')
        ax.add_patch(photon_sphere)
        ax.legend()

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.trajectories)))
        lines = [ax.plot([], [], alpha=0.7, linewidth=2, color=colors[i])[0]
                for i in range(len(self.trajectories))]

        max_length = max(len(traj) for traj in self.trajectories)
        n_frames = min(500, max_length)

        interpolated_trajs = []
        for traj in self.trajectories:
            if len(traj) > 1:
                tau_interp = np.linspace(traj.tau[0], traj.tau[-1], n_frames)
                x_interp = np.interp(tau_interp, traj.tau, traj.x)
                y_interp = np.interp(tau_interp, traj.tau, traj.y)
                interpolated_trajs.append((x_interp, y_interp))
            else:
                interpolated_trajs.append((traj.x, traj.y))

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for i, (x_data, y_data) in enumerate(interpolated_trajs):
                if frame < len(x_data):
                    lines[i].set_data(x_data[:frame+1], y_data[:frame+1])
                else:
                    lines[i].set_data(x_data, y_data)
            return lines

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        blit=True, interval=20)

        ani.save(video_path, writer='ffmpeg', fps=30, dpi=100)
        plt.close(fig)

    def analyze_speeds(self):
        """Analyze and plot photon speeds in different reference frames."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.trajectories)))

        ax1 = axes[0, 0]
        for i, traj in enumerate(self.trajectories):
            ax1.plot(traj.tau, traj.t, color=colors[i], alpha=0.7,
                    label=traj.label)
        ax1.set_xlabel('Affine Parameter τ', fontsize=12)
        ax1.set_ylabel('Coordinate Time t', fontsize=12)
        ax1.set_title('Coordinate Time vs Affine Parameter', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

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

        ax3 = axes[1, 0]
        for i, traj in enumerate(self.trajectories):
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
        """Animate trajectories in coordinate time (what distant observer sees)."""
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (M)', fontsize=12)
        ax.set_ylabel('y (M)', fontsize=12)
        ax.set_title('Photon Trajectories (Coordinate Time)', fontsize=14)

        horizon = plt.Circle((0, 0), self.metric.r_s, color='black', label='Event Horizon')
        ax.add_patch(horizon)
        photon_sphere = plt.Circle((0, 0), self.metric.r_photon, color='orange',
                                fill=False, linestyle='--', linewidth=2,
                                label='Photon Sphere')
        ax.add_patch(photon_sphere)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=14, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.legend()

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.trajectories)))
        lines = [ax.plot([], [], alpha=0.5, linewidth=1.5, color=colors[i])[0]
                for i in range(len(self.trajectories))]
        particles = [ax.plot([], [], 'o', markersize=8, color=colors[i])[0]
                    for i in range(len(self.trajectories))]

        t_min = min(traj.t[0] for traj in self.trajectories)
        t_max = min(traj.t[-1] for traj in self.trajectories)
        n_frames = 500
        t_grid = np.linspace(t_min, t_max, n_frames)

        interpolated_trajs = []
        for traj in self.trajectories:
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

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        blit=True, interval=20)

        ani.save(video_path, writer='ffmpeg', fps=30, dpi=100)
        plt.close(fig)

if __name__ == "__main__":
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/videos', exist_ok=True)

    plt.rcParams["animation.writer"] = "ffmpeg"
    plt.rcParams["animation.bitrate"] = 800
    plt.rcParams["animation.ffmpeg_args"] = [
        "-crf", "24",
        "-preset", "slow",
        "-pix_fmt", "yuv420p"
    ]
    plt.rcParams["animation.embed_limit"] = 25

    sim = PhotonSimulation(mass=1.0)
    M = sim.metric.M

    r0_values = np.linspace(3.1, 6, 20)
    tau_span = (0, 300)

    trajectories = sim.simulate_bundle(r0_values, 5.5, tau_span)

    if len(trajectories) > 0:
        plot_path = 'results/plots/photon_trajectories.png'
        sim.plot_trajectories(figsize=(10, 10))
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        sim.analyze_speeds()

        video_path_affine = 'results/videos/photon_affine_time.mp4'
        sim.animate_trajectories(video_path=video_path_affine)

        video_path_coord = 'results/videos/photon_coordinate_time.mp4'
        sim.animate_trajectories_coordinate_time(video_path=video_path_coord)