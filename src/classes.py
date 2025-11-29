import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional


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
        self.tau = solution.t
        self.t = solution.y[0]
        self.r = solution.y[1]
        self.phi = solution.y[2]
        self.label = label
        
        # Convert to Cartesian coordinates for plotting
        self.x = self.r * np.cos(self.phi)
        self.y = self.r * np.sin(self.phi)
    
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
        
        # Solve for dt/dτ from null condition with dr/dτ = 0 initially
        # (1-2M/r)(dt/dτ)² = r²(dφ/dτ)²
        dt_dtau = r0 * phi_dot / np.sqrt(f) if f > 0 else 1.0
        
        # Initial radial velocity (negative = inward)
        dr_dtau = -0.001  # Small inward velocity
        
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
    
    def simulate_bundle(self, r0: float, phi_dot_values: np.ndarray,
                       tau_span: Tuple[float, float]) -> List[Trajectory]:
        """
        Simulate multiple photons with different angular velocities.
        
        Parameters:
        -----------
        r0 : float
            Initial radius for all photons
        phi_dot_values : np.ndarray
            Array of angular velocities to simulate
        tau_span : tuple
            Integration time range
        
        Returns:
        --------
        trajectories : List[Trajectory]
            List of computed trajectories
        """
        self.trajectories = []
        for phi_dot in phi_dot_values:
            traj = self.simulate_photon(r0, phi_dot, tau_span)
            self.trajectories.append(traj)
            print(f"Simulated photon with φ̇={phi_dot:.4f}, points: {len(traj)}")
        
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


# Example usage
if __name__ == "__main__":
    # Create simulation
    sim = PhotonSimulation(mass=1.0)
    
    # Initial conditions
    r0 = 15.0  # Initial radius
    phi_dot_values = np.linspace(0.015, 0.08, 12)  # Angular velocities
    tau_span = (0, 80)  # Affine parameter range
    
    # Simulate photon bundle
    print("Starting photon simulations...")
    trajectories = sim.simulate_bundle(r0, phi_dot_values, tau_span)
    
    # Plot results
    print("Plotting trajectories...")
    sim.plot_trajectories()