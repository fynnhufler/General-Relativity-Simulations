"""
Geodesic Simulator for Black Holes
===================================
Simulates timelike (massive particles) and null (photons) geodesics
around Schwarzschild and Kerr black holes.

Fixed version with proper initial conditions and geodesic equations.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, List
from dataclasses import dataclass


# ============================================================================
# METRIC BASE CLASS
# ============================================================================

class Metric:
    """Base class for spacetime metrics."""
    
    def __init__(self, mass: float = 1.0):
        """
        Initialize metric.
        
        Parameters:
        -----------
        mass : float
            Black hole mass in geometric units (G = c = 1)
        """
        self.M = mass
        self.r_s = 2 * self.M  # Schwarzschild radius
    
    def geodesic_equations(self, tau: float, state: np.ndarray, 
                          is_timelike: bool = False) -> np.ndarray:
        """Geodesic equations: must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement geodesic_equations")
    
    def get_initial_state(self, r0: float, phi0: float, 
                         impact_param: float,
                         is_timelike: bool = False,
                         E: float = 1.0) -> np.ndarray:
        """Construct initial state satisfying geodesic constraints."""
        raise NotImplementedError("Subclasses must implement get_initial_state")


# ============================================================================
# SCHWARZSCHILD METRIC
# ============================================================================

class SchwarzschildMetric(Metric):
    """Schwarzschild metric for non-rotating black holes."""
    
    def __init__(self, mass: float = 1.0):
        super().__init__(mass)
        self.r_photon = 3 * self.M  # Photon sphere
        self.r_isco = 6 * self.M    # Innermost stable circular orbit (timelike)
    
    def metric_factor(self, r: float) -> float:
        """f(r) = 1 - 2M/r"""
        return 1 - 2*self.M/r if r > self.r_s else 0.0
    
    def geodesic_equations(self, tau: float, state: np.ndarray, 
                          is_timelike: bool = False) -> np.ndarray:
        """
        Schwarzschild geodesic equations in equatorial plane.
        
        State: [t, r, φ, dt/dτ, dr/dτ, dφ/dτ]
        """
        t, r, phi, ut, ur, uphi = state
        
        # Safety check
        if r <= self.r_s * 1.01:
            return np.zeros(6)
        
        f = self.metric_factor(r)
        M = self.M
        
        # Christoffel symbols (equatorial plane)
        # d²t/dτ² = -2 (M/r²) / f * (dt/dτ)(dr/dτ)
        d2t = -2 * (M / (r**2 * f)) * ut * ur
        
        # d²r/dτ² = -M(1-2M/r)/r² (dt/dτ)² + M/(r²(1-2M/r)) (dr/dτ)² - (r-2M)(dφ/dτ)²
        d2r = (-M * f / r**2 * ut**2 
               + M / (r**2 * f) * ur**2 
               - (r - 2*M) * uphi**2)
        
        # d²φ/dτ² = -2/r (dr/dτ)(dφ/dτ)
        d2phi = -2 / r * ur * uphi
        
        return np.array([ut, ur, uphi, d2t, d2r, d2phi])
    
    def get_initial_state(self, r0: float, phi0: float = 0.0, 
                         impact_param: float = 5.0,
                         is_timelike: bool = False,
                         E: float = 1.0) -> np.ndarray:
        """
        Create initial conditions using conserved quantities.
        
        For geodesics in Schwarzschild, we have two conserved quantities:
        - E: Energy per unit mass (or energy for photons)
        - L: Angular momentum per unit mass (or angular momentum for photons)
        
        The impact parameter b = L/E relates them.
        
        For photons starting at large r moving tangentially:
        b ≈ r * sin(angle) where angle is the initial direction
        
        Parameters:
        -----------
        r0 : float
            Initial radius
        phi0 : float
            Initial angle
        impact_param : float
            Impact parameter b = L/E (determines how close photon passes)
        is_timelike : bool
            True for massive particles, False for photons
        E : float
            Energy (per unit mass for massive particles)
        """
        t0 = 0.0
        f = self.metric_factor(r0)
        
        # Set angular momentum from impact parameter
        L = impact_param * E
        
        # For null geodesics (photons):
        # Energy equation: E² = f(dr/dτ)² + (L²/r² + κ)f
        # where κ = 0 for null, κ = 1 for timelike
        
        # Angular velocity dφ/dτ
        dphi_dtau = L / r0**2
        
        if not is_timelike:
            # Null geodesic (photon)
            # From constraint: -f(dt/dτ)² + (1/f)(dr/dτ)² + r²(dφ/dτ)² = 0
            # Using E² = f²(dt/dτ)² and L = r²(dφ/dτ):
            # (dr/dτ)² = E² - f*L²/r²
            
            dr_dtau_sq = E**2 - f * L**2 / r0**2
            
            # Start moving inward if dr/dτ² < 0, outward otherwise
            if dr_dtau_sq < 0:
                # Photon will spiral in - start with dr/dτ = 0 at turning point
                dr_dtau = 0.0
                # Recalculate to ensure consistency
                dr_dtau_sq = max(0, E**2 - f * L**2 / r0**2)
            
            dr_dtau = -np.sqrt(abs(dr_dtau_sq))  # Negative = inward
            
            # Time component: dt/dτ = E/f
            dt_dtau = E / f
        else:
            # Timelike geodesic (massive particle)
            # From constraint: -f(dt/dτ)² + (1/f)(dr/dτ)² + r²(dφ/dτ)² = -1
            # (dr/dτ)² = E² - (1 + L²/r²)f
            
            dr_dtau_sq = E**2 - (1 + L**2 / r0**2) * f
            
            if dr_dtau_sq < 0:
                dr_dtau = 0.0
            else:
                dr_dtau = -np.sqrt(dr_dtau_sq)  # Negative = inward
            
            # Time component: dt/dτ = E/f
            dt_dtau = E / f
        
        return np.array([t0, r0, phi0, dt_dtau, dr_dtau, dphi_dtau])


# ============================================================================
# KERR METRIC
# ============================================================================

class KerrMetric(Metric):
    """Kerr metric for rotating black holes (equatorial plane only)."""
    
    def __init__(self, mass: float = 1.0, spin: float = 0.5):
        """
        Initialize Kerr metric.
        
        Parameters:
        -----------
        mass : float
            Black hole mass
        spin : float
            Dimensionless spin parameter a/M ∈ [0, 1]
        """
        super().__init__(mass)
        self.a = spin * self.M  # Angular momentum per unit mass
        
        # Event horizons
        self.r_plus = self.M + np.sqrt(self.M**2 - self.a**2)
        self.r_minus = self.M - np.sqrt(self.M**2 - self.a**2)
        
        # Ergosphere (at equator)
        self.r_ergo = 2 * self.M
    
    def metric_functions(self, r: float):
        """
        Compute Kerr metric auxiliary functions at θ = π/2 (equatorial).
        
        Returns: Σ, Δ, A
        """
        # At θ = π/2: cos(θ) = 0, sin(θ) = 1
        Sigma = r**2  # r² + a²cos²(π/2) = r²
        Delta = r**2 - 2*self.M*r + self.a**2
        A = (r**2 + self.a**2)**2 - self.a**2 * Delta  # Simplifies at equator
        
        return Sigma, Delta, A
    
    def geodesic_equations(self, tau: float, state: np.ndarray, 
                          is_timelike: bool = False) -> np.ndarray:
        """
        Kerr geodesic equations in equatorial plane (θ = π/2).
        
        Uses the full equations with proper metric functions.
        """
        t, r, phi, ut, ur, uphi = state
        
        # Safety check
        if r <= self.r_plus * 1.01:
            return np.zeros(6)
        
        M = self.M
        a = self.a
        
        Sigma, Delta, A = self.metric_functions(r)
        
        # Derivatives of metric functions
        dSigma_dr = 2*r
        dDelta_dr = 2*r - 2*M
        dA_dr = 4*r*(r**2 + a**2) - 2*r*a**2 + 2*M*a**2
        
        # Geodesic equations (equatorial plane, using conservation of E and L)
        # These are simplified using the fact that θ = π/2 is a stable orbit
        
        # d²t/dτ²
        d2t = -(dSigma_dr/Sigma) * ut * ur - (2*a*M/Sigma**2) * dSigma_dr * uphi * ur
        
        # d²r/dτ² (effective potential method)
        # This is the most complex one
        term1 = (dDelta_dr/(2*Delta)) * (ur**2)
        term2 = ((r**2 + a**2)**2/A - r**2/Sigma) * ut**2 * dA_dr / (2*A)
        term3 = (a**2/A - 1/Sigma) * uphi**2 * r
        term4 = (2*a*M*r/A) * ut * uphi * dSigma_dr / Sigma
        
        d2r = -term1 + term2 - term3 + term4
        
        # d²φ/dτ²
        d2phi = -(dSigma_dr/Sigma) * ur * uphi + (2*a*M/Sigma**2) * dSigma_dr * ut * ur
        
        return np.array([ut, ur, uphi, d2t, d2r, d2phi])
    
    def get_initial_state(self, r0: float, phi0: float = 0.0,
                         impact_param: float = 5.0,
                         is_timelike: bool = False,
                         E: float = 1.0) -> np.ndarray:
        """
        Create initial conditions for Kerr geodesics (equatorial plane).
        
        Uses conserved quantities E (energy) and L (angular momentum).
        """
        t0 = 0.0
        
        Sigma, Delta, A = self.metric_functions(r0)
        
        # Angular momentum from impact parameter
        L = impact_param * E
        
        # Angular velocity
        dphi_dtau = L / Sigma  # Simplified for equatorial
        
        if not is_timelike:
            # Null geodesic
            # Using the full constraint at θ = π/2
            # 0 = g_tt(dt/dτ)² + 2g_tφ(dt/dτ)(dφ/dτ) + g_rr(dr/dτ)² + g_φφ(dφ/dτ)²
            
            # At equator:
            g_tt = -(1 - 2*self.M*r0/Sigma)
            g_tphi = -2*self.M*self.a*r0/Sigma
            g_rr = Sigma/Delta
            g_phiphi = A/Sigma
            
            # Time component: relates to energy
            # E = -g_tt(dt/dτ) - g_tφ(dφ/dτ)
            dt_dtau = (E + g_tphi * dphi_dtau) / (-g_tt)
            
            # Radial velocity from null constraint
            # g_rr(dr/dτ)² = -g_tt(dt/dτ)² - 2g_tφ(dt/dτ)(dφ/dτ) - g_φφ(dφ/dτ)²
            dr_dtau_sq = (-g_tt * dt_dtau**2 - 2*g_tphi * dt_dtau * dphi_dtau 
                         - g_phiphi * dphi_dtau**2) / g_rr
            
            if dr_dtau_sq < 0:
                dr_dtau = 0.0
            else:
                dr_dtau = -np.sqrt(dr_dtau_sq)  # Inward
        else:
            # Timelike geodesic
            g_tt = -(1 - 2*self.M*r0/Sigma)
            g_tphi = -2*self.M*self.a*r0/Sigma
            g_rr = Sigma/Delta
            g_phiphi = A/Sigma
            
            dt_dtau = (E + g_tphi * dphi_dtau) / (-g_tt)
            
            # Timelike constraint: g_μν u^μ u^ν = -1
            dr_dtau_sq = (-1 - g_tt * dt_dtau**2 - 2*g_tphi * dt_dtau * dphi_dtau 
                         - g_phiphi * dphi_dtau**2) / g_rr
            
            if dr_dtau_sq < 0:
                dr_dtau = 0.0
            else:
                dr_dtau = -np.sqrt(dr_dtau_sq)
        
        return np.array([t0, r0, phi0, dt_dtau, dr_dtau, dphi_dtau])


# ============================================================================
# GEODESIC INTEGRATOR
# ============================================================================

class GeodesicIntegrator:
    """Integrates geodesic equations for any metric."""
    
    def __init__(self, metric: Metric):
        self.metric = metric
    
    def integrate(self, initial_state: np.ndarray, 
                 tau_span: Tuple[float, float],
                 is_timelike: bool = False,
                 max_step: float = 0.1,
                 r_max: float = 100.0) -> dict:
        """Integrate geodesic equation."""
        
        # Event: particle hits horizon
        def hit_horizon(tau, state):
            if isinstance(self.metric, KerrMetric):
                return state[1] - self.metric.r_plus * 1.05
            else:
                return state[1] - self.metric.r_s * 1.05
        hit_horizon.terminal = True
        hit_horizon.direction = -1
        
        # Event: particle escapes
        def escape(tau, state):
            return state[1] - r_max
        escape.terminal = True
        escape.direction = 1
        
        # Wrapper for geodesic equations
        def equations(tau, state):
            return self.metric.geodesic_equations(tau, state, is_timelike)
        
        solution = solve_ivp(
            equations,
            tau_span,
            initial_state,
            method='DOP853',
            max_step=max_step,
            events=[hit_horizon, escape],
            dense_output=True,
            rtol=1e-9,
            atol=1e-12
        )
        
        return solution


# ============================================================================
# TRAJECTORY CLASS
# ============================================================================

@dataclass
class Trajectory:
    """Stores a computed geodesic trajectory."""
    
    tau: np.ndarray
    t: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    ut: np.ndarray
    ur: np.ndarray
    uphi: np.ndarray
    is_timelike: bool
    label: str = ""
    
    def __post_init__(self):
        """Compute derived quantities."""
        self.x = self.r * np.cos(self.phi)
        self.y = self.r * np.sin(self.phi)
        
        # Coordinate velocities (avoid division by zero)
        valid = self.t[1:] != self.t[:-1]
        self.vr_coord = np.zeros_like(self.r)
        self.vphi_coord = np.zeros_like(self.phi)
        if np.any(valid):
            self.vr_coord[1:][valid] = np.diff(self.r)[valid] / np.diff(self.t)[valid]
            self.vphi_coord[1:][valid] = np.diff(self.phi)[valid] / np.diff(self.t)[valid]
    
    @classmethod
    def from_solution(cls, solution: dict, is_timelike: bool = False, 
                     label: str = ""):
        """Create Trajectory from integration solution."""
        return cls(
            tau=solution.t,
            t=solution.y[0],
            r=solution.y[1],
            phi=solution.y[2],
            ut=solution.y[3],
            ur=solution.y[4],
            uphi=solution.y[5],
            is_timelike=is_timelike,
            label=label
        )
    
    def __len__(self):
        return len(self.tau)


# ============================================================================
# SIMULATION CLASS
# ============================================================================

class GeodesicSimulation:
    """High-level interface for simulating geodesics."""
    
    def __init__(self, metric: Metric):
        self.metric = metric
        self.integrator = GeodesicIntegrator(metric)
        self.trajectories: List[Trajectory] = []
    
    def simulate(self, r0: float, phi0: float = 0.0,
                impact_param: float = 5.0,
                is_timelike: bool = False,
                E: float = 1.0,
                tau_span: Tuple[float, float] = (0, 100),
                label: str = "") -> Trajectory:
        """
        Simulate single particle trajectory.
        
        Parameters:
        -----------
        r0 : float
            Initial radius
        phi0 : float
            Initial angle
        impact_param : float
            Impact parameter b = L/E
        is_timelike : bool
            True for massive particle, False for photon
        E : float
            Energy (per unit mass for massive particles)
        tau_span : tuple
            Integration range
        label : str
            Trajectory label
        """
        initial_state = self.metric.get_initial_state(
            r0, phi0, impact_param, is_timelike, E
        )
        
        solution = self.integrator.integrate(
            initial_state, tau_span, is_timelike
        )
        
        trajectory = Trajectory.from_solution(solution, is_timelike, label)
        self.trajectories.append(trajectory)
        
        return trajectory
    
    def simulate_bundle(self, r0_values: np.ndarray, 
                       impact_param: float = 5.0,
                       is_timelike: bool = False,
                       tau_span: Tuple[float, float] = (0, 100)) -> List[Trajectory]:
        """
        Simulate multiple particles at different radii.
        """
        self.trajectories = []
        
        for i, r0 in enumerate(r0_values):
            label = f"r₀={r0:.2f}"
            traj = self.simulate(r0, 0.0, impact_param, is_timelike, 1.0,
                               tau_span, label)
            print(f"Simulated: {label}, points: {len(traj)}")
        
        return self.trajectories
    
    def clear(self):
        """Clear all stored trajectories."""
        self.trajectories = []