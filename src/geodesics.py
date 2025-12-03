"""
Geodesic Simulator for Black Holes
===================================
Simulates timelike (massive particles) and null (photons) geodesics
around Schwarzschild and Kerr black holes.
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
                         E: float = 1.0,
                         radial_direction: str = "tangent") -> np.ndarray:
        """Construct initial state satisfying geodesic constraints."""
        raise NotImplementedError("Subclasses must implement get_initial_state")
    
    def critical_impact_parameter(self, is_timelike: bool = False) -> float:
        """Return critical impact parameter for capture."""
        raise NotImplementedError("Subclasses must implement critical_impact_parameter")


# ============================================================================
# SCHWARZSCHILD METRIC
# ============================================================================

class SchwarzschildMetric(Metric):
    """Schwarzschild metric for non-rotating black holes."""
    
    def __init__(self, mass: float = 1.0):
        super().__init__(mass)
        self.r_photon = 3 * self.M  # Photon sphere
        self.r_isco = 6 * self.M    # Innermost stable circular orbit (timelike)
        
        # Critical impact parameters
        self.b_crit_photon = np.sqrt(27) * self.M  # ≈ 5.196 M for photons
        self.b_crit_massive = 4 * self.M  # Approximate for massive particles
    
    def metric_factor(self, r: float) -> float:
        """f(r) = 1 - 2M/r"""
        return 1 - 2*self.M/r if r > self.r_s else 0.0
    
    def critical_impact_parameter(self, is_timelike: bool = False) -> float:
        """
        Return critical impact parameter for capture.
        
        For photons: b_crit = √27 M ≈ 5.196 M
        For massive particles: b_crit ≈ 4 M (approximate)
        
        Particles with b > b_crit escape
        Particles with b < b_crit are captured
        """
        if is_timelike:
            return self.b_crit_massive
        else:
            return self.b_crit_photon
    
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
        # IMPORTANT: Sign convention!
        # The centrifugal term +(r-2M)(dφ/dτ)² is POSITIVE (pushes outward)
        # This comes from Γ^r_φφ = -(r-2M), so the term is -Γ^r_φφ(dφ/dτ)²
        
        # d²t/dτ² = -2 (M/r²) / f * (dt/dτ)(dr/dτ)
        d2t = -2 * (M / (r**2 * f)) * ut * ur
        
        # d²r/dτ² = -M(1-2M/r)/r² (dt/dτ)² + M/(r²(1-2M/r)) (dr/dτ)² + (r-2M)(dφ/dτ)²
        # Note: The centrifugal term is POSITIVE (pushes outward)
        d2r = (-M * f / r**2 * ut**2 
               + M / (r**2 * f) * ur**2 
               + (r - 2*M) * uphi**2)  # POSITIVE! This was the bug!
        
        # d²φ/dτ² = -2/r (dr/dτ)(dφ/dτ)
        d2phi = -2 / r * ur * uphi
        
        return np.array([ut, ur, uphi, d2t, d2r, d2phi])
    
    def get_initial_state(self, r0: float, phi0: float = 0.0, 
                         impact_param: float = 5.0,
                         is_timelike: bool = False,
                         E: float = 1.0,
                         radial_direction: str = "tangent") -> np.ndarray:
        """
        Create initial conditions using conserved quantities.
        
        For geodesics in Schwarzschild, we have two conserved quantities:
        - E: Energy per unit mass (or energy for photons)
        - L: Angular momentum per unit mass (or angular momentum for photons)
        
        The impact parameter b = L/E relates them.
        
        Critical values for photons:
        - b > sqrt(27) M ≈ 5.196 M: photon escapes
        - b < sqrt(27) M: photon is captured
        - b = sqrt(27) M: photon orbits at r = 3M (photon sphere)
        
        Parameters:
        -----------
        r0 : float
            Initial radius
        phi0 : float
            Initial angle
        impact_param : float
            Impact parameter b = L/E (determines trajectory fate)
        is_timelike : bool
            True for massive particles, False for photons
        E : float
            Energy (per unit mass for massive particles)
        radial_direction : str
            "tangent" = dr/dτ = 0 (tangential start)
            "inward" = dr/dτ < 0 (moving inward)
            "outward" = dr/dτ > 0 (moving outward)
            "auto" = determine from effective potential
        """
        t0 = 0.0
        f = self.metric_factor(r0)
        
        # Set angular momentum from impact parameter
        L = impact_param * E
        
        # Angular velocity dphi/dtau
        dphi_dtau = L / r0**2
        
        if not is_timelike:
            # ============================================================
            # NULL GEODESIC (PHOTON)
            # ============================================================
            # From constraint: -f(dt/dτ)² + (1/f)(dr/dτ)² + r²(dφ/dτ)² = 0
            
            if radial_direction == "tangent":
                # Tangential start: dr/dtau ≈ 0 (very small)
                # For truly tangential motion at r0, the impact parameter is fixed:
                # b_tangential = r0 / sqrt(1 - 2M/r0)
                # If user specifies different b, we add small radial velocity
                
                # First: compute what b would be for EXACTLY tangential
                b_exact_tangential = r0 / np.sqrt(f) if f > 0 else impact_param
                
                # Set dt/dtau from energy
                dt_dtau = E / f
                
                # Compute dr/dtau from null constraint
                # (dr/dtau)² = E² - f*L²/r² 
                dr_dtau_sq = E**2 - f * L**2 / r0**2
                
                if abs(impact_param - b_exact_tangential) / b_exact_tangential < 0.01:
                    # Close enough to tangential - set dr/dtau = 0
                    dr_dtau = 0.0
                elif dr_dtau_sq >= 0:
                    # Need radial motion to achieve desired impact parameter
                    # Determine direction based on impact parameter
                    if impact_param > b_exact_tangential:
                        # Larger b means photon should move outward initially
                        dr_dtau = np.sqrt(dr_dtau_sq)
                    else:
                        # Smaller b means photon should move inward initially  
                        dr_dtau = -np.sqrt(dr_dtau_sq)
                else:
                    # dr²/dtau² < 0: at turning point
                    dr_dtau = 0.0
                
            elif radial_direction == "inward":
                # Inward motion with proper null constraint
                # We need: -f(dt/dtau)² + (1/f)(dr/dtau)² + r²(dphi/dtau)² = 0
                # Solve for dr/dtau given E and L
                
                # From E = f*dt/dtau and null constraint:
                # (dr/dtau)² = E² - f*L²/r²
                dr_dtau_sq = E**2 - f * L**2 / r0**2
                
                if dr_dtau_sq >= 0:
                    dr_dtau = -np.sqrt(dr_dtau_sq)  # negative = inward
                    dt_dtau = E / f
                else:
                    # At turning point
                    dr_dtau = 0.0
                    dt_dtau = r0 * dphi_dtau / np.sqrt(f) if f > 0 else E / f
                    
            elif radial_direction == "outward":
                # Outward motion with proper null constraint
                dr_dtau_sq = E**2 - f * L**2 / r0**2
                
                if dr_dtau_sq >= 0:
                    dr_dtau = np.sqrt(dr_dtau_sq)  # positive = outward
                    dt_dtau = E / f
                else:
                    # At turning point
                    dr_dtau = 0.0
                    dt_dtau = r0 * dphi_dtau / np.sqrt(f) if f > 0 else E / f
                    
            elif radial_direction == "auto":
                # Determine direction from effective potential
                dr_dtau_sq = E**2 - f * L**2 / r0**2
                
                if dr_dtau_sq < 0:
                    # Beyond turning point - start tangentially
                    dr_dtau = 0.0
                    dt_dtau = r0 * dphi_dtau / np.sqrt(f) if f > 0 else E / f
                else:
                    dt_dtau = E / f
                    # Check if we're at a stable/unstable point
                    # For r > 3M with b > b_crit: move outward
                    # For r < 3M or b < b_crit: move inward
                    
                    if impact_param > self.b_crit_photon:
                        # Should escape - move outward
                        dr_dtau = np.sqrt(dr_dtau_sq)
                    else:
                        # Should be captured - move inward
                        dr_dtau = -np.sqrt(dr_dtau_sq)
            else:
                raise ValueError(f"Unknown radial_direction: {radial_direction}")
        
        else:
            # ============================================================
            # TIMELIKE GEODESIC (MASSIVE PARTICLE)
            # ============================================================
            # From constraint: -f(dt/dtau)² + (1/f)(dr/dtau)² + r²(dphi/dtau)² = -1
            
            if radial_direction == "tangent":
                # Tangential start: dr/dtau = 0
                # For dr/dtau = 0: -f(dt/dtau)² + r²(dphi/dtau)² = -1
                # → f(dt/dtau)² = 1 + r²(dphi/dtau)²
                # → dt/dtau = sqrt[(1 + r²(dphi/dtau)²) / f]
                
                dr_dtau = 0.0
                dt_dtau = np.sqrt((1 + r0**2 * dphi_dtau**2) / f) if f > 0 else E / f
                
            elif radial_direction == "inward":
                # Inward motion
                # (dr/dtau)² = E² - (1 + L²/r²)f
                dr_dtau_sq = E**2 - (1 + L**2 / r0**2) * f
                
                if dr_dtau_sq >= 0:
                    dr_dtau = -np.sqrt(dr_dtau_sq)
                    dt_dtau = E / f
                else:
                    dr_dtau = 0.0
                    dt_dtau = np.sqrt((1 + r0**2 * dphi_dtau**2) / f) if f > 0 else E / f
                    
            elif radial_direction == "outward":
                # Outward motion
                dr_dtau_sq = E**2 - (1 + L**2 / r0**2) * f
                
                if dr_dtau_sq >= 0:
                    dr_dtau = np.sqrt(dr_dtau_sq)
                    dt_dtau = E / f
                else:
                    dr_dtau = 0.0
                    dt_dtau = np.sqrt((1 + r0**2 * dphi_dtau**2) / f) if f > 0 else E / f
                    
            elif radial_direction == "auto":
                dr_dtau_sq = E**2 - (1 + L**2 / r0**2) * f
                
                if dr_dtau_sq < 0:
                    dr_dtau = 0.0
                    dt_dtau = np.sqrt((1 + r0**2 * dphi_dtau**2) / f) if f > 0 else E / f
                else:
                    dt_dtau = E / f
                    if impact_param > self.b_crit_massive:
                        dr_dtau = np.sqrt(dr_dtau_sq)
                    else:
                        dr_dtau = -np.sqrt(dr_dtau_sq)
            else:
                raise ValueError(f"Unknown radial_direction: {radial_direction}")
        
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
        
        # Critical impact parameter (approximate, depends on spin and co/counter-rotating)
        self.b_crit_photon = np.sqrt(27) * self.M  # Rough estimate
    
    def critical_impact_parameter(self, is_timelike: bool = False) -> float:
        """
        Return approximate critical impact parameter for Kerr.
        Note: This is spin-dependent and different for co/counter-rotating orbits.
        """
        # This is a rough approximation
        return self.b_crit_photon
    
    def metric_functions(self, r: float):
        """
        Compute Kerr metric auxiliary functions at theta = pi/2 (equatorial).
        
        Returns: \Sigma, \Delta, A
        """
        # At θ = π/2: cos(θ) = 0, sin(θ) = 1
        Sigma = r**2  # r² + a²cos²(π/2) = r²
        Delta = r**2 - 2*self.M*r + self.a**2
        A = (r**2 + self.a**2)**2 - self.a**2 * Delta  # Simplifies at equator
        
        return Sigma, Delta, A
    
    def geodesic_equations(self, tau: float, state: np.ndarray, 
                          is_timelike: bool = False) -> np.ndarray:
        """
        Kerr geodesic equations in equatorial plane (theta = pi/2).
        
        SIMPLIFIED: For a=0, this must reduce exactly to Schwarzschild.
        For a>0, we add frame-dragging corrections.
        
        Reference: Bardeen et al. (1972), Eq. 2.2-2.5
        """
        t, r, phi, ut, ur, uphi = state
        
        # Safety check
        if r <= self.r_plus * 1.01:
            return np.zeros(6)
        
        M = self.M
        a = self.a
        
        # Metric functions
        Sigma = r**2  # + a^2cos^2(theta), but theta=pi/2 so cos(theta) = 0
        Delta = r**2 - 2*M*r + a**2
        
        # For a=0, this must give Schwarzschild!
        
        if abs(a) < 1e-10:
            # Pure Schwarzschild limit
            f = 1 - 2*M/r
            
            d2t = -2 * (M / (r**2 * f)) * ut * ur
            d2r = (-M * f / r**2 * ut**2 
                   + M / (r**2 * f) * ur**2 
                   + (r - 2*M) * uphi**2) 
            d2phi = -2 / r * ur * uphi
        else:
            # Kerr with spin
            # Christoffel symbols for equatorial Kerr
            
            # Time equation
            Gamma_t_tr = M*r / (Sigma * Delta)
            Gamma_t_rphi = a*M / (Sigma * Delta)
            
            d2t = -2 * Gamma_t_tr * ut * ur - 2 * Gamma_t_rphi * ur * uphi
            
            # Radial equation (critical!)
            # Gamma^r_tt, Gamma^r_rr, Gamma^r_phiphi, Gamma^r_tphi
            Gamma_r_tt = M*Delta*(r**2 - a**2) / Sigma**3
            Gamma_r_rr = (M*r**2 - M*a**2 - r*Delta) / (Sigma * Delta)
            Gamma_r_pp = -Delta*r / Sigma 
            Gamma_r_tp = 2*a*M**2*r / Sigma**3
            
            # The geodesic equation: d²r/dtau² = -Gamma^r_munu u^mu u^nu
            d2r = (-Gamma_r_tt * ut**2
                   - Gamma_r_rr * ur**2
                   - Gamma_r_pp * uphi**2      # -(-Delta*r/Sigma) = +Delta*r/Sigma
                   - 2*Gamma_r_tp * ut * uphi)
            
            # Angular equation  
            Gamma_p_rp = 1/r
            Gamma_p_tr = -a*M / (Sigma * Delta)
            
            d2phi = -2 * Gamma_p_rp * ur * uphi - 2 * Gamma_p_tr * ut * ur
        
        return np.array([ut, ur, uphi, d2t, d2r, d2phi])
    
    def get_initial_state(self, r0: float, phi0: float = 0.0,
                         impact_param: float = 5.0,
                         is_timelike: bool = False,
                         E: float = 1.0,
                         radial_direction: str = "tangent") -> np.ndarray:
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
            # For Kerr, energy and angular momentum are:
            # E = -p_t = -(g_tt dt/dτ + g_tφ dφ/dτ)
            # L = p_φ = g_tφ dt/dτ + g_φφ dφ/dτ
            
            # At equator:
            g_tt = -(1 - 2*self.M*r0/Sigma)
            g_tphi = -2*self.M*self.a*r0/Sigma
            g_rr = Sigma/Delta
            g_phiphi = A/Sigma
            
            # From L = b*E, we have dφ/dτ = L/Sigma (simplified for equatorial)
            # Then solve for dt/dτ and dr/dτ
            
            if radial_direction == "tangent":
                # Tangential: dr/dτ = 0
                # Use null constraint: 0 = g_tt(dt/dτ)² + 2g_tφ(dt/dτ)(dφ/dτ) + g_φφ(dφ/dτ)²
                
                a_coef = g_tt
                b_coef = 2 * g_tphi * dphi_dtau
                c_coef = g_phiphi * dphi_dtau**2
                
                discriminant = b_coef**2 - 4*a_coef*c_coef
                if discriminant >= 0:
                    dt_dtau = (-b_coef + np.sqrt(discriminant)) / (2*a_coef)
                else:
                    dt_dtau = E / (-g_tt)  # Fallback
                
                dr_dtau = 0.0
                
            elif radial_direction == "inward":
                # For Kerr, the relation between E, L and velocities involves solving:
                # E = -(g_tt u^t + g_tphi u^phi)
                # L = g_phiphi u^phi + g_tphi u^t
                
                # From these two equations, solve for u^t and u^phi in terms of E, L:
                # This gives (at equator):
                det_metric = g_tt * g_phiphi - g_tphi**2
                
                # u^t = (g_tphi L - g_phiphi E) / det
                # u^phi = (g_tphi E - g_tt L) / det
                
                ut_from_EL = (g_tphi * L - g_phiphi * E) / det_metric
                uphi_from_EL = (g_tphi * E - g_tt * L) / det_metric
                
                dt_dtau = ut_from_EL
                # Note: dphi_dtau should equal uphi_from_EL, let's check
                dphi_dtau = uphi_from_EL  # Override with correct value
                
                #dr/dtau from null constraint
                dr_dtau_sq = (-g_tt * dt_dtau**2 - 2*g_tphi * dt_dtau * dphi_dtau 
                             - g_phiphi * dphi_dtau**2) / g_rr
                if dr_dtau_sq >= 0:
                    dr_dtau = -np.sqrt(dr_dtau_sq)
                else:
                    dr_dtau = 0.0
                    
            elif radial_direction == "outward":
                det_metric = g_tt * g_phiphi - g_tphi**2
                ut_from_EL = (g_tphi * L - g_phiphi * E) / det_metric
                uphi_from_EL = (g_tphi * E - g_tt * L) / det_metric
                
                dt_dtau = ut_from_EL
                dphi_dtau = uphi_from_EL
                
                dr_dtau_sq = (-g_tt * dt_dtau**2 - 2*g_tphi * dt_dtau * dphi_dtau 
                             - g_phiphi * dphi_dtau**2) / g_rr
                if dr_dtau_sq >= 0:
                    dr_dtau = np.sqrt(dr_dtau_sq)
                else:
                    dr_dtau = 0.0
                    
            elif radial_direction == "auto":
                det_metric = g_tt * g_phiphi - g_tphi**2
                ut_from_EL = (g_tphi * L - g_phiphi * E) / det_metric
                uphi_from_EL = (g_tphi * E - g_tt * L) / det_metric
                
                dt_dtau = ut_from_EL
                dphi_dtau = uphi_from_EL
                
                dr_dtau_sq = (-g_tt * dt_dtau**2 - 2*g_tphi * dt_dtau * dphi_dtau 
                             - g_phiphi * dphi_dtau**2) / g_rr
                if dr_dtau_sq < 0:
                    dr_dtau = 0.0
                else:
                    if impact_param > self.b_crit_photon:
                        dr_dtau = np.sqrt(dr_dtau_sq)
                    else:
                        dr_dtau = -np.sqrt(dr_dtau_sq)
            else:
                raise ValueError(f"Unknown radial_direction: {radial_direction}")
                
        else:
            # Timelike geodesic
            g_tt = -(1 - 2*self.M*r0/Sigma)
            g_tphi = -2*self.M*self.a*r0/Sigma
            g_rr = Sigma/Delta
            g_phiphi = A/Sigma
            
            if radial_direction == "tangent":
                # Tangential: dr/dtau = 0
                # -1 = g_tt(dt/dtau)² + 2g_tphi(dt/dtau)(dphi/dtau) + g_phiphi(dphi/dtau)²
                # Quadratic in dt/dtau
                
                a_coef = g_tt
                b_coef = 2 * g_tphi * dphi_dtau
                c_coef = g_phiphi * dphi_dtau**2 + 1
                
                discriminant = b_coef**2 - 4*a_coef*c_coef
                if discriminant >= 0:
                    dt_dtau = (-b_coef + np.sqrt(discriminant)) / (2*a_coef)
                else:
                    dt_dtau = (E + g_tphi * dphi_dtau) / (-g_tt)
                
                dr_dtau = 0.0
            elif radial_direction == "inward":
                dt_dtau = (E + g_tphi * dphi_dtau) / (-g_tt)
                dr_dtau_sq = (-1 - g_tt * dt_dtau**2 - 2*g_tphi * dt_dtau * dphi_dtau 
                             - g_phiphi * dphi_dtau**2) / g_rr
                if dr_dtau_sq >= 0:
                    dr_dtau = -np.sqrt(dr_dtau_sq)
                else:
                    dr_dtau = 0.0
                    
            elif radial_direction == "outward":
                dt_dtau = (E + g_tphi * dphi_dtau) / (-g_tt)
                dr_dtau_sq = (-1 - g_tt * dt_dtau**2 - 2*g_tphi * dt_dtau * dphi_dtau 
                             - g_phiphi * dphi_dtau**2) / g_rr
                if dr_dtau_sq >= 0:
                    dr_dtau = np.sqrt(dr_dtau_sq)
                else:
                    dr_dtau = 0.0
                    
            elif radial_direction == "auto":
                dt_dtau = (E + g_tphi * dphi_dtau) / (-g_tt)
                dr_dtau_sq = (-1 - g_tt * dt_dtau**2 - 2*g_tphi * dt_dtau * dphi_dtau 
                             - g_phiphi * dphi_dtau**2) / g_rr
                if dr_dtau_sq < 0:
                    dr_dtau = 0.0
                else:
                    if impact_param > self.b_crit_photon:
                        dr_dtau = np.sqrt(dr_dtau_sq)
                    else:
                        dr_dtau = -np.sqrt(dr_dtau_sq)
            else:
                raise ValueError(f"Unknown radial_direction: {radial_direction}")
        
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
                E: float = None,  # Will auto-calculate if None
                tau_span: Tuple[float, float] = (0, 100),
                radial_direction: str = "tangent",
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
        E : float or None
            Energy (per unit mass for massive particles).
            For photons: E=1.0 is standard normalization
            For massive particles:
              - E < 1: Bound orbits
              - E = 1: Particle at rest at infinity (marginally bound)
              - E > 1: Unbound (hyperbolic) trajectories
            If None: auto-calculated as 95% of circular orbit energy
        tau_span : tuple
            Integration range
        radial_direction : str
            "tangent", "inward", "outward", or "auto"
        label : str
            Trajectory label
        """
        # Auto-calculate energy if not specified
        if E is None:
            if is_timelike:
                # For massive particles: use slightly sub-circular energy
                if r0 > 3 * self.metric.M:
                    E_circ = np.sqrt((r0 - 2*self.metric.M)/(r0 - 3*self.metric.M)) / np.sqrt(r0)
                    E = E_circ * 0.95  # 5% below circular for slow inspiral
                else:
                    E = 0.90  # Default for close orbits
            else:
                # For photons: standard normalization
                E = 1.0
        initial_state = self.metric.get_initial_state(
            r0, phi0, impact_param, is_timelike, E, radial_direction
        )
        
        solution = self.integrator.integrate(
            initial_state, tau_span, is_timelike
        )
        
        trajectory = Trajectory.from_solution(solution, is_timelike, label)
        self.trajectories.append(trajectory)
        
        return trajectory
    
    def simulate_bundle(self, r0: float, 
                       impact_params: np.ndarray,
                       is_timelike: bool = False,
                       tau_span: Tuple[float, float] = (0, 100),
                       radial_direction: str = "tangent") -> List[Trajectory]:
        """
        Simulate multiple particles with different impact parameters.
        """
        self.trajectories = []
        
        b_crit = self.metric.critical_impact_parameter(is_timelike)
        particle_type = "massive" if is_timelike else "photon"
        
        print(f"\n{'='*60}")
        print(f"Simulating {particle_type}s around {self.metric.__class__.__name__}")
        print(f"Critical impact parameter: b_crit = {b_crit:.3f} M")
        print(f"Initial radius: r₀ = {r0:.2f} M")
        print(f"Radial direction: {radial_direction}")
        print(f"{'='*60}\n")
        
        for i, b in enumerate(impact_params):
            fate = "ESCAPE" if b > b_crit else "CAPTURE"
            label = f"b={b:.3f}M ({fate})"
            
            traj = self.simulate(r0, 0.0, b, is_timelike, 1.0,
                               tau_span, radial_direction, label)
            
            # Determine actual fate
            if len(traj) > 0:
                if traj.r[-1] > 50:
                    actual = "escaped"
                elif traj.r[-1] < self.metric.r_s * 2:
                    actual = "captured"
                else:
                    actual = "orbiting"
                print(f"  {label:30s} → {actual:10s} (points: {len(traj)})")
        
        print(f"\n{'='*60}\n")
        return self.trajectories
    
    def simulate_random_bundle(self, n_particles: int = 20,
                              r_range: Tuple[float, float] = (5, 30),
                              impact_range: Tuple[float, float] = (2, 10),
                              is_timelike: bool = False,
                              tau_span: Tuple[float, float] = (0, 100)) -> List[Trajectory]:
        """
        Simulate particles with random initial conditions.
        
        Useful for exploring the phase space of trajectories.
        """
        self.trajectories = []
        
        particle_type = "massive" if is_timelike else "photon"
        print(f"\n{'='*60}")
        print(f"Simulating {n_particles} random {particle_type}s")
        print(f"{'='*60}\n")
        
        # Random initial conditions
        r0_values = np.random.uniform(r_range[0], r_range[1], n_particles)
        phi0_values = np.random.uniform(0, 2*np.pi, n_particles)
        b_values = np.random.uniform(impact_range[0], impact_range[1], n_particles)
        
        # Random radial directions
        directions = np.random.choice(["tangent", "inward", "outward"], n_particles)
        
        for i in range(n_particles):
            label = f"particle_{i+1}"
            
            traj = self.simulate(
                r0_values[i], phi0_values[i], b_values[i], 
                is_timelike, 1.0, tau_span, directions[i], label
            )
            
            if len(traj) > 0:
                if traj.r[-1] > 50:
                    fate = "escaped"
                elif traj.r[-1] < self.metric.r_s * 2:
                    fate = "captured"
                else:
                    fate = "orbiting"
                    
                print(f"  Particle {i+1:2d}: r₀={r0_values[i]:5.2f}, "
                      f"b={b_values[i]:5.2f}, dir={directions[i]:8s} → {fate}")
        
        print(f"\n{'='*60}\n")
        return self.trajectories
    
    def clear(self):
        """Clear all stored trajectories."""
        self.trajectories = []