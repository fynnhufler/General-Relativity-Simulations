"""
Celestial Sphere Visualization for Black Hole Ray Tracing
===========================================================

Creates sky maps showing how a black hole lenses background stars
as viewed from different observer positions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from geodesics import GeodesicSimulation, SchwarzschildMetric, KerrMetric


def create_celestial_grid(n_alpha=40, n_beta=40):
    """
    Create a grid of initial directions for photon ray tracing.
    
    Parameters:
    -----------
    n_alpha, n_beta : int
        Number of rays in each direction
        
    Returns:
    --------
    alpha, beta : arrays
        Sky coordinates (in radians from optical axis)
    """
    alpha_max = np.deg2rad(25)  # Field of view
    beta_max = np.deg2rad(25)
    
    alpha = np.linspace(-alpha_max, alpha_max, n_alpha)
    beta = np.linspace(-beta_max, beta_max, n_beta)
    
    return np.meshgrid(alpha, beta)


def trace_ray_backward(sim, observer_r, alpha, beta, max_tau=200):
    """
    Trace a photon ray backward from observer to determine if it hits
    the black hole or escapes to infinity.
    
    Parameters:
    -----------
    sim : GeodesicSimulation
        Simulator object
    observer_r : float
        Observer radial position
    alpha, beta : float
        Sky coordinates (angular offsets from optical axis)
    max_tau : float
        Maximum integration time
        
    Returns:
    --------
    fate : str
        "captured" or "escaped"
    final_phi : float
        Final azimuthal angle (for escaping rays)
    """
    # Convert sky coordinates to impact parameter and direction
    # For small angles: b ≈ r * α (in the appropriate direction)
    
    # Approximate impact parameter from angular position
    # This is simplified - proper ray tracing would use geodesic equation
    impact_param = observer_r * np.sqrt(alpha**2 + beta**2)
    
    if impact_param < 0.1:
        impact_param = 0.1  # Avoid singularity
    
    # Initial azimuthal angle
    phi0 = np.arctan2(beta, alpha)
    
    # Simulate photon trajectory (backward in time = outward initially)
    trajectory = sim.simulate(
        r0=observer_r,
        phi0=phi0,
        impact_param=impact_param,
        is_timelike=False,
        tau_span=(0, max_tau),
        radial_direction="inward",  # Actually backward in time
        label=""
    )
    
    if len(trajectory) == 0:
        return "escaped", phi0
    
    # Check fate
    final_r = trajectory.r[-1]
    
    if final_r < sim.metric.r_s * 2:
        return "captured", 0.0
    else:
        final_phi = trajectory.phi[-1]
        return "escaped", final_phi


def create_celestial_map(metric, observer_r=10.0, n_rays=30):
    """
    Create a celestial sphere map showing the black hole shadow.
    
    Parameters:
    -----------
    metric : Metric
        Black hole metric (Schwarzschild or Kerr)
    observer_r : float
        Observer distance from black hole
    n_rays : int
        Number of rays per direction
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    sim = GeodesicSimulation(metric)
    
    # Create ray grid
    alphas, betas = create_celestial_grid(n_rays, n_rays)
    
    # Result arrays
    captured = np.zeros_like(alphas, dtype=bool)
    
    print(f"Ray tracing with {n_rays}x{n_rays} = {n_rays**2} rays...")
    print(f"Observer at r = {observer_r}M")
    
    # Trace each ray
    for i in range(n_rays):
        for j in range(n_rays):
            alpha = alphas[i, j]
            beta = betas[i, j]
            
            fate, _ = trace_ray_backward(sim, observer_r, alpha, beta)
            captured[i, j] = (fate == "captured")
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{n_rays} rows complete")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Convert to degrees for display
    alphas_deg = np.rad2deg(alphas)
    betas_deg = np.rad2deg(betas)
    
    # Plot captured vs escaped regions
    ax.contourf(alphas_deg, betas_deg, captured.astype(int), 
                levels=[0, 0.5, 1], colors=['white', 'black'], alpha=0.8)
    
    # Add contour line
    ax.contour(alphas_deg, betas_deg, captured.astype(int),
               levels=[0.5], colors=['red'], linewidths=2)
    
    # Formatting
    ax.set_xlabel('α (degrees)', fontsize=14)
    ax.set_ylabel('β (degrees)', fontsize=14)
    ax.set_title(f'Black Hole Shadow (Observer at r = {observer_r}M)', 
                fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add info text
    metric_name = type(metric).__name__.replace('Metric', '')
    if hasattr(metric, 'a'):
        info_text = f'{metric_name}\na/M = {metric.a/metric.M:.2f}'
    else:
        info_text = f'{metric_name}'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    print(f"Ray tracing complete!")
    
    return fig, ax


def plot_trajectory_bundle(trajectories, metric, title="Photon Trajectories"):
    """
    Plot a bundle of photon trajectories with celestial coordinates.
    
    Parameters:
    -----------
    trajectories : list
        List of Trajectory objects
    metric : Metric
        Black hole metric
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectories
    for traj in trajectories:
        if len(traj) > 0:
            # Color by fate
            if traj.r[-1] > 50:
                color = 'blue'
                alpha = 0.6
            elif traj.r[-1] < metric.r_s * 2:
                color = 'red'
                alpha = 0.8
            else:
                color = 'green'
                alpha = 0.7
            
            ax.plot(traj.x, traj.y, color=color, alpha=alpha, linewidth=1.5)
    
    # Draw black hole features
    if hasattr(metric, 'r_plus'):
        # Kerr
        outer = Circle((0, 0), metric.r_plus, color='black', 
                      label=f'Horizon r+ = {metric.r_plus:.2f}M')
        ax.add_patch(outer)
        
        if metric.a > 0:
            ergo = Circle((0, 0), metric.r_ergo, fill=False, 
                         color='orange', linestyle='--', linewidth=2,
                         label='Ergosphere')
            ax.add_patch(ergo)
    else:
        # Schwarzschild
        horizon = Circle((0, 0), metric.r_s, color='black',
                        label=f'Horizon = {metric.r_s}M')
        ax.add_patch(horizon)
        
        photon = Circle((0, 0), metric.r_photon, fill=False,
                       color='orange', linestyle='--', linewidth=2,
                       label=f'Photon Sphere = {metric.r_photon}M')
        ax.add_patch(photon)
    
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (M)', fontsize=12)
    ax.set_ylabel('y (M)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    return fig, ax


def compare_schwarzschild_kerr_shadows(observer_r=10.0, n_rays=25):
    """
    Create side-by-side comparison of Schwarzschild and Kerr shadows.
    
    Parameters:
    -----------
    observer_r : float
        Observer distance
    n_rays : int
        Ray resolution
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Schwarzschild
    print("\n" + "="*70)
    print("Computing Schwarzschild shadow...")
    print("="*70)
    metric_sch = SchwarzschildMetric(mass=1.0)
    sim_sch = GeodesicSimulation(metric_sch)
    
    alphas, betas = create_celestial_grid(n_rays, n_rays)
    captured_sch = np.zeros_like(alphas, dtype=bool)
    
    for i in range(n_rays):
        for j in range(n_rays):
            fate, _ = trace_ray_backward(sim_sch, observer_r, alphas[i,j], betas[i,j])
            captured_sch[i, j] = (fate == "captured")
    
    alphas_deg = np.rad2deg(alphas)
    betas_deg = np.rad2deg(betas)
    
    axes[0].contourf(alphas_deg, betas_deg, captured_sch.astype(int),
                     levels=[0, 0.5, 1], colors=['white', 'black'])
    axes[0].contour(alphas_deg, betas_deg, captured_sch.astype(int),
                    levels=[0.5], colors=['red'], linewidths=2)
    axes[0].set_title('Schwarzschild (a=0)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('α (degrees)')
    axes[0].set_ylabel('β (degrees)')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Kerr moderate spin
    print("\n" + "="*70)
    print("Computing Kerr shadow (a=0.5M)...")
    print("="*70)
    metric_kerr1 = KerrMetric(mass=1.0, spin=0.5)
    sim_kerr1 = GeodesicSimulation(metric_kerr1)
    
    captured_kerr1 = np.zeros_like(alphas, dtype=bool)
    
    for i in range(n_rays):
        for j in range(n_rays):
            fate, _ = trace_ray_backward(sim_kerr1, observer_r, alphas[i,j], betas[i,j])
            captured_kerr1[i, j] = (fate == "captured")
    
    axes[1].contourf(alphas_deg, betas_deg, captured_kerr1.astype(int),
                     levels=[0, 0.5, 1], colors=['white', 'black'])
    axes[1].contour(alphas_deg, betas_deg, captured_kerr1.astype(int),
                    levels=[0.5], colors=['red'], linewidths=2)
    axes[1].set_title('Kerr (a=0.5M)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('α (degrees)')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    # Kerr high spin
    print("\n" + "="*70)
    print("Computing Kerr shadow (a=0.9M)...")
    print("="*70)
    metric_kerr2 = KerrMetric(mass=1.0, spin=0.9)
    sim_kerr2 = GeodesicSimulation(metric_kerr2)
    
    captured_kerr2 = np.zeros_like(alphas, dtype=bool)
    
    for i in range(n_rays):
        for j in range(n_rays):
            fate, _ = trace_ray_backward(sim_kerr2, observer_r, alphas[i,j], betas[i,j])
            captured_kerr2[i, j] = (fate == "captured")
    
    axes[2].contourf(alphas_deg, betas_deg, captured_kerr2.astype(int),
                     levels=[0, 0.5, 1], colors=['white', 'black'])
    axes[2].contour(alphas_deg, betas_deg, captured_kerr2.astype(int),
                    levels=[0.5], colors=['red'], linewidths=2)
    axes[2].set_title('Kerr (a=0.9M)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('α (degrees)')
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, axes


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CELESTIAL SPHERE VISUALIZATION DEMO")
    print("="*70)
    
    # Create comparison plot
    fig, axes = compare_schwarzschild_kerr_shadows(observer_r=10.0, n_rays=30)
    plt.savefig('black_hole_shadows.png', dpi=300, bbox_inches='tight')
    print("\nSaved: black_hole_shadows.png")
    plt.show()