#!/usr/bin/env python3
"""
Kerr Black Hole Simulations - SCATTER MODE
===========================================
50 photons scattered from a single point outside the black hole
Shows gravitational lensing and frame dragging effects

MODIFICATIONS:
- 50 photons (was 25)
- Scattered from single source point
- Impact parameters: b ∈ [2, 12]M (wider range)
- Angular spread: ±17°
- Color coded by fate: red (captured), green (escaped), blue (orbiting)
"""

import sys
sys.path.insert(0, './src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import time

from geodesics import KerrMetric, GeodesicSimulation

# Dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['savefig.facecolor'] = '#1a1a1a'

PHOTON_COLOR = '#87CEEB'
HORIZON_COLOR = 'black'
HORIZON_RING_COLOR = 'white'
ERGOSPHERE_COLOR = '#FF6B6B'

print("\n" + "="*70)
print("KERR BLACK HOLE SIMULATIONS - SCATTER MODE")
print("50 Photons from Single Source Point")
print("="*70)

# Spin parameters to simulate
spin_configs = [
    (0.0, "Schwarzschild (a=0)", "schwarzschild"),
    (0.5, "Moderate Spin (a=0.5M)", "moderate"),
    (0.9, "Fast Spin (a=0.9M)", "fast"),
    (0.998, "Near-Extremal (a=0.998M)", "extremal")
]

for spin, description, filename in spin_configs:
    
    print("\n" + "="*70)
    print(f"SIMULATION: {description}")
    print("="*70)
    
    start_time = time.time()
    
    # Create Kerr metric
    metric = KerrMetric(mass=1.0, spin=spin)
    sim = GeodesicSimulation(metric)
    
    print(f"\nMetric parameters:")
    print(f"  Spin parameter: a/M = {spin}")
    print(f"  Horizon radius: r+ = {metric.r_plus:.3f}M")
    print(f"  Ergosphere (equator): r_ergo = {metric.r_ergo:.3f}M")
    
    # ========================================================================
    # SCATTER MODE - 50 photons from single point
    # ========================================================================
    
    n_photons = 50  # More photons!
    
    # Starting point: single location outside BH
    r0_start = 25.0  # Far away
    phi0_start = np.pi  # Left side (negative x-axis)
    
    # Generate scattered photons with varying parameters
    impact_params = np.random.uniform(-2.0, 8.0, n_photons)  # Wide range
    phi_spread = np.random.uniform(-1.5, 1.5, n_photons)  # angular spread
    
    trajectories = []
    
    print(f"\nSimulating {n_photons} photons (SCATTER MODE)...")
    print(f"  Source position: r={r0_start}M, φ={np.degrees(phi0_start):.0f}°")
    print(f"  Impact parameters: b ∈ [2, 12]M")
    print(f"  Angular spread: ±17°")
    
    for i in range(n_photons):
        b = impact_params[i]
        phi0 = phi0_start + phi_spread[i]
        
        traj = sim.simulate(
            r0=r0_start,
            phi0=phi0,
            impact_param=b,
            is_timelike=False,
            E=1.0,
            tau_span=(0, 600),
            radial_direction="inward",
            label=f"photon_{i}"
        )
        
        if len(traj) > 0:
            trajectories.append(traj)
            
            # Check fate
            if traj.r[-1] < metric.r_plus * 1.5:
                fate = "captured"
            elif traj.r[-1] > 50:
                fate = "escaped"
            else:
                fate = "orbiting"
            
            if i % 10 == 0:  # Print every 10th
                print(f"  Photon {i+1}/50: b={b:.2f}M, φ={np.degrees(phi0):.1f}°, {fate}")
    
    elapsed = time.time() - start_time
    print(f"✅ Simulation complete in {elapsed:.2f}s")
    print(f"   Generated {len(trajectories)} trajectories")
    
    # ========================================================================
    # STATIC PLOT
    # ========================================================================
    
    print("Creating static plot...")
    
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_xlabel('x (M)', fontsize=14, color='white')
    ax.set_ylabel('y (M)', fontsize=14, color='white')
    ax.set_title(f'Kerr Black Hole - Scattered Photons: {description}', 
                 fontsize=16, fontweight='bold', color='white')
    
    # Mark source point
    x_source = r0_start * np.cos(phi0_start)
    y_source = r0_start * np.sin(phi0_start)
    ax.plot(x_source, y_source, '*', color='yellow', markersize=20, 
           label='Source Point', zorder=12, markeredgecolor='white', markeredgewidth=2)
    
    # Draw features
    horizon = Circle((0, 0), metric.r_plus, color=HORIZON_COLOR, zorder=10)
    ax.add_patch(horizon)
    horizon_ring = Circle((0, 0), metric.r_plus, fill=False,
                          color=HORIZON_RING_COLOR, linewidth=2, zorder=11,
                          label=f'Horizon ({metric.r_plus:.2f}M)')
    ax.add_patch(horizon_ring)
    
    # Ergosphere
    if spin > 0:
        ergosphere = Circle((0, 0), metric.r_ergo, fill=False,
                           color=ERGOSPHERE_COLOR, linestyle='--', linewidth=2,
                           label=f'Ergosphere ({metric.r_ergo:.2f}M)', zorder=9, alpha=0.7)
        ax.add_patch(ergosphere)
    
    # Plot trajectories
    for traj in trajectories:
        if len(traj) > 0:
            # Color code by fate
            if traj.r[-1] < metric.r_plus * 1.5:
                color = '#FF4444'  # Red for captured
                alpha = 0.7
            elif traj.r[-1] > 50:
                color = '#44FF44'  # Green for escaped
                alpha = 0.6
            else:
                color = PHOTON_COLOR  # Blue for orbiting
                alpha = 0.8
            
            ax.plot(traj.x, traj.y, color=color, linewidth=1.5, alpha=alpha)
    
    # Rotation indicator
    if spin > 0:
        arrow_props = dict(arrowstyle='->', lw=3, color='yellow', alpha=0.8)
        ax.annotate('', xy=(0, metric.r_ergo + 1), 
                   xytext=(0, metric.r_ergo + 3),
                   arrowprops=arrow_props)
        ax.text(0.5, metric.r_ergo + 3.5, 'Rotation ↻', 
               ha='center', fontsize=11, color='yellow', fontweight='bold')
    
    # Info box
    info_text = f"Spin: a/M = {spin}\n"
    info_text += f"Horizon: r₊ = {metric.r_plus:.3f}M\n"
    if spin > 0:
        info_text += f"Ergosphere: {metric.r_ergo:.1f}M\n"
    info_text += f"\nPhotons: {len(trajectories)}\n"
    info_text += f"Source: r={r0_start}M"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10, color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'./results/plots/kerr_{filename}_scatter_static.png', 
                dpi=150, facecolor='#1a1a1a')
    plt.close()
    
    print(f"Saved: results/plots/kerr_{filename}_scatter_static.png")
    
    # ========================================================================
    # ANIMATION
    # ========================================================================
    
    print("Creating animation...")
    
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_xlabel('x (M)', fontsize=14, color='white')
    ax.set_ylabel('y (M)', fontsize=14, color='white')
    
    # Mark source
    ax.plot(x_source, y_source, '*', color='yellow', markersize=20, 
           zorder=12, markeredgecolor='white', markeredgewidth=2)
    
    # Features
    horizon = Circle((0, 0), metric.r_plus, color=HORIZON_COLOR, zorder=10)
    ax.add_patch(horizon)
    horizon_ring = Circle((0, 0), metric.r_plus, fill=False,
                          color=HORIZON_RING_COLOR, linewidth=2, zorder=11)
    ax.add_patch(horizon_ring)
    
    if spin > 0:
        ergosphere = Circle((0, 0), metric.r_ergo, fill=False,
                           color=ERGOSPHERE_COLOR, linestyle='--', linewidth=2,
                           zorder=9, alpha=0.7)
        ax.add_patch(ergosphere)
        
        # Rotation indicator
        arrow_props = dict(arrowstyle='->', lw=3, color='yellow', alpha=0.8)
        ax.annotate('', xy=(0, metric.r_ergo + 1), 
                   xytext=(0, metric.r_ergo + 3),
                   arrowprops=arrow_props)
    
    # Animation lines
    lines = []
    for traj in trajectories:
        if len(traj) > 0:
            # Color code
            if traj.r[-1] < metric.r_plus * 1.5:
                color = '#FF4444'
            elif traj.r[-1] > 50:
                color = '#44FF44'
            else:
                color = PHOTON_COLOR
            
            line, = ax.plot([], [], color=color, linewidth=2, alpha=0.7)
            lines.append((line, traj))
    
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                   ha='center', fontsize=16, fontweight='bold', color='white')
    
    def init():
        for line, _ in lines:
            line.set_data([], [])
        title.set_text(f'Kerr Scatter: {description}')
        return [l for l, _ in lines] + [title]
    
    def animate(frame):
        for line, traj in lines:
            idx = min(frame * 3, len(traj) - 1)
            line.set_data(traj.x[:idx], traj.y[:idx])
        
        if len(lines) > 0:
            max_idx = min(frame * 3, min(len(traj) for _, traj in lines) - 1)
            if max_idx >= 0 and max_idx < len(lines[0][1].tau):
                tau_val = lines[0][1].tau[max_idx]
                title.set_text(f'Kerr Scatter ({description}): τ = {tau_val:.1f}')
        
        return [l for l, _ in lines] + [title]
    
    max_len = max([len(traj) for _, traj in lines]) if lines else 100
    n_frames = min(max_len // 3, 200)
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=n_frames, interval=50, blit=True)
    
    writer = PillowWriter(fps=20)
    anim.save(f'./results/videos/kerr_{filename}_scatter.gif', writer=writer, dpi=80)
    plt.close()
    
    print(f"Saved: results/videos/kerr_{filename}_scatter.gif")
    print(f"   Time for this spin: {time.time() - start_time:.2f}s")
