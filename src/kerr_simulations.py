#!/usr/bin/env python3
"""
Kerr Black Hole Simulations - Multiple Spins
=============================================
Simulates photon geodesics around rotating black holes
with different spin parameters: a/M = 0, 0.5, 0.9, 0.998
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyBboxPatch
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
print("KERR BLACK HOLE SIMULATIONS")
print("Photon Geodesics around Rotating Black Holes")
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
    
    # Simulate photons with different impact parameters
    n_photons = 25
    impact_params = np.linspace(2.5, 9, n_photons)
    
    trajectories = []
    
    print(f"\nSimulating {n_photons} photons...")
    for i, b in enumerate(impact_params):
        traj = sim.simulate(
            r0=20.0,
            phi0=0,
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
            
            if i % 5 == 0:  # Print every 5th
                print(f"  b={b:.2f}M: {len(traj)} points, {fate}")
    
    elapsed = time.time() - start_time
    print(f"✅ Simulation complete in {elapsed:.2f}s")
    print(f"   Generated {len(trajectories)} trajectories")
    
    # ========================================================================
    # STATIC PLOT
    # ========================================================================
    
    print("Creating static plot...")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_xlabel('x (M)', fontsize=14, color='white')
    ax.set_ylabel('y (M)', fontsize=14, color='white')
    ax.set_title(f'Kerr Black Hole: {description}', 
                 fontsize=16, fontweight='bold', color='white')
    
    # Draw features
    # Horizon
    horizon = Circle((0, 0), metric.r_plus, color=HORIZON_COLOR, zorder=10)
    ax.add_patch(horizon)
    horizon_ring = Circle((0, 0), metric.r_plus, fill=False,
                          color=HORIZON_RING_COLOR, linewidth=2, zorder=11)
    ax.add_patch(horizon_ring)
    
    # Ergosphere (at equator)
    if spin > 0:
        ergosphere = Circle((0, 0), metric.r_ergo, fill=False,
                           color=ERGOSPHERE_COLOR, linestyle='--', linewidth=2,
                           label=f'Ergosphere ({metric.r_ergo:.2f}M)', zorder=9, alpha=0.7)
        ax.add_patch(ergosphere)
    
    # Plot trajectories
    for traj in trajectories:
        if len(traj) > 0:
            ax.plot(traj.x, traj.y, color=PHOTON_COLOR, linewidth=1.5, alpha=0.6)
            # Mark starting point
            ax.plot(traj.x[0], traj.y[0], 'o', color=PHOTON_COLOR, 
                   markersize=4, alpha=0.8)
    
    # Add rotation indicator
    if spin > 0:
        # Arrow showing rotation direction
        arrow_props = dict(arrowstyle='->', lw=3, color='yellow', alpha=0.8)
        ax.annotate('', xy=(0, metric.r_ergo + 1), 
                   xytext=(0, metric.r_ergo + 3),
                   arrowprops=arrow_props)
        ax.text(0.5, metric.r_ergo + 3.5, 'Rotation', 
               ha='center', fontsize=11, color='yellow', fontweight='bold')
    
    # Info box
    info_text = f"Spin: a/M = {spin}\n"
    info_text += f"Horizon: r₊ = {metric.r_plus:.3f}M\n"
    if spin > 0:
        info_text += f"Ergosphere: {metric.r_ergo:.1f}M\n"
    info_text += f"Photons: {len(trajectories)}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10, color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    if spin > 0:
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'./results/plots/kerr_{filename}_static.png', 
                dpi=150, facecolor='#1a1a1a')
    plt.close()
    
    print(f"💾 Saved: results/plots/kerr_{filename}_static.png")
    
    # ========================================================================
    # ANIMATION
    # ========================================================================
    
    print("Creating animation...")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_xlabel('x (M)', fontsize=14, color='white')
    ax.set_ylabel('y (M)', fontsize=14, color='white')
    
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
            line, = ax.plot([], [], color=PHOTON_COLOR, linewidth=2, alpha=0.7)
            lines.append((line, traj))
    
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                   ha='center', fontsize=16, fontweight='bold', color='white')
    
    info_box = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                      verticalalignment='top', fontsize=10, color='white',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def init():
        for line, _ in lines:
            line.set_data([], [])
        title.set_text(f'Kerr: {description}')
        info_box.set_text(f'a/M = {spin}\nr₊ = {metric.r_plus:.2f}M')
        return [l for l, _ in lines] + [title, info_box]
    
    def animate(frame):
        for line, traj in lines:
            idx = min(frame * 3, len(traj) - 1)
            line.set_data(traj.x[:idx], traj.y[:idx])
        
        if len(lines) > 0:
            max_idx = min(frame * 3, min(len(traj) for _, traj in lines) - 1)
            if max_idx >= 0 and max_idx < len(lines[0][1].tau):
                tau_val = lines[0][1].tau[max_idx]
                title.set_text(f'Kerr ({description}): τ = {tau_val:.1f}')
        
        return [l for l, _ in lines] + [title]
    
    max_len = max([len(traj) for _, traj in lines]) if lines else 100
    n_frames = min(max_len // 3, 200)
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=n_frames, interval=50, blit=True)
    
    writer = PillowWriter(fps=20)
    anim.save(f'./results/videos/kerr_{filename}.gif', writer=writer, dpi=80)
    plt.close()
    
    print(f"💾 Saved: results/videos/kerr_{filename}.gif")
    print(f"   Time for plot + animation: {time.time() - start_time:.2f}s")

# ============================================================================
# COMPARISON PLOT - All spins side by side
# ============================================================================

print("\n" + "="*70)
print("Creating comparison plot...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
fig.patch.set_facecolor('#1a1a1a')

for idx, (spin, description, filename) in enumerate(spin_configs):
    ax = axes[idx // 2, idx % 2]
    ax.set_facecolor('#1a1a1a')
    
    # Recreate metric and simulate
    metric = KerrMetric(mass=1.0, spin=spin)
    sim = GeodesicSimulation(metric)
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_xlabel('x (M)', fontsize=12, color='white')
    ax.set_ylabel('y (M)', fontsize=12, color='white')
    ax.set_title(description, fontsize=14, fontweight='bold', color='white')
    
    # Quick simulation
    impact_params = np.linspace(3, 9, 20)
    for b in impact_params:
        traj = sim.simulate(
            r0=20.0, phi0=0, impact_param=b,
            is_timelike=False, E=1.0, tau_span=(0, 400),
            radial_direction="inward"
        )
        if len(traj) > 0:
            ax.plot(traj.x, traj.y, color=PHOTON_COLOR, 
                   linewidth=1.2, alpha=0.6)
    
    # Features
    horizon = Circle((0, 0), metric.r_plus, color=HORIZON_COLOR, zorder=10)
    ax.add_patch(horizon)
    horizon_ring = Circle((0, 0), metric.r_plus, fill=False,
                         color=HORIZON_RING_COLOR, linewidth=1.5, zorder=11)
    ax.add_patch(horizon_ring)
    
    if spin > 0:
        ergosphere = Circle((0, 0), metric.r_ergo, fill=False,
                           color=ERGOSPHERE_COLOR, linestyle='--', 
                           linewidth=1.5, zorder=9, alpha=0.6)
        ax.add_patch(ergosphere)
    
    # Info
    info_text = f"a/M = {spin}\nr₊ = {metric.r_plus:.2f}M"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10, color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

plt.suptitle('Kerr Black Holes: Spin Comparison', 
            fontsize=18, fontweight='bold', color='white', y=0.995)
plt.tight_layout()
plt.savefig('./results/plots/kerr_comparison_all.png', 
            dpi=150, facecolor='#1a1a1a')
plt.close()