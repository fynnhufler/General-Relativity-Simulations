#!/usr/bin/env python3
"""
Static Plots for Schwarzschild Geodesics
==========================================
Creates publication-quality static visualizations showing:
- Null geodesics with varying impact parameters
- Timelike geodesics with energy constraints
- Critical radii and dynamical features
"""

import sys
sys.path.insert(0, './src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.collections import LineCollection
import time

from geodesics import SchwarzschildMetric, GeodesicSimulation

# Dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['savefig.facecolor'] = '#1a1a1a'

# Colors
PHOTON_COLOR = '#87CEEB'
MASSIVE_COLOR = '#FF6B6B'
CAPTURED_COLOR = '#FF4444'
ESCAPED_COLOR = '#44FF44'
HORIZON_COLOR = 'black'
HORIZON_RING_COLOR = 'white'
PHOTON_SPHERE_COLOR = '#FFA500'
ISCO_COLOR = '#00FF00'

metric = SchwarzschildMetric(mass=1.0)
M = metric.M

start_time = time.time()

sim = GeodesicSimulation(metric)

# Range of impact parameters around critical value
b_crit = metric.b_crit_photon
n_photons = 25
impact_params = np.linspace(2.5, 9, n_photons)

# Identify which are captured vs escaped
captured_indices = []
escaped_indices = []
critical_indices = []

trajectories_null = []

# Draw critical radii
horizon = Circle((0, 0), metric.r_s, color=HORIZON_COLOR, zorder=10)
ax.add_patch(horizon)
horizon_ring = Circle((0, 0), metric.r_s, fill=False,
                      color=HORIZON_RING_COLOR, linewidth=2.5, zorder=11)
ax.add_patch(horizon_ring)

isco = Circle((0, 0), metric.r_isco, fill=False,
              color=ISCO_COLOR, linestyle='--', linewidth=2.5,
              label=f'ISCO (r = {metric.r_isco:.1f}M)', zorder=9, alpha=0.8)
ax.add_patch(isco)

# Color gradient for energies
colors_energy = plt.cm.plasma(np.linspace(0.1, 0.9, len(energies)))

for i, (traj, label) in enumerate(zip(trajectories_timelike, energy_labels)):
    if len(traj) > 0:
        color = colors_energy[i]
        ax.plot(traj.x, traj.y, color=color, linewidth=2.5, 
               alpha=0.8, label=label, zorder=5)
        
        # Mark starting point
        ax.plot(traj.x[0], traj.y[0], 'o', color=color, 
               markersize=8, markeredgecolor='white', 
               markeredgewidth=1.5, zorder=6)

# Mark starting radius circle
start_circle = Circle((0, 0), r0, fill=False, color='white',
                      linestyle=':', linewidth=1, alpha=0.3, zorder=2)
ax.add_patch(start_circle)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9, 
         facecolor='#2a2a2a', edgecolor='white')

# Info box
info_text = f"Timelike Geodesics\n"
info_text += f"r₀ = {r0}M\n"
info_text += f"b = {b}M\n"
info_text += f"E_circ = {E_circ:.4f}\n"
info_text += f"E_ISCO = {np.sqrt(8/9):.4f}"
ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
       verticalalignment='top', horizontalalignment='right',
       fontsize=11, color='white',
       bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9,
                edgecolor='white', linewidth=1.5))

plt.tight_layout()
plt.savefig('./results/plots/plot2_timelike_energy.png', 
            dpi=200, facecolor='#1a1a1a', bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 3: Impact Parameter Phase Space
# ============================================================================

for i, b in enumerate(b_values):
    traj = sim.simulate(
        r0=r0_test, phi0=0, impact_param=b,
        is_timelike=False, E=1.0, tau_span=(0, 400),
        radial_direction="inward"
    )
    
    if len(traj) > 0:
        min_r = np.min(traj.r)
        final_r = traj.r[-1]
        delta_phi = abs(traj.phi[-1] - traj.phi[0])
        
        min_radii.append(min_r)
        max_phi.append(delta_phi)
        
        if final_r < 5*M:  # Captured
            fates.append(0)
        else:  # Escaped
            fates.append(1)
    else:
        min_radii.append(np.nan)
        max_phi.append(np.nan)
        fates.append(-1)
    
    if (i+1) % 20 == 0:
        
ax2.axvline(metric.r_s, color=HORIZON_RING_COLOR, linestyle='--', 
           linewidth=2, alpha=0.5, label='Horizon (2M)')
ax2.axvline(metric.r_isco, color=ISCO_COLOR, linestyle='--', 
           linewidth=2, alpha=0.5, label='ISCO (6M)')
ax2.axhline(1, color='white', linestyle=':', linewidth=1, alpha=0.3)
ax2.set_ylim(0.85, 1.05)
ax2.set_title('Effective Potential: Timelike Geodesics', 
             fontsize=16, fontweight='bold', color='white', pad=15)
ax2.legend(fontsize=10, framealpha=0.9, facecolor='#2a2a2a', loc='upper right')

# Style tick labels
for ax in [ax1, ax2]:
    ax.tick_params(colors='white', which='both')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('white')

plt.tight_layout()
plt.savefig('./results/plots/plot4_effective_potential.png', 
            dpi=200, facecolor='#1a1a1a', bbox_inches='tight')
plt.close()