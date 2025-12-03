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

print("\n" + "="*70)
print("SCHWARZSCHILD GEODESICS - STATIC PLOTS")
print("="*70)

# ============================================================================
# PLOT 1: Null Geodesics - Impact Parameter Sweep
# ============================================================================

print("\n" + "="*70)
print("PLOT 1: Null Geodesics - Impact Parameter Sweep")
print("="*70)

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

print(f"Critical impact parameter: b_crit = {b_crit:.6f}M")
print(f"Simulating {n_photons} photons...")

for i, b in enumerate(impact_params):
    traj = sim.simulate(
        r0=20.0, phi0=0, impact_param=b,
        is_timelike=False, E=1.0, tau_span=(0, 600),
        radial_direction="inward", label=f"photon_b={b:.2f}"
    )
    trajectories_null.append(traj)
    
    if len(traj) > 0:
        final_r = traj.r[-1]
        if final_r < 3*M:  # Captured
            captured_indices.append(i)
        elif final_r > 50*M:  # Escaped
            escaped_indices.append(i)
        
        # Near critical
        if abs(b - b_crit) < 0.2:
            critical_indices.append(i)

print(f"Captured: {len(captured_indices)}, Escaped: {len(escaped_indices)}")
print(f" Complete in {time.time() - start_time:.2f}s")

# Create figure
print("Creating plot...")
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#1a1a1a')

ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_aspect('equal')
ax.grid(True, alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
ax.set_xlabel('x (M)', fontsize=16, color='white', fontweight='bold')
ax.set_ylabel('y (M)', fontsize=16, color='white', fontweight='bold')
ax.set_title('Null Geodesics: Impact Parameter Dependence', 
             fontsize=18, fontweight='bold', color='white', pad=20)

# Draw critical radii
horizon = Circle((0, 0), metric.r_s, color=HORIZON_COLOR, zorder=10)
ax.add_patch(horizon)
horizon_ring = Circle((0, 0), metric.r_s, fill=False,
                      color=HORIZON_RING_COLOR, linewidth=2.5, zorder=11,
                      label=f'Event Horizon (r = {metric.r_s:.1f}M)')
ax.add_patch(horizon_ring)

photon_sphere = Circle((0, 0), metric.r_photon, fill=False,
                       color=PHOTON_SPHERE_COLOR, linestyle='--', linewidth=2.5,
                       label=f'Photon Sphere (r = {metric.r_photon:.1f}M)', 
                       zorder=9, alpha=0.8)
ax.add_patch(photon_sphere)

# Plot trajectories with color coding
for i, traj in enumerate(trajectories_null):
    if len(traj) > 0:
        b = impact_params[i]
        
        # Color based on fate
        if i in captured_indices:
            color = CAPTURED_COLOR
            linewidth = 1.5
            alpha = 0.7
        elif i in escaped_indices:
            color = ESCAPED_COLOR
            linewidth = 1.5
            alpha = 0.7
        else:
            color = PHOTON_COLOR
            linewidth = 2
            alpha = 0.8
        
        # Highlight critical trajectories
        if i in critical_indices:
            linewidth = 3
            alpha = 1.0
        
        ax.plot(traj.x, traj.y, color=color, linewidth=linewidth, 
               alpha=alpha, zorder=5)
        
        # Mark starting point
        ax.plot(traj.x[0], traj.y[0], 'o', color=color, 
               markersize=6, alpha=0.9, zorder=6)

# Add impact parameter labels for key trajectories
label_indices = [0, len(impact_params)//4, len(impact_params)//2, 
                 3*len(impact_params)//4, -1]
for i in label_indices:
    if i < len(trajectories_null) and len(trajectories_null[i]) > 0:
        traj = trajectories_null[i]
        b = impact_params[i]
        # Label at starting point
        ax.annotate(f'b={b:.1f}M', 
                   xy=(traj.x[0], traj.y[0]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color='white', alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                            alpha=0.7, edgecolor='none'))

# Add critical b_crit line
ax.axhline(y=b_crit, color='yellow', linestyle=':', linewidth=1.5, 
          alpha=0.5, zorder=1)
ax.axhline(y=-b_crit, color='yellow', linestyle=':', linewidth=1.5, 
          alpha=0.5, zorder=1)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=CAPTURED_COLOR, linewidth=2, label=f'Captured (b < {b_crit:.2f}M)'),
    Line2D([0], [0], color=ESCAPED_COLOR, linewidth=2, label=f'Escaped (b > {b_crit:.2f}M)'),
    Line2D([0], [0], color=PHOTON_SPHERE_COLOR, linewidth=2.5, 
           linestyle='--', label='Photon Sphere (3M)'),
    Line2D([0], [0], color=HORIZON_RING_COLOR, linewidth=2.5, label='Event Horizon (2M)'),
    Line2D([0], [0], color='yellow', linewidth=1.5, linestyle=':', 
           label=f'Critical b = {b_crit:.3f}M')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
         framealpha=0.9, facecolor='#2a2a2a', edgecolor='white')

# Info box
info_text = f"Null Geodesics\n"
info_text += f"n = {n_photons} photons\n"
info_text += f"b ∈ [{impact_params[0]:.1f}, {impact_params[-1]:.1f}]M\n"
info_text += f"b_crit = √27M = {b_crit:.4f}M"
ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
       verticalalignment='top', fontsize=11, color='white',
       bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9,
                edgecolor='white', linewidth=1.5))

plt.tight_layout()
plt.savefig('./results/plots/plot1_null_geodesics.png', 
            dpi=200, facecolor='#1a1a1a', bbox_inches='tight')
plt.close()

print(f" Saved: results/plots/plot1_null_geodesics.png")

# ============================================================================
# PLOT 2: Timelike Geodesics - Energy Dependence
# ============================================================================

print("\n" + "="*70)
print("PLOT 2: Timelike Geodesics - Energy Dependence")
print("="*70)

start_time = time.time()

sim.clear()

# Fixed starting radius and impact parameter, vary energy
r0 = 10.0
b = 6.0
E_circ = np.sqrt((r0 - 2)/(r0 - 3)) / np.sqrt(r0)

"""
# Test various energies
energies = [
    (E_circ * 0.80, "E = 0.80 E_circ (deep bound)"),
    (E_circ * 0.90, "E = 0.90 E_circ (bound)"),
    (E_circ * 0.95, "E = 0.95 E_circ (sub-circular)"),
    (E_circ * 1.00, "E = E_circ (circular)"),
    (E_circ * 1.05, "E = 1.05 E_circ (super-circular)"),
    (E_circ * 1.20, "E = 1.20 E_circ (unbound)"),
]
"""

energies = [
    (0.4, "E = 0.4"),
    (0.5, "E = 0.5"),
    (0.6, "E = 0.6"),
    (0.7, "E = 0.7"),
    (0.8, "E = 0.8"),
    (0.9, "E = 0.9"),
]
# Clear labels that E>=1 escapes

trajectories_timelike = []
energy_labels = []

print(f"Starting radius: r0 = {r0}M")
print(f"Impact parameter: b = {b}M")
print(f"Circular energy: E_circ = {E_circ:.4f}")
print(f"\nSimulating {len(energies)} trajectories...")

for E, label in energies:
    traj = sim.simulate(
        r0=r0, phi0=0, impact_param=b,
        is_timelike=True, E=E, tau_span=(0, 800),
        radial_direction="inward", label=label
    )
    trajectories_timelike.append(traj)
    energy_labels.append(label)
    
    if len(traj) > 0:
        min_r = np.min(traj.r)
        final_r = traj.r[-1]
        print(f"  {label}: min_r={min_r:.2f}M, final_r={final_r:.2f}M")

print(f"Complete in {time.time() - start_time:.2f}s")

# Create figure
print("Creating plot...")
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#1a1a1a')

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')
ax.grid(True, alpha=0.15, color='gray', linestyle=':', linewidth=0.5)
ax.set_xlabel('x (M)', fontsize=16, color='white', fontweight='bold')
ax.set_ylabel('y (M)', fontsize=16, color='white', fontweight='bold')
ax.set_title('Timelike Geodesics: Energy Dependence', 
             fontsize=18, fontweight='bold', color='white', pad=20)

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

print(f" Saved: results/plots/plot2_timelike_energy.png")

# ============================================================================
# PLOT 3: Impact Parameter Phase Space
# ============================================================================

print("\n" + "="*70)
print("PLOT 3: Impact Parameter Phase Space")
print("="*70)

start_time = time.time()

sim.clear()

# Compute fate for many impact parameters
b_values = np.linspace(2, 10, 80)
r0_test = 20.0

fates = []  # 0 = captured, 1 = escaped
min_radii = []
max_phi = []

print(f"Computing phase space for {len(b_values)} impact parameters...")

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
        print(f"  Progress: {i+1}/{len(b_values)}")

print(f" Complete in {time.time() - start_time:.2f}s")

# Create figure with subplots
print("Creating plot...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
fig.patch.set_facecolor('#1a1a1a')

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#1a1a1a')
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_xlabel('Impact Parameter b (M)', fontsize=14, 
                  color='white', fontweight='bold')

# Panel 1: Fate (captured vs escaped)
ax1.scatter(b_values, fates, c=fates, cmap='RdYlGn', 
           s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
ax1.axvline(b_crit, color='yellow', linestyle='--', linewidth=2, 
           label=f'b_crit = {b_crit:.3f}M')
ax1.set_ylabel('Fate', fontsize=14, color='white', fontweight='bold')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Captured', 'Escaped'], fontsize=12)
ax1.set_title('Photon Fate vs Impact Parameter', 
             fontsize=16, fontweight='bold', color='white', pad=15)
ax1.legend(fontsize=11, framealpha=0.9, facecolor='#2a2a2a')

# Panel 2: Minimum radius achieved
ax2.plot(b_values, min_radii, 'o-', color=PHOTON_COLOR, 
        linewidth=2, markersize=4, alpha=0.8)
ax2.axhline(metric.r_s, color=HORIZON_RING_COLOR, linestyle='-', 
           linewidth=2, label='Event Horizon (2M)')
ax2.axhline(metric.r_photon, color=PHOTON_SPHERE_COLOR, linestyle='--', 
           linewidth=2, label='Photon Sphere (3M)')
ax2.axvline(b_crit, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_ylabel('Minimum Radius r_min (M)', fontsize=14, 
              color='white', fontweight='bold')
ax2.set_ylim(0, 20)
ax2.set_title('Closest Approach vs Impact Parameter', 
             fontsize=16, fontweight='bold', color='white', pad=15)
ax2.legend(fontsize=11, framealpha=0.9, facecolor='#2a2a2a')

# Panel 3: Total angle traversed
ax3.plot(b_values, np.array(max_phi)/np.pi, 'o-', color='#FF6B6B', 
        linewidth=2, markersize=4, alpha=0.8)
ax3.axvline(b_crit, color='yellow', linestyle='--', linewidth=2, 
           label=f'b_crit = {b_crit:.3f}M')
ax3.set_ylabel('Total Angle Δφ (π radians)', fontsize=14, 
              color='white', fontweight='bold')
ax3.set_title('Angular Deflection vs Impact Parameter', 
             fontsize=16, fontweight='bold', color='white', pad=15)
ax3.legend(fontsize=11, framealpha=0.9, facecolor='#2a2a2a')

# Style tick labels
for ax in [ax1, ax2, ax3]:
    ax.tick_params(colors='white', which='both')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('white')

plt.tight_layout()
plt.savefig('./results/plots/plot3_phase_space.png', 
            dpi=200, facecolor='#1a1a1a', bbox_inches='tight')
plt.close()

print(f" Saved: results/plots/plot3_phase_space.png")

# ============================================================================
# PLOT 4: Effective Potential Visualization
# ============================================================================

print("\n" + "="*70)
print("PLOT 4: Effective Potential")
print("="*70)

# Compute effective potentials
r_range = np.linspace(2.1, 20, 1000)

# Null geodesics
L_values_null = [2*M, 3*M, 4*M, np.sqrt(27)*M, 6*M]
V_eff_null = []

for L in L_values_null:
    V = (L**2 / r_range**2) * (1 - 2*M/r_range)
    V_eff_null.append(V)

# Timelike geodesics
L_values_timelike = [2*M, 3*M, 4*M, 2*np.sqrt(3)*M]
V_eff_timelike = []

for L in L_values_timelike:
    V = (1 + L**2/r_range**2) * (1 - 2*M/r_range)
    V_eff_timelike.append(V)

# Create figure
print("Creating plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#1a1a1a')

for ax in [ax1, ax2]:
    ax.set_facecolor('#1a1a1a')
    ax.grid(True, alpha=0.2, color='gray')
    ax.set_xlabel('Radius r (M)', fontsize=14, color='white', fontweight='bold')
    ax.set_ylabel('Effective Potential V_eff', fontsize=14, 
                 color='white', fontweight='bold')

# Panel 1: Null geodesics
colors_null = plt.cm.Blues(np.linspace(0.4, 0.9, len(L_values_null)))

for i, (V, L) in enumerate(zip(V_eff_null, L_values_null)):
    label = f'L = {L/M:.2f}M'
    if abs(L - np.sqrt(27)*M) < 0.01:
        label += ' (critical)'
        linewidth = 3
        linestyle = '-'
    else:
        linewidth = 2
        linestyle = '-'
    
    ax1.plot(r_range, V, color=colors_null[i], linewidth=linewidth,
            linestyle=linestyle, label=label, alpha=0.8)

ax1.axvline(metric.r_s, color=HORIZON_RING_COLOR, linestyle='--', 
           linewidth=2, alpha=0.5, label='Horizon (2M)')
ax1.axvline(metric.r_photon, color=PHOTON_SPHERE_COLOR, linestyle='--', 
           linewidth=2, alpha=0.5, label='Photon Sphere (3M)')
ax1.axhline(0, color='white', linestyle=':', linewidth=1, alpha=0.3)
ax1.set_ylim(-0.05, 0.15)
ax1.set_title('Effective Potential: Null Geodesics', 
             fontsize=16, fontweight='bold', color='white', pad=15)
ax1.legend(fontsize=10, framealpha=0.9, facecolor='#2a2a2a', loc='upper right')

# Panel 2: Timelike geodesics
colors_timelike = plt.cm.Reds(np.linspace(0.4, 0.9, len(L_values_timelike)))

for i, (V, L) in enumerate(zip(V_eff_timelike, L_values_timelike)):
    label = f'L = {L/M:.2f}M'
    if abs(L - 2*np.sqrt(3)*M) < 0.01:
        label += ' (ISCO)'
        linewidth = 3
    else:
        linewidth = 2
    
    ax2.plot(r_range, V, color=colors_timelike[i], linewidth=linewidth,
            label=label, alpha=0.8)

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