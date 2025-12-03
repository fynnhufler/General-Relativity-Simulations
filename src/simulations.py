#!/usr/bin/env python3
"""
Advanced Schwarzschild Animations - CORRECTED ENERGIES
=======================================================
Multiple scenarios with photons and massive particles
NOW WITH PHYSICALLY CORRECT ENERGIES FOR MASSIVE PARTICLES
"""

import sys
sys.path.insert(0, './src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import time

from geodesics import SchwarzschildMetric, GeodesicSimulation

# Dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['savefig.facecolor'] = '#1a1a1a'

# Colors
PHOTON_COLOR = '#87CEEB'  # Light blue
MASSIVE_COLOR = '#FF6B6B'  # Light red
HORIZON_COLOR = 'black'
HORIZON_RING_COLOR = 'white'
PHOTON_SPHERE_COLOR = '#FFA500'
ISCO_COLOR = '#00FF00'  # Green for ISCO

metric = SchwarzschildMetric(mass=1.0)

print("\n" + "="*70)
print("ADVANCED SCHWARZSCHILD ANIMATIONS - CORRECTED ENERGIES")
print("Multiple Scenarios | Photons + Massive Particles")
print("="*70)

# ============================================================================
# VIDEO 1: Random Particle Cloud (Mixed) - CORRECTED
# ============================================================================

print("\n" + "="*70)
print("VIDEO 1: Random Particle Cloud (Photons + Massive Particles)")
print("="*70)

start_time = time.time()

sim = GeodesicSimulation(metric)

# Mix of photons and massive particles
n_photons = 25
n_massive = 25

trajectories_mixed = []

# Random photons
print("Simulating photons...")
for i in range(n_photons):
    r0 = np.random.uniform(10, 25)
    phi0 = np.random.uniform(0, 2*np.pi)
    b = np.random.uniform(3, 9)
    direction = np.random.choice(['inward', 'outward', 'tangent'])
    
    traj = sim.simulate(
        r0=r0, phi0=phi0, impact_param=b,
        is_timelike=False, E=1.0,  # Correct for photons
        tau_span=(0, 500),
        radial_direction=direction, label=f"photon_{i}"
    )
    trajectories_mixed.append(('photon', traj))

# Random massive particles with CORRECT energies
print("Simulating massive particles with correct energies...")
for i in range(n_massive):
    r0 = np.random.uniform(10, 25)
    phi0 = np.random.uniform(0, 2*np.pi)
    b = np.random.uniform(3, 9)
    direction = np.random.choice(['inward', 'outward', 'tangent'])
    
    # AUTO-CALCULATE correct energy (E = 0.95 * E_circular)
    traj = sim.simulate(
        r0=r0, phi0=phi0, impact_param=b,
        is_timelike=True, E=None,  # AUTO: correct energy!
        tau_span=(0, 500),
        radial_direction=direction, label=f"massive_{i}"
    )
    trajectories_mixed.append(('massive', traj))

elapsed = time.time() - start_time
print(f"Complete in {elapsed:.2f}s")

# Animate
print("Creating animation...")
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#1a1a1a')

ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, color='gray')
ax.set_xlabel('x (M)', fontsize=14, color='white')
ax.set_ylabel('y (M)', fontsize=14, color='white')

# Features
horizon = Circle((0, 0), metric.r_s, color=HORIZON_COLOR, zorder=10)
ax.add_patch(horizon)
horizon_ring = Circle((0, 0), metric.r_s, fill=False,
                      color=HORIZON_RING_COLOR, linewidth=1.5, zorder=11)
ax.add_patch(horizon_ring)
photon_sphere = Circle((0, 0), metric.r_photon, fill=False,
                       color=PHOTON_SPHERE_COLOR, linestyle='--', linewidth=1.5, zorder=9)
ax.add_patch(photon_sphere)
isco = Circle((0, 0), metric.r_isco, fill=False,
              color=ISCO_COLOR, linestyle=':', linewidth=1.5, 
              label='ISCO (6M)', zorder=8)
ax.add_patch(isco)

# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=PHOTON_COLOR, label='Photons'),
    Patch(facecolor=MASSIVE_COLOR, label='Massive Particles'),
    Patch(facecolor=PHOTON_SPHERE_COLOR, label='Photon Sphere (3M)'),
    Patch(facecolor=ISCO_COLOR, label='ISCO (6M)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

lines = []
for ptype, traj in trajectories_mixed:
    if len(traj) > 0:
        color = PHOTON_COLOR if ptype == 'photon' else MASSIVE_COLOR
        line, = ax.plot([], [], color=color, linewidth=2, alpha=0.7)
        lines.append((line, traj))

title = ax.text(0.5, 1.02, '', transform=ax.transAxes,
               ha='center', fontsize=16, fontweight='bold', color='white')

def init1():
    for line, _ in lines:
        line.set_data([], [])
    title.set_text('Random Particle Cloud: τ = 0')
    return [l for l, _ in lines] + [title]

def animate1(frame):
    for line, traj in lines:
        idx = min(frame * 3, len(traj) - 1)
        line.set_data(traj.x[:idx], traj.y[:idx])
    
    if len(lines) > 0:
        max_idx = min(frame * 3, min(len(traj) for _, traj in lines) - 1)
        if max_idx >= 0 and max_idx < len(lines[0][1].tau):
            tau_val = lines[0][1].tau[max_idx]
            title.set_text(f'Random Particle Cloud: τ = {tau_val:.1f}')
    
    return [l for l, _ in lines] + [title]

max_len = max([len(traj) for _, traj in lines]) if lines else 100
n_frames = min(max_len // 3, 200)

anim1 = FuncAnimation(fig, animate1, init_func=init1,
                     frames=n_frames, interval=50, blit=True)

writer = PillowWriter(fps=20)
anim1.save('./results/videos/video1_random_cloud_corrected.gif', writer=writer, dpi=80)
plt.close()

print(f"Saved: results/videos/video1_random_cloud_corrected.gif")

# ============================================================================
# VIDEO 2: Parallel Photon Beam (Gravitational Lensing) - UNCHANGED
# ============================================================================

print("\n" + "="*70)
print("VIDEO 2: Parallel Photon Beam (Gravitational Lensing)")
print("="*70)

start_time = time.time()

sim.clear()


# Parallel beam from left side
n_beam = 20
y_positions = np.linspace(-15, 0, n_beam)  # Different impact parameters
x_start = -20

trajectories_beam = []

print("Simulating parallel beam...")
for i, y0 in enumerate(y_positions):
    # Start position: far left, at different y
    r0 = np.sqrt(x_start**2 + y0**2)
    phi0 = np.arctan2(y0, x_start)
    
    # Impact parameter is approximately |y0| for parallel beam
    b = abs(y0)
    
    traj = sim.simulate(
        r0=r0, phi0=phi0, impact_param=b,
        is_timelike=False, E=1.0, tau_span=(0, 400),
        radial_direction="inward", label=f"beam_{i}"
    )
    trajectories_beam.append(traj)
"""
#Parallel beam from left side - CORRECTED
n_beam = 20
y_positions = np.linspace(-15, 15, n_beam)
x_start = -50  # CHANGED: Start farther away

trajectories_beam = []

print("Simulating parallel beam (moving in +x direction)...")
for i, y0 in enumerate(y_positions):
    r0 = np.sqrt(x_start**2 + y0**2)
    phi0 = np.arctan2(y0, x_start)
    b = abs(y0)
    
    traj = sim.simulate(
        r0=r0, phi0=phi0, impact_param=b,
        is_timelike=False, E=1.0, tau_span=(0, 500),
        radial_direction="auto",  #CHANGED: Proper parallel motion
        label=f"beam_{i}"
    )
"""
elapsed = time.time() - start_time
print(f"Complete in {elapsed:.2f}s")

# Animate
print("Creating animation...")
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#1a1a1a')

ax.set_xlim(-35, 35)
ax.set_ylim(-20, 20)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, color='gray')
ax.set_xlabel('x (M)', fontsize=14, color='white')
ax.set_ylabel('y (M)', fontsize=14, color='white')

# Features
horizon = Circle((0, 0), metric.r_s, color=HORIZON_COLOR, zorder=10)
ax.add_patch(horizon)
horizon_ring = Circle((0, 0), metric.r_s, fill=False,
                      color=HORIZON_RING_COLOR, linewidth=1.5, zorder=11)
ax.add_patch(horizon_ring)
photon_sphere = Circle((0, 0), metric.r_photon, fill=False,
                       color=PHOTON_SPHERE_COLOR, linestyle='--', linewidth=1.5,
                       label='Photon Sphere', zorder=9)
ax.add_patch(photon_sphere)

lines_beam = []
for traj in trajectories_beam:
    if len(traj) > 0:
        line, = ax.plot([], [], color=PHOTON_COLOR, linewidth=2, alpha=0.7)
        lines_beam.append((line, traj))

title2 = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                ha='center', fontsize=16, fontweight='bold', color='white')

def init2():
    for line, _ in lines_beam:
        line.set_data([], [])
    title2.set_text('Parallel Photon Beam: τ = 0')
    return [l for l, _ in lines_beam] + [title2]

def animate2(frame):
    for line, traj in lines_beam:
        idx = min(frame * 3, len(traj) - 1)
        line.set_data(traj.x[:idx], traj.y[:idx])
    
    if len(lines_beam) > 0:
        max_idx = min(frame * 3, min(len(traj) for _, traj in lines_beam) - 1)
        if max_idx >= 0 and max_idx < len(lines_beam[0][1].tau):
            tau_val = lines_beam[0][1].tau[max_idx]
            title2.set_text(f'Parallel Photon Beam: τ = {tau_val:.1f}')
    
    return [l for l, _ in lines_beam] + [title2]

max_len2 = max([len(traj) for _, traj in lines_beam]) if lines_beam else 100
n_frames2 = min(max_len2 // 3, 150)

anim2 = FuncAnimation(fig, animate2, init_func=init2,
                     frames=n_frames2, interval=50, blit=True)

writer = PillowWriter(fps=20)
anim2.save('./results/videos/video2_parallel_beam.gif', writer=writer, dpi=80)
plt.close()

print(f"Saved: results/videos/video2_parallel_beam.gif")

# ============================================================================
# VIDEO 3: Photons vs Massive Particles (Direct Comparison) - CORRECTED
# ============================================================================

print("\n" + "="*70)
print("VIDEO 3: Photons vs Massive Particles (Direct Comparison)")
print("="*70)

start_time = time.time()

sim.clear()

# Same starting conditions for both
n_compare = 15
impact_params_compare = np.linspace(4, 8, n_compare)

trajectories_compare = []

print("Simulating photons...")
for i, b in enumerate(impact_params_compare):
    traj_photon = sim.simulate(
        r0=20.0, phi0=0, impact_param=b,
        is_timelike=False, E=1.0, tau_span=(0, 600),
        radial_direction="inward", label=f"photon_{i}"
    )
    trajectories_compare.append(('photon', traj_photon))

print("Simulating massive particles with correct energies...")
for i, b in enumerate(impact_params_compare):
    # AUTO-CALCULATE correct energy (not E=1.1!)
    traj_massive = sim.simulate(
        r0=20.0, phi0=0, impact_param=b,
        is_timelike=True, E=None,  # AUTO: correct energy!
        tau_span=(0, 600),
        radial_direction="inward", label=f"massive_{i}"
    )
    trajectories_compare.append(('massive', traj_massive))

elapsed = time.time() - start_time
print(f"Complete in {elapsed:.2f}s")

# Animate
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
horizon = Circle((0, 0), metric.r_s, color=HORIZON_COLOR, zorder=10)
ax.add_patch(horizon)
horizon_ring = Circle((0, 0), metric.r_s, fill=False,
                      color=HORIZON_RING_COLOR, linewidth=1.5, zorder=11)
ax.add_patch(horizon_ring)
photon_sphere = Circle((0, 0), metric.r_photon, fill=False,
                       color=PHOTON_SPHERE_COLOR, linestyle='--', linewidth=1.5, zorder=9)
ax.add_patch(photon_sphere)
isco = Circle((0, 0), metric.r_isco, fill=False,
              color=ISCO_COLOR, linestyle=':', linewidth=1.5, zorder=8)
ax.add_patch(isco)

legend_elements = [
    Patch(facecolor=PHOTON_COLOR, label='Photons (m=0)'),
    Patch(facecolor=MASSIVE_COLOR, label='Massive (m>0, E=auto)'),
    Patch(facecolor=PHOTON_SPHERE_COLOR, label='Photon Sphere (3M)'),
    Patch(facecolor=ISCO_COLOR, label='ISCO (6M)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

lines_compare = []
for ptype, traj in trajectories_compare:
    if len(traj) > 0:
        color = PHOTON_COLOR if ptype == 'photon' else MASSIVE_COLOR
        line, = ax.plot([], [], color=color, linewidth=2, alpha=0.7)
        lines_compare.append((line, traj))

title3 = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                ha='center', fontsize=16, fontweight='bold', color='white')

def init3():
    for line, _ in lines_compare:
        line.set_data([], [])
    title3.set_text('Photons vs Massive Particles: τ = 0')
    return [l for l, _ in lines_compare] + [title3]

def animate3(frame):
    for line, traj in lines_compare:
        idx = min(frame * 3, len(traj) - 1)
        line.set_data(traj.x[:idx], traj.y[:idx])
    
    if len(lines_compare) > 0:
        max_idx = min(frame * 3, min(len(traj) for _, traj in lines_compare) - 1)
        if max_idx >= 0 and max_idx < len(lines_compare[0][1].tau):
            tau_val = lines_compare[0][1].tau[max_idx]
            title3.set_text(f'Photons vs Massive Particles: τ = {tau_val:.1f}')
    
    return [l for l, _ in lines_compare] + [title3]

max_len3 = max([len(traj) for _, traj in lines_compare]) if lines_compare else 100
n_frames3 = min(max_len3 // 3, 200)

anim3 = FuncAnimation(fig, animate3, init_func=init3,
                     frames=n_frames3, interval=50, blit=True)

writer = PillowWriter(fps=20)
anim3.save('./results/videos/video3_comparison_corrected.gif', writer=writer, dpi=80)
plt.close()

print(f"Saved: results/videos/video3_comparison_corrected.gif")

# ============================================================================
# VIDEO 4: Orbital Trajectories (Massive Particles Only) - CORRECTED
# ============================================================================

print("\n" + "="*70)
print("VIDEO 4: Orbital Trajectories (Massive Particles)")
print("="*70)

start_time = time.time()

sim.clear()

# Various starting radii outside ISCO
n_orbits = 12
trajectories_orbits = []

print("Simulating orbital trajectories with correct energies...")

# Start at various radii, all OUTSIDE ISCO
r_starts = np.linspace(7.0, 15.0, n_orbits)

for i, r0 in enumerate(r_starts):
    # Calculate circular orbit parameters
    E_circ = np.sqrt((r0 - 2)/(r0 - 3)) / np.sqrt(r0)
    L_circ = np.sqrt(r0**3 / (r0 - 3))
    b_circ = L_circ / E_circ
    
    # Use slightly sub-circular for slow inspiral
    E = E_circ * 0.96
    b = b_circ * 0.96
    
    traj = sim.simulate(
        r0=r0, phi0=0, impact_param=b,
        is_timelike=True, E=E, tau_span=(0, 1000),
        radial_direction="tangent", label=f"orbit_{i}"
    )
    trajectories_orbits.append(traj)

elapsed = time.time() - start_time
print(f"Complete in {elapsed:.2f}s")

# Animate
print("Creating animation...")
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#1a1a1a')

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, color='gray')
ax.set_xlabel('x (M)', fontsize=14, color='white')
ax.set_ylabel('y (M)', fontsize=14, color='white')

# Features
horizon = Circle((0, 0), metric.r_s, color=HORIZON_COLOR, zorder=10)
ax.add_patch(horizon)
horizon_ring = Circle((0, 0), metric.r_s, fill=False,
                      color=HORIZON_RING_COLOR, linewidth=1.5, zorder=11)
ax.add_patch(horizon_ring)
isco = Circle((0, 0), metric.r_isco, fill=False,
              color=ISCO_COLOR, linestyle='--', linewidth=2,
              label='ISCO (6M)', zorder=9)
ax.add_patch(isco)

ax.legend(loc='upper right', fontsize=12)

lines_orbits = []
for traj in trajectories_orbits:
    if len(traj) > 0:
        line, = ax.plot([], [], color=MASSIVE_COLOR, linewidth=2, alpha=0.7)
        lines_orbits.append((line, traj))

title4 = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                ha='center', fontsize=16, fontweight='bold', color='white')

def init4():
    for line, _ in lines_orbits:
        line.set_data([], [])
    title4.set_text('Orbital Trajectories (Massive Particles): τ = 0')
    return [l for l, _ in lines_orbits] + [title4]

def animate4(frame):
    for line, traj in lines_orbits:
        idx = min(frame * 4, len(traj) - 1)
        line.set_data(traj.x[:idx], traj.y[:idx])
    
    if len(lines_orbits) > 0:
        max_idx = min(frame * 4, min(len(traj) for _, traj in lines_orbits) - 1)
        if max_idx >= 0 and max_idx < len(lines_orbits[0][1].tau):
            tau_val = lines_orbits[0][1].tau[max_idx]
            title4.set_text(f'Orbital Trajectories: τ = {tau_val:.1f}')
    
    return [l for l, _ in lines_orbits] + [title4]

max_len4 = max([len(traj) for _, traj in lines_orbits]) if lines_orbits else 100
n_frames4 = min(max_len4 // 4, 200)

anim4 = FuncAnimation(fig, animate4, init_func=init4,
                     frames=n_frames4, interval=50, blit=True)

writer = PillowWriter(fps=20)
anim4.save('./results/videos/video4_orbits_corrected.gif', writer=writer, dpi=80)
plt.close()

print(f"Saved: results/videos/video4_orbits_corrected.gif")