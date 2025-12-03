#!/usr/bin/env python3
"""
Additional Creative Animations - CORRECTED ENERGIES
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

PHOTON_COLOR = '#87CEEB'
MASSIVE_COLOR = '#FF6B6B'
HORIZON_COLOR = 'black'
HORIZON_RING_COLOR = 'white'
PHOTON_SPHERE_COLOR = '#FFA500'
ISCO_COLOR = '#00FF00'

metric = SchwarzschildMetric(mass=1.0)

print("\n" + "="*70)
print("ADDITIONAL CREATIVE ANIMATIONS - CORRECTED ENERGIES")
print("="*70)

# ============================================================================
# VIDEO 5: Particle Spray (Radial Explosion) - CORRECTED
# ============================================================================

print("\n" + "="*70)
print("VIDEO 5: Particle Spray (Radial Explosion)")
print("="*70)

start_time = time.time()

sim = GeodesicSimulation(metric)

# Start from a point, shoot in all directions
n_spray = 30
r0_spray = 12.0  # Starting radius
angles = np.linspace(0, 2*np.pi, n_spray, endpoint=False)

trajectories_spray = []

print("Simulating particle spray with correct energies...")
for i, angle in enumerate(angles):
    b = np.random.uniform(4, 9)
    
    # Mix of photons and massive
    if i < n_spray // 2:
        # Photons
        traj = sim.simulate(
            r0=r0_spray, phi0=angle, impact_param=b,
            is_timelike=False, E=1.0, tau_span=(0, 400),
            radial_direction="outward", label=f"spray_photon_{i}"
        )
        trajectories_spray.append(('photon', traj))
    else:
        # Massive particles with AUTO energy
        traj = sim.simulate(
            r0=r0_spray, phi0=angle, impact_param=b,
            is_timelike=True, E=None,  # AUTO!
            tau_span=(0, 400),
            radial_direction="outward", label=f"spray_massive_{i}"
        )
        trajectories_spray.append(('massive', traj))

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

# Mark starting point
ax.plot(r0_spray, 0, 'o', color='yellow', markersize=10, 
        markeredgecolor='white', markeredgewidth=2, zorder=12,
        label='Explosion Point')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=PHOTON_COLOR, label='Photons'),
    Patch(facecolor=MASSIVE_COLOR, label='Massive'),
    Patch(facecolor='yellow', label='Origin')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

lines_spray = []
for ptype, traj in trajectories_spray:
    if len(traj) > 0:
        color = PHOTON_COLOR if ptype == 'photon' else MASSIVE_COLOR
        line, = ax.plot([], [], color=color, linewidth=2, alpha=0.7)
        lines_spray.append((line, traj))

title5 = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                ha='center', fontsize=16, fontweight='bold', color='white')

def init5():
    for line, _ in lines_spray:
        line.set_data([], [])
    title5.set_text('Particle Spray: τ = 0')
    return [l for l, _ in lines_spray] + [title5]

def animate5(frame):
    for line, traj in lines_spray:
        idx = min(frame * 3, len(traj) - 1)
        line.set_data(traj.x[:idx], traj.y[:idx])
    
    if len(lines_spray) > 0:
        max_idx = min(frame * 3, min(len(traj) for _, traj in lines_spray) - 1)
        if max_idx >= 0 and max_idx < len(lines_spray[0][1].tau):
            tau_val = lines_spray[0][1].tau[max_idx]
            title5.set_text(f'Particle Spray: τ = {tau_val:.1f}')
    
    return [l for l, _ in lines_spray] + [title5]

max_len5 = max([len(traj) for _, traj in lines_spray]) if lines_spray else 100
n_frames5 = min(max_len5 // 3, 150)

anim5 = FuncAnimation(fig, animate5, init_func=init5,
                     frames=n_frames5, interval=50, blit=True)

writer = PillowWriter(fps=20)
anim5.save('./results/videos/video5_particle_spray_corrected.gif', writer=writer, dpi=80)
plt.close()

print(f"Saved: results/videos/video5_particle_spray_corrected.gif")

# ============================================================================
# VIDEO 6: Accretion Disk Simulation - CORRECTED
# ============================================================================

print("\n" + "="*70)
print("VIDEO 6: Accretion Disk (Massive Particles) - CORRECTED")
print("="*70)

start_time = time.time()

sim.clear()

# Ring of particles at various radii - ALL OUTSIDE ISCO
n_disk = 25
r_disk = np.random.uniform(7, 15, n_disk)  # Start outside ISCO
phi_disk = np.random.uniform(0, 2*np.pi, n_disk)

trajectories_disk = []

print("Simulating accretion disk with physically correct energies...")
for i in range(n_disk):
    r0 = r_disk[i]
    phi0 = phi_disk[i]
    
    # Calculate circular orbit parameters
    E_circ = np.sqrt((r0 - 2)/(r0 - 3)) / np.sqrt(r0)
    L_circ = np.sqrt(r0**3 / (r0 - 3))
    
    # Use sub-circular energy for slow inspiral
    # Lower reduction = slower inspiral
    reduction = 0.92 + 0.06 * np.random.rand()  # 92-98% of circular
    
    E = E_circ * reduction
    L = L_circ * reduction
    b = L / E
    
    traj = sim.simulate(
        r0=r0, phi0=phi0, impact_param=b,
        is_timelike=True, E=E, tau_span=(0, 1500),
        radial_direction="tangent", label=f"disk_{i}"
    )
    trajectories_disk.append((r0, traj))

elapsed = time.time() - start_time
print(f"✅ Complete in {elapsed:.2f}s")

# Check if any crossed ISCO
crossed_isco = sum(1 for _, t in trajectories_disk if len(t) > 0 and np.min(t.r) < 6.0)
print(f"Particles that crossed ISCO: {crossed_isco}/{n_disk}")

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

# Color gradient by radius
colors_disk = plt.cm.YlOrRd(np.linspace(0.3, 1, n_disk))

lines_disk = []
for i, (r0, traj) in enumerate(trajectories_disk):
    if len(traj) > 0:
        line, = ax.plot([], [], color=colors_disk[i], linewidth=2, alpha=0.7)
        lines_disk.append((line, traj))

title6 = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                ha='center', fontsize=16, fontweight='bold', color='white')

def init6():
    for line, _ in lines_disk:
        line.set_data([], [])
    title6.set_text('Accretion Disk (Corrected Physics): τ = 0')
    return [l for l, _ in lines_disk] + [title6]

def animate6(frame):
    for line, traj in lines_disk:
        idx = min(frame * 4, len(traj) - 1)
        line.set_data(traj.x[:idx], traj.y[:idx])
    
    if len(lines_disk) > 0:
        max_idx = min(frame * 4, min(len(traj) for _, traj in lines_disk) - 1)
        if max_idx >= 0 and max_idx < len(lines_disk[0][1].tau):
            tau_val = lines_disk[0][1].tau[max_idx]
            title6.set_text(f'Accretion Disk: τ = {tau_val:.1f}')
    
    return [l for l, _ in lines_disk] + [title6]

max_len6 = max([len(traj) for _, traj in lines_disk]) if lines_disk else 100
n_frames6 = min(max_len6 // 4, 250)

anim6 = FuncAnimation(fig, animate6, init_func=init6,
                     frames=n_frames6, interval=50, blit=True)

writer = PillowWriter(fps=20)
anim6.save('./results/videos/video6_accretion_disk_corrected.gif', writer=writer, dpi=80)
plt.close()

print(f"💾 Saved: results/videos/video6_accretion_disk_corrected.gif")

# ============================================================================
# VIDEO 7: Critical Photon Orbits - UNCHANGED (photons only)
# ============================================================================

print("\n" + "="*70)
print("VIDEO 7: Critical Photon Orbits (Unstable Photon Sphere)")
print("="*70)

start_time = time.time()

sim.clear()

# Photons very close to critical impact parameter
b_crit = metric.b_crit_photon
n_critical = 15

# Slightly above and below b_crit
impact_params_critical = np.concatenate([
    np.linspace(b_crit * 0.995, b_crit * 0.999, 7),
    [b_crit],
    np.linspace(b_crit * 1.001, b_crit * 1.005, 7)
])

trajectories_critical = []

print("Simulating critical orbits...")
for i, b in enumerate(impact_params_critical):
    traj = sim.simulate(
        r0=20.0, phi0=0, impact_param=b,
        is_timelike=False, E=1.0, tau_span=(0, 1000),
        radial_direction="inward", label=f"critical_{i}"
    )
    trajectories_critical.append(traj)

elapsed = time.time() - start_time
print(f"✅ Complete in {elapsed:.2f}s")

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
                       color=PHOTON_SPHERE_COLOR, linestyle='-', linewidth=2.5,
                       label='Photon Sphere (3M)', zorder=9)
ax.add_patch(photon_sphere)

ax.legend(loc='upper right', fontsize=12)

lines_critical = []
for i, traj in enumerate(trajectories_critical):
    if len(traj) > 0:
        line, = ax.plot([], [], color=PHOTON_COLOR, linewidth=2, alpha=0.8)
        lines_critical.append((line, traj))

title7 = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                ha='center', fontsize=16, fontweight='bold', color='white')

def init7():
    for line, _ in lines_critical:
        line.set_data([], [])
    title7.set_text('Critical Photon Orbits: τ = 0')
    return [l for l, _ in lines_critical] + [title7]

def animate7(frame):
    for line, traj in lines_critical:
        idx = min(frame * 4, len(traj) - 1)
        line.set_data(traj.x[:idx], traj.y[:idx])
    
    if len(lines_critical) > 0:
        max_idx = min(frame * 4, min(len(traj) for _, traj in lines_critical) - 1)
        if max_idx >= 0 and max_idx < len(lines_critical[0][1].tau):
            tau_val = lines_critical[0][1].tau[max_idx]
            title7.set_text(f'Critical Photon Orbits: τ = {tau_val:.1f}')
    
    return [l for l, _ in lines_critical] + [title7]

max_len7 = max([len(traj) for _, traj in lines_critical]) if lines_critical else 100
n_frames7 = min(max_len7 // 4, 250)

anim7 = FuncAnimation(fig, animate7, init_func=init7,
                     frames=n_frames7, interval=50, blit=True)

writer = PillowWriter(fps=20)
anim7.save('./results/videos/video7_critical_orbits.gif', writer=writer, dpi=80)
plt.close()

print(f"Saved: results/videos/video7_critical_orbits.gif")