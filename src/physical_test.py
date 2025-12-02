#!/usr/bin/env python3
"""
Test script to verify the geodesic simulator fix
Tests that photons with b > b_crit escape and b < b_crit are captured
"""

import numpy as np
from geodesics import SchwarzschildMetric, GeodesicSimulation

def test_photon_capture_and_escape():
    """Test that photons behave correctly around critical impact parameter"""
    
    print("\n" + "="*70)
    print("VERIFICATION TEST: Photon Capture vs Escape")
    print("="*70)
    
    # Create metric
    metric = SchwarzschildMetric(mass=1.0)
    sim = GeodesicSimulation(metric)
    
    b_crit = metric.b_crit_photon
    print(f"\nCritical impact parameter: b_crit = {b_crit:.6f}M")
    print(f"Expected: b_crit = √27 M = {np.sqrt(27):.6f}M")
    
    # Test cases
    test_cases = [
        (b_crit * 0.6, "CAPTURE", "should be captured"),
        (b_crit * 0.8, "CAPTURE", "should be captured"),
        (b_crit * 1.2, "ESCAPE", "should escape"),
        (b_crit * 1.5, "ESCAPE", "should escape"),
    ]
    
    print("\n" + "-"*70)
    print(f"{'Impact Parameter':<20} {'Expected':<15} {'Result':<15} {'Status'}")
    print("-"*70)
    
    all_passed = True
    
    for b, expected, description in test_cases:
        # Simulate
        traj = sim.simulate(
            r0=15.0,
            impact_param=b,
            is_timelike=False,
            tau_span=(0, 150),
            radial_direction="inward",
            label=f"b={b:.3f}M"
        )
        
        # Check result
        if len(traj) > 0:
            final_r = traj.r[-1]
            if final_r > 50:
                result = "ESCAPE"
            elif final_r < metric.r_s * 2:
                result = "CAPTURE"
            else:
                result = "ORBIT"
        else:
            result = "ERROR"
        
        # Verify
        passed = (result == expected)
        status = "PASS" if passed else "FAIL"
        
        print(f"b = {b:6.3f}M ({description:<20}) {expected:<15} {result:<15} {status}")
        
        if not passed:
            all_passed = False
    
    print("-"*70)
    
    if all_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
    
    print("="*70 + "\n")
    
    return all_passed


def test_energy_conservation():
    """Test that energy is conserved along geodesics"""
    
    print("\n" + "="*70)
    print("VERIFICATION TEST: Energy Conservation")
    print("="*70)
    
    metric = SchwarzschildMetric(mass=1.0)
    sim = GeodesicSimulation(metric)
    
    # Simulate a photon
    traj = sim.simulate(
        r0=15.0,
        impact_param=6.0,
        is_timelike=False,
        E=1.0,
        tau_span=(0, 100),
        radial_direction="inward"
    )
    
    if len(traj) < 10:
        print("Trajectory too short for testing")
        return False
    
    # Check energy conservation E = f * dt/dτ
    energies = []
    for i in range(len(traj)):
        r = traj.r[i]
        ut = traj.ut[i]
        f = metric.metric_factor(r)
        E = f * ut
        energies.append(E)
    
    energies = np.array(energies)
    E_mean = np.mean(energies)
    E_std = np.std(energies)
    E_var = np.max(energies) - np.min(energies)
    
    print(f"\nEnergy statistics:")
    print(f"  Mean E = {E_mean:.10f}")
    print(f"  Std  E = {E_std:.10e}")
    print(f"  Variation = {E_var:.10e}")
    
    # Check if energy is conserved (within numerical precision)
    tolerance = 1e-6
    conserved = E_var < tolerance
    
    if conserved:
        print(f"\nPASS: Energy conserved within tolerance ({tolerance})")
    else:
        print(f"\nFAIL: Energy not conserved. Variation = {E_var:.10e}")
    
    print("="*70 + "\n")
    
    return conserved


def test_null_constraint():
    """Test that null geodesic constraint is satisfied"""
    
    print("\n" + "="*70)
    print("VERIFICATION TEST: Null Geodesic Constraint")
    print("="*70)
    
    metric = SchwarzschildMetric(mass=1.0)
    sim = GeodesicSimulation(metric)
    
    # Simulate a photon
    traj = sim.simulate(
        r0=15.0,
        impact_param=6.0,
        is_timelike=False,
        tau_span=(0, 100),
        radial_direction="inward"
    )
    
    if len(traj) < 10:
        print("Trajectory too short for testing")
        return False
    
    # Check null constraint: g_μν u^μ u^ν = 0
    # In Schwarzschild: -f(dt/dτ)² + (1/f)(dr/dτ)² + r²(dφ/dτ)² = 0
    
    constraints = []
    for i in range(len(traj)):
        r = traj.r[i]
        ut = traj.ut[i]
        ur = traj.ur[i]
        uphi = traj.uphi[i]
        
        f = metric.metric_factor(r)
        
        constraint = -f * ut**2 + (1/f) * ur**2 + r**2 * uphi**2
        constraints.append(constraint)
    
    constraints = np.array(constraints)
    max_violation = np.max(np.abs(constraints))
    mean_violation = np.mean(np.abs(constraints))
    
    print(f"\nNull constraint statistics:")
    print(f"  Max |g_μν u^μ u^ν| = {max_violation:.10e}")
    print(f"  Mean |g_μν u^μ u^ν| = {mean_violation:.10e}")
    
    # Check if constraint is satisfied
    tolerance = 1e-8
    satisfied = max_violation < tolerance
    
    if satisfied:
        print(f"\nPASS: Null constraint satisfied within tolerance ({tolerance})")
    else:
        print(f"\nFAIL: Null constraint violated! Max = {max_violation:.10e}")
    
    print("="*70 + "\n")
    
    return satisfied


def main():
    """Run all tests"""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  GEODESIC SIMULATOR - VERIFICATION TESTS".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Capture vs Escape", test_photon_capture_and_escape()))
    results.append(("Energy Conservation", test_energy_conservation()))
    results.append(("Null Constraint", test_null_constraint()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<30} {status}")
    
    print("="*70)
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
    
    print("\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)