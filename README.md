
# Photon Trajectories Around a Schwarzschild Black Hole

## **Project Overview**
This project simulates photon trajectories around a **Schwarzschild black hole** using lightlike geodesics, integrates their motion numerically, and visualizes the photon paths. The simulation is visualized through static plots and animations, demonstrating gravitational lensing and photon bending near the event horizon.

---

## **Theory: Schwarzschild Black Hole**

The **Schwarzschild solution** is a static, spherically symmetric solution to Einstein's field equations in vacuum, describing the spacetime around a non-rotating, uncharged black hole. The metric is given by:

\[
ds^2 = -\left(1 - rac{2GM}{r}
ight) dt^2 + \left(1 - rac{2GM}{r}
ight)^{-1} dr^2 + r^2 d	heta^2 + r^2 \sin^2	heta d\phi^2
\]

Where:
- \( G \) is the gravitational constant,
- \( M \) is the mass of the black hole,
- \( r \) is the radial coordinate,
- \( 	heta \) and \( \phi \) are the angular coordinates.

### **Key Concepts**
1. **Event Horizon**: The radius at which the escape velocity equals the speed of light, \( r_s = 2GM/c^2 \). Photons or any object inside this radius cannot escape.
2. **Photon Sphere**: A spherical region at \( r = 3GM/c^2 \), where light can orbit the black hole.

Photons traveling along **null geodesics** follow paths determined by the Schwarzschild metric, and we are particularly interested in the **bending of light** as it passes near the black hole.

---

## **Computations: Geodesic Equation**

The motion of photons is governed by the **geodesic equation**:

\[
rac{d^2 x^\mu}{d	au^2} + \Gamma^\mu_{lphaeta} rac{dx^lpha}{d	au} rac{dx^eta}{d	au} = 0
\]

Where:
- \( \Gamma^\mu_{lphaeta} \) are the **Christoffel symbols** (which encode the curvature of spacetime),
- \( x^\mu \) are the coordinates of the photon,
- \( 	au \) is the proper time, but for photons, we use the affine parameter.

### **Christoffel Symbols**
For the Schwarzschild metric, the nonzero Christoffel symbols \( \Gamma^\mu_{lphaeta} \) are computed using the metric tensor. These are required to solve the geodesic equation and determine the trajectory of photons.

The **null condition** for photons means that their 4-velocity is lightlike, i.e., \( g_{\mu
u} u^\mu u^
u = 0 \).

---

## **Project Structure**

1. **SchwarzschildMetric**: Defines the Schwarzschild metric and computes the Christoffel symbols.
2. **GeodesicIntegrator**: Integrates the geodesic equation using `scipy.integrate.solve_ivp`.
3. **PhotonSimulation**: Generates initial photon conditions and simulates their motion.
4. **Trajectory Class**: Stores and visualizes photon paths.
5. **Animation**: Creates an animation of photon motion around the black hole.

---

## **Installation & Setup**

1. Clone or download the project repository.
2. Install required Python libraries:
   ```bash
   pip install numpy scipy matplotlib ffmpeg
   ```

---

## **Workflow**

### **1. Metric & Geodesic Integration (Days 1-2)**

- **Implement the Schwarzschild Metric**: 
  - Define the metric and compute the Christoffel symbols.
  - Use symbolic computation (e.g., `sympy`) for Christoffel symbols.

- **Geodesic Solver**: 
  - Implement numerical integration of the geodesic equation using `scipy.integrate.solve_ivp`.

- **Testing**: 
  - Test with a single photon starting at a fixed radius and angular velocity.

### **2. Photon Simulation (Days 3-4)**

- **Photon Initial Conditions**: 
  - Generate multiple photons with varying angular velocities (or impact parameters).
  - Normalize the 4-velocity to ensure lightlike geodesics.

- **Simulate Multiple Photons**: 
  - Integrate each photon’s motion using the geodesic solver.

- **Testing**: 
  - Simulate a few photons and verify expected bending near the black hole.

### **3. Visualization (Days 5-6)**

- **Trajectory Class**: 
  - Store photon data (position and velocity) in the `Trajectory` class.

- **Basic Plotting**: 
  - Plot photon trajectories in the equatorial plane, showing photon bending.
  - Label key features like the photon sphere and event horizon.

### **4. Animation & Video (Days 7-8)**

- **Photon Path Animation**: 
  - Use `matplotlib.animation` to animate photon trajectories over time.

- **Save to Video/GIF**: 
  - Export the animation to a video file (e.g., `.mp4` or `.gif`) using `ffmpeg` or `matplotlib`.

### **5. Final Testing & Documentation (Days 9-10)**

- **Test Full Photon Bundle**: 
  - Run the simulation with multiple photons and visualize their trajectories in one plot.

- **Optimize Performance**: 
  - Ensure smooth performance with a larger photon bundle or long simulation times.

- **Documentation**: 
  - Comment code and prepare a README file with usage instructions.

---

## **Usage**

1. **Run the Simulation**:  
   Modify the initial conditions for photons (radius \( r_0 \), azimuthal velocity \( \dot{\phi} \)) to generate different photon paths.
   Example:
   ```python
   phidot_vals = np.linspace(0.015, 0.08, 12)  # Varying azimuthal velocity
   r0 = 15  # Initial radius
   tau_span = (0, 80)  # Time span
   trajectories = simulate_photon_bundle(r0, phidot_vals, tau_span)
   plot_trajectories(trajectories)
   ```

2. **Generate Animation**:  
   After simulating photon paths, animate the trajectories and save to video:
   ```python
   animate_trajectories(trajectories)
   save_video('photon_trajectories.mp4')
   ```

---

## **Deliverables**

1. **Source Code**: Python script or Jupyter notebook for simulating photon trajectories.
2. **Visualization**: Static plots of photon paths and a video/GIF of photon trajectories around the black hole.
3. **Documentation**: Code comments and a README file with setup and usage instructions.

---

## **Extensions (Optional)**

- **Kerr Black Hole**: Simulate photon paths around a rotating black hole.
- **3D Simulations**: Extend the simulation to 3D or off-equatorial orbits.
- **Interactive Visualizations**: Use `Plotly` or `Mayavi` for more interactive visualizations.

---

## **References**

- Schwarzschild Solution: [Wikipedia - Schwarzschild Metric](https://en.wikipedia.org/wiki/Schwarzschild_metric)
- Geodesic Equation: [General Relativity Textbooks](https://www.amazon.com/Gravitation-Charles-Misner/dp/0691177790)
- Python Libraries:  
  - [SymPy](https://www.sympy.org/)
  - [SciPy](https://scipy.org/)
  - [Matplotlib](https://matplotlib.org/)
