# FluxSim: High-Performance Python Physics Engine

FluxSim is a custom-built 3D rigid body physics engine designed for high numerical stability and physical accuracy. It is implemented entirely in Python with minimal dependencies (NumPy only for core math arrays).

## Core Architecture

### 1. Mathematical Foundation (`math_util.py`)
- **Quaternion Rotations**: Full implementation of quaternion algebra including Hamilton product, conjugation, and normalization.
- **Stable Integration**: Uses the derivative $\dot{q} = \frac{1}{2}\omega q$ for accumulating rotations, avoiding gimbal lock completely.
- **Data Structures**: Custom `Vec3`, `Mat3`, and `Quat` classes optimized for readability and performance.

### 2. Rigid Body Dynamics (`body.py`)
- **RK4 Integrator**: Implements 4th-order Runge-Kutta integration for position and velocity, providing superior energy conservation compared to standard Euler methods.
- **Inertia Tensors**: Full $3 \times 3$ inertia tensor support with world-space transformation $I_{world} = R I_{body} R^T$.
- **Gyroscopic Forces**: Accurate simulation of torque-free precession and gyroscopic stabilization (Dzhanibekov effect capable).

### 3. Collision Detection (`collision.py`)
- **GJK Algorithm**: Gilbert-Johnson-Keerthi algorithm for detecting intersections between distinct convex shapes (Spheres, Boxes, Convex Hulls).
- **EPA Algorithm**: Expanding Polytope Algorithm to extract precise penetration depth and contact normals for collision resolution.
- **Support Functions**: efficient support mapping for generic convex shapes.

### 4. Constraint Solver (`solver.py`)
- **Impulse Resolution**: Velocity-based impulse solver handling restitution (bounciness) and varying mass ratios.
- **Friction Model**: Coulomb friction model applying tangent impulses to simulate surface roughness and spin decay.
- **Stabilization**: Baumgarte stabilization to correct position drift (penetration) over time.

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/fluxsim.git
cd fluxsim
```

### Set Up Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Prerequisites
- Python 3.8+
- NumPy (for matrix operations)
- Matplotlib (for visualization)

### Running the Demo
Launch the interactive visualization with three distinct demos:
```bash
python main.py
```
1. **Bouncing Balls**: Demonstrates restitution and gravity.
2. **Spinning Box**: Shows angular momentum and friction.
3. **Collision Chain**: Verifies momentum transfer (Newton's Cradle).

### Running Tests
Execute the comprehensive test suite to verify physics accuracy:
```bash
python test_engine.py
```

## Project Structure
```text
fluxsim/
├── body.py          # Physics state and integration
├── collision.py     # GJK/EPA collision detection
├── main.py          # Visualization and demos
├── math_util.py     # Vector/matrix/quaternion math
├── solver.py        # Impulse and friction solver
├── test_engine.py   # Unit and simulation tests
└── requirements.txt # Dependencies
```
