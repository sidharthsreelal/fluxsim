# FluxSim

A lightweight 3D rigid body physics engine written from scratch in Python. Built for learning and experimentation - no external physics libraries, just pure math with numpy under the hood.

## Features

- Quaternion-based rotations (no gimbal lock)
- RK4 integration for stable physics
- GJK + EPA collision detection
- Impulse-based solver with friction

## Structure

| File | What it does |
|------|--------------|
| `math_util.py` | Vec3, Mat3, Quat - all the linear algebra primitives |
| `body.py` | RigidBody class, inertia tensors, RK4/semi-implicit integrators |
| `collision.py` | GJK algorithm + EPA for penetration depth, supports sphere/box/convex |
| `solver.py` | Sequential impulse solver, handles restitution and friction |
| `main.py` | Demo runner with visualization |
| `test_engine.py` | Unit tests for each module |

## Installation

Make sure you have Python 3.8+

```bash
git clone <repo-url>
cd flux
pip install -r requirements.txt
```

Dependencies are just numpy and matplotlib.

## Usage

Run demos:
```bash
python main.py 1    # bouncing balls
python main.py 2    # spinning box
python main.py 3    # collision chain
python main.py all  # run everything
```

Or just `python main.py` for interactive menu.

## Testing

```bash
python test_engine.py
```

Runs tests for math utils, body dynamics, collision detection, and solver.
