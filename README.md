# FluxSim

A 3D physics engine I built from scratch in Python. No external physics libs, just NumPy for the arrays.

## What's In Here

- **math_util.py** - Vectors, matrices, quaternions. The usual stuff
- **body.py** - Rigid body state, RK4 integration (way more stable than Euler)
- **collision.py** - GJK + EPA for convex shapes. Took forever to debug
- **solver.py** - Impulse-based collision response, friction

## The Math

Using quaternions for rotation so no gimbal lock headaches. Inertia tensors are full 3x3, transformed to world space like `I_world = R * I_body * R^T`

RK4 for integration because Euler was giving me energy drift problems

## Setup

```bash
git clone https://github.com/sidharthsreelal/fluxsim.git
cd fluxsim
pip install -r requirements.txt
```

Or with a venv if you prefer:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Running It

```bash
python main.py
```

Gives you 3 demos:
1. Bouncing balls - gravity + restitution
2. Spinning box - angular momentum stuff
3. Collision chain - like Newton's cradle

## Tests

```bash
python test_engine.py
```

Should all pass. Checks energy conservation, collision detection, and that nothing falls through the ground

## Notes

- No comments in the code, its meant to be readable as-is
- Variable names are short (inv_m, ang_vel, etc) 
- Tried to keep it dense but not unreadable

That's about it
