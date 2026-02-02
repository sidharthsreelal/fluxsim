import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from math import pi, cos, sin
from typing import List, Tuple

from math_util import Vec3, Quat
from body import RigidBody, World, integrate_rk4
from collision import SphereCollider, BoxCollider, GroundPlane, detect_collision, detect_ground_collision
from solver import ContactSolver

_SPHERE_CACHE = {}

def sphere_wireframe(center: Vec3, radius: float, segments: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if segments not in _SPHERE_CACHE:
        u = np.linspace(0, 2 * np.pi, segments)
        v = np.linspace(0, np.pi, segments)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        _SPHERE_CACHE[segments] = (x, y, z)

    ux, uy, uz = _SPHERE_CACHE[segments]
    return center.x + radius * ux, center.y + radius * uy, center.z + radius * uz

def box_corners(body: RigidBody, half: Vec3) -> np.ndarray:
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                local = Vec3(half.x * sx, half.y * sy, half.z * sz)
                world = body.local_to_world(local)
                corners.append([world.x, world.y, world.z])
    return np.array(corners)

def draw_box(ax: Axes3D, body: RigidBody, half: Vec3, color: str = 'blue') -> None:
    c = box_corners(body, half)
    
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for e in edges:
        ax.plot3D([c[e[0], 0], c[e[1], 0]],
                  [c[e[0], 1], c[e[1], 1]],
                  [c[e[0], 2], c[e[1], 2]], color=color)

def run_bouncing_balls() -> None:
    print("Running Bouncing Balls Demo...")
    
    world = World()
    world.gravity = Vec3(0, -9.81, 0)
    
    bodies = []
    colliders = []
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i in range(5):
        b = RigidBody(mass=1.0, pos=Vec3((i - 2) * 1.5, 3 + i * 0.5, 0))
        b.set_sphere_inertia(0.3)
        b.restitution = 0.6 + i * 0.05
        b.friction = 0.3
        world.add_body(b)
        bodies.append(b)
        colliders.append(SphereCollider(b, 0.3))
    
    ground = GroundPlane(y=0)
    solver = ContactSolver(iterations=10)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    dt = 1.0 / 60.0
    history: List[List[List[float]]] = [[] for _ in bodies]
    
    for frame in range(360):
        for b in bodies:
            integrate_rk4(b, dt, world.gravity)
        
        solver.clear()
        
        for col in colliders:
            contact = detect_ground_collision(col, ground)
            if contact:
                solver.add_contact(contact)
        
        for i in range(len(colliders)):
            for j in range(i + 1, len(colliders)):
                contact = detect_collision(colliders[i], colliders[j])
                if contact:
                    solver.add_contact(contact)
        
        solver.solve()
        
        for i, b in enumerate(bodies):
            history[i].append([b.pos.x, b.pos.y, b.pos.z])
    
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 6)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('FluxSim - Bouncing Balls')
    
    xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-3, 3, 10))
    ax.plot_surface(xx, np.zeros_like(xx), yy, alpha=0.2, color='gray')
    
    for i, (b, col) in enumerate(zip(bodies, colors)):
        x, y, z = sphere_wireframe(b.pos, 0.3, 8)
        ax.plot_wireframe(x, y, z, color=col, alpha=0.7)
        
        h = np.array(history[i])
        ax.plot3D(h[:, 0], h[:, 1], h[:, 2], color=col, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('bouncing_balls.png', dpi=150)
    print("Saved: bouncing_balls.png")

def run_spinning_box() -> None:
    print("Running Spinning Box Demo...")
    
    box = RigidBody(mass=2.0, pos=Vec3(0, 2, 0))
    box.set_box_inertia(1, 0.5, 0.8)
    box.ang_vel = Vec3(2, 5, 1)
    box.restitution = 0.3
    box.friction = 0.5
    
    box_col = BoxCollider(box, Vec3(0.5, 0.25, 0.4))
    ground = GroundPlane(y=0)
    solver = ContactSolver(iterations=10)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    dt = 1.0 / 60.0
    history = []
    
    gravity = Vec3(0, -9.81, 0)
    
    for frame in range(480):
        integrate_rk4(box, dt, gravity)
        
        solver.clear()
        contact = detect_ground_collision(box_col, ground)
        if contact:
            solver.add_contact(contact)
        solver.solve()
        
        if frame % 4 == 0:
            history.append((box.pos.copy(), box.rot.copy()))
    
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 4)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('FluxSim - Spinning Box')
    
    xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
    ax.plot_surface(xx, np.zeros_like(xx), yy, alpha=0.2, color='gray')
    
    for i, (pos, rot) in enumerate(history[::10]):
        alpha = 0.2 + 0.8 * (i / (len(history) // 10))
        temp_body = RigidBody(mass=1, pos=pos, rot=rot)
        draw_box(ax, temp_body, Vec3(0.5, 0.25, 0.4), color=(0, 0, alpha))
    
    draw_box(ax, box, Vec3(0.5, 0.25, 0.4), color='red')
    
    positions = [h[0] for h in history]
    xs = [p.x for p in positions]
    ys = [p.y for p in positions]
    zs = [p.z for p in positions]
    ax.plot3D(xs, ys, zs, 'r-', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('spinning_box.png', dpi=150)
    print("Saved: spinning_box.png")

def run_collision_chain() -> None:
    print("Running Collision Chain Demo...")
    
    bodies = []
    colliders = []
    
    for i in range(6):
        b = RigidBody(mass=1.0, pos=Vec3(i * 1.0, 0.5, 0))
        b.set_sphere_inertia(0.4)
        b.restitution = 0.95
        b.friction = 0.1
        bodies.append(b)
        colliders.append(SphereCollider(b, 0.4))
    
    bodies[0].vel = Vec3(5, 0, 0)
    
    ground = GroundPlane(y=0)
    solver = ContactSolver(iterations=10)
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    dt = 1.0 / 60.0
    
    for frame in range(300):
        for b in bodies:
            integrate_rk4(b, dt, Vec3(0, -9.81, 0))
        
        solver.clear()
        
        for col in colliders:
            contact = detect_ground_collision(col, ground)
            if contact:
                solver.add_contact(contact)
        
        for i in range(len(colliders)):
            for j in range(i + 1, len(colliders)):
                contact = detect_collision(colliders[i], colliders[j])
                if contact:
                    solver.add_contact(contact)
        
        solver.solve()
    
    ax.clear()
    ax.set_xlim(-2, 10)
    ax.set_ylim(0, 3)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('FluxSim - Collision Chain')
    
    xx, yy = np.meshgrid(np.linspace(-2, 10, 10), np.linspace(-2, 2, 10))
    ax.plot_surface(xx, np.zeros_like(xx), yy, alpha=0.2, color='gray')
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i, b in enumerate(bodies):
        x, y, z = sphere_wireframe(b.pos, 0.4, 10)
        ax.plot_wireframe(x, y, z, color=colors[i], alpha=0.8)
        
        ax.quiver(b.pos.x, b.pos.y, b.pos.z, 
                  b.vel.x * 0.2, b.vel.y * 0.2, b.vel.z * 0.2,
                  color=colors[i], arrow_length_ratio=0.3)
    
    plt.tight_layout()
    plt.savefig('collision_chain.png', dpi=150)
    print("Saved: collision_chain.png")

def print_system_info() -> None:
    print("-" * 50)
    print("FluxSim Physics Engine")
    print("-" * 50)
    print("Modules:")
    print("  math_util.py : Vector/Matrix/Quaternion math")
    print("  body.py      : RigidBody dynamics, RK4 integration")
    print("  collision.py : GJK + EPA collision detection")
    print("  solver.py    : Impulse-based collision resolution")
    print("-" * 50)

def main() -> None:
    parser = argparse.ArgumentParser(description="FluxSim Physics Demos")
    parser.add_argument('demo', nargs='?', choices=['1', '2', '3', 'all'], help="Demo to run: 1=Balls, 2=Box, 3=Chain, all=All")
    args = parser.parse_args()
    
    print_system_info()
    
    if args.demo:
        choice = args.demo
    else:
        print("Select demo:")
        print("  1. Bouncing Balls")
        print("  2. Spinning Box")
        print("  3. Collision Chain")
        print("  4. Run All")

        try:
            choice = input("\nEnter choice (1-4): ").strip()
        except EOFError:
            choice = '4'
    
    if choice == '1':
        run_bouncing_balls()
    elif choice == '2':
        run_spinning_box()
    elif choice == '3':
        run_collision_chain()
    elif choice in ['4', 'all']:
        run_bouncing_balls()
        run_spinning_box()
        run_collision_chain()
    else:
        print("Invalid choice, running Bouncing Balls.")
        run_bouncing_balls()

if __name__ == "__main__":
    main()
