import traceback
import sys
from typing import List, Tuple, Callable

from math_util import Vec3, Mat3, Quat
from body import RigidBody, World, integrate_rk4
from collision import (SphereCollider, BoxCollider, ConvexHullCollider, 
                       GroundPlane, detect_collision, detect_ground_collision)
from solver import ContactSolver, resolve_ground

def test_math_util() -> None:
    print("Testing math_util.py...")
    
    v1 = Vec3(1, 2, 3)
    v2 = Vec3(4, 5, 6)
    
    assert (v1 + v2).x == 5, "Vec3 add failed"
    assert (v1 - v2).x == -3, "Vec3 sub failed"
    assert abs(v1.dot(v2) - 32) < 0.001, "Vec3 dot failed"
    
    cross = v1.cross(v2)
    assert abs(cross.x - (-3)) < 0.001, "Vec3 cross failed"
    
    v_norm = Vec3(3, 0, 0).normalized()
    assert abs(v_norm.x - 1.0) < 0.001, "Vec3 normalize failed"
    
    q = Quat.from_axis_angle(Vec3(0, 1, 0), 3.14159/2)
    rotated = q.rotate_vec(Vec3(1, 0, 0))
    assert abs(rotated.z - (-1)) < 0.01, f"Quat rotation failed: {rotated}"
    
    m = q.to_mat3()
    q2 = Quat.from_mat3(m)
    assert abs(q.w - q2.w) < 0.01, "Quat<->Mat3 conversion failed"
    
    print("PASSED")

def test_body() -> None:
    print("Testing body.py...")
    
    body = RigidBody(mass=1.0, pos=Vec3(0, 10, 0))
    body.set_sphere_inertia(1.0)
    
    gravity = Vec3(0, -9.81, 0)
    dt = 1.0 / 60.0
    
    for _ in range(60):
        integrate_rk4(body, dt, gravity)
    
    expected_y = 10 - 0.5 * 9.81 * 1.0 * 1.0
    assert abs(body.pos.y - expected_y) < 0.1, f"Free fall failed: {body.pos.y} vs {expected_y}"
    
    expected_vy = -9.81 * 1.0
    assert abs(body.vel.y - expected_vy) < 0.1, f"Velocity failed: {body.vel.y} vs {expected_vy}"
    
    print("PASSED")

def test_collision() -> None:
    print("Testing collision.py...")
    
    b1 = RigidBody(mass=1.0, pos=Vec3(0, 0, 0))
    b2 = RigidBody(mass=1.0, pos=Vec3(1.5, 0, 0))
    
    s1 = SphereCollider(b1, 1.0)
    s2 = SphereCollider(b2, 1.0)
    
    contact = detect_collision(s1, s2)
    assert contact is not None, "Sphere-sphere collision missed"
    assert contact.depth > 0, f"Invalid penetration depth: {contact.depth}"
    
    b3 = RigidBody(mass=1.0, pos=Vec3(0, 0, 0))
    b4 = RigidBody(mass=1.0, pos=Vec3(5, 0, 0))
    
    s3 = SphereCollider(b3, 1.0)
    s4 = SphereCollider(b4, 1.0)
    
    contact_none = detect_collision(s3, s4)
    assert contact_none is None, "False positive collision"
    
    b5 = RigidBody(mass=1.0, pos=Vec3(0, 0.8, 0))
    s5 = SphereCollider(b5, 1.0)
    ground = GroundPlane(y=0)
    
    ground_contact = detect_ground_collision(s5, ground)
    assert ground_contact is not None, "Ground collision missed"
    assert ground_contact.depth > 0, f"Invalid ground penetration: {ground_contact.depth}"
    
    print("PASSED")

def test_solver() -> None:
    print("Testing solver.py...")
    
    ball = RigidBody(mass=1.0, pos=Vec3(0, 2, 0))
    ball.set_sphere_inertia(0.5)
    ball.restitution = 0.8
    ball.friction = 0.3
    
    ball_col = SphereCollider(ball, 0.5)
    ground = GroundPlane(y=0)
    solver = ContactSolver(iterations=10)
    
    gravity = Vec3(0, -9.81, 0)
    dt = 1.0 / 60.0
    
    bounced = False
    prev_vy = 0.0
    
    for frame in range(300):
        integrate_rk4(ball, dt, gravity)
        
        contact = detect_ground_collision(ball_col, ground)
        solver.clear()
        if contact:
            solver.add_contact(contact)
        solver.solve()
        
        if ball.vel.y > 0 and prev_vy < 0:
            bounced = True
        prev_vy = ball.vel.y
    
    assert ball.pos.y >= -0.1, f"Ball fell through ground: {ball.pos.y}"
    assert bounced, "Ball did not bounce"
    
    print("PASSED")

def test_full_simulation() -> None:
    print("Testing Full Simulation...")
    
    world = World()
    world.gravity = Vec3(0, -9.81, 0)
    
    bodies = []
    colliders = []
    
    for i in range(3):
        b = RigidBody(mass=1.0, pos=Vec3((i-1) * 2, 3 + i, 0))
        b.set_sphere_inertia(0.4)
        b.restitution = 0.6
        b.friction = 0.4
        world.add_body(b)
        bodies.append(b)
        colliders.append(SphereCollider(b, 0.4))
    
    ground = GroundPlane(y=0)
    solver = ContactSolver(iterations=10)
    
    dt = 1.0 / 60.0
    
    for frame in range(300):
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
    
    all_above = True
    for i, b in enumerate(bodies):
        if b.pos.y < -0.5:
            all_above = False
            print(f"Body {i} fell through: {b.pos.y}")
    
    assert all_above, "Bodies fell through ground"
    print("PASSED")

if __name__ == "__main__":
    print("FluxSim Test Suite\n")
    
    tests: List[Tuple[str, Callable[[], None]]] = [
        ("math_util", test_math_util),
        ("body", test_body),
        ("collision", test_collision),
        ("solver", test_solver),
        ("full_simulation", test_full_simulation),
    ]
    
    failed = []
    
    for name, test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed.append(name)
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            failed.append(name)
    
    print("\n" + "="*50)
    if not failed:
        print("ALL TESTS PASSED!")
    else:
        print(f"FAILED TESTS: {failed}")
    print("="*50)
    
    sys.exit(0 if not failed else 1)
