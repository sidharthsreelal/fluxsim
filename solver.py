from math_util import Vec3, clamp

def resolve_collision(contact, baumgarte=0.2, slop=0.01):
    a = contact.body_a
    b = contact.body_b
    n = contact.normal
    
    if a is None or a.is_static:
        if b is None or b.is_static:
            return
    
    inv_m_a = a.inv_mass if a and not a.is_static else 0.0
    inv_m_b = b.inv_mass if b and not b.is_static else 0.0
    
    r_a = contact.pt - a.pos if a else Vec3.zero()
    r_b = contact.pt - b.pos if b else Vec3.zero()
    
    I_inv_a = a.get_world_inertia_inv() if a and not a.is_static else None
    I_inv_b = b.get_world_inertia_inv() if b and not b.is_static else None
    
    v_a = a.vel + a.ang_vel.cross(r_a) if a else Vec3.zero()
    v_b = b.vel + b.ang_vel.cross(r_b) if b else Vec3.zero()
    v_rel = v_a - v_b
    
    v_n = v_rel.dot(n)
    
    if v_n > 0:
        return
    
    e = min(a.restitution if a else 0.5, b.restitution if b else 0.5)
    
    rn_a = r_a.cross(n)
    rn_b = r_b.cross(n)
    
    ang_term = 0.0
    if I_inv_a:
        ang_term += (I_inv_a * rn_a).cross(r_a).dot(n)
    if I_inv_b:
        ang_term += (I_inv_b * rn_b).cross(r_b).dot(n)
    
    eff_mass = inv_m_a + inv_m_b + ang_term
    
    if eff_mass < 1e-10:
        return
    
    bias = 0.0
    pen = contact.depth - slop
    if pen > 0:
        bias = baumgarte * pen / (1.0/60.0)
    
    j_n = (-(1 + e) * v_n + bias) / eff_mass
    j_n = max(j_n, 0.0)
    
    impulse_n = n * j_n
    
    if a and not a.is_static:
        a.vel = a.vel + impulse_n * inv_m_a
        a.ang_vel = a.ang_vel + I_inv_a * r_a.cross(impulse_n)
    
    if b and not b.is_static:
        b.vel = b.vel - impulse_n * inv_m_b
        b.ang_vel = b.ang_vel - I_inv_b * r_b.cross(impulse_n)
    
    apply_friction(contact, j_n, inv_m_a, inv_m_b, I_inv_a, I_inv_b, r_a, r_b)


def apply_friction(contact, j_n, inv_m_a, inv_m_b, I_inv_a, I_inv_b, r_a, r_b):
    a = contact.body_a
    b = contact.body_b
    n = contact.normal
    
    v_a = a.vel + a.ang_vel.cross(r_a) if a else Vec3.zero()
    v_b = b.vel + b.ang_vel.cross(r_b) if b else Vec3.zero()
    v_rel = v_a - v_b
    
    v_t = v_rel - n * v_rel.dot(n)
    
    if v_t.mag_sq() < 1e-10:
        return
    
    t = v_t.normalized()
    
    rt_a = r_a.cross(t)
    rt_b = r_b.cross(t)
    
    ang_term_t = 0.0
    if I_inv_a:
        ang_term_t += (I_inv_a * rt_a).cross(r_a).dot(t)
    if I_inv_b:
        ang_term_t += (I_inv_b * rt_b).cross(r_b).dot(t)
    
    eff_mass_t = inv_m_a + inv_m_b + ang_term_t
    
    if eff_mass_t < 1e-10:
        return
    
    mu = (a.friction if a else 0.4) * (b.friction if b else 0.4)
    mu = min(mu, 1.0)
    
    j_t = -v_t.mag() / eff_mass_t
    j_t = clamp(j_t, -mu * j_n, mu * j_n)
    
    impulse_t = t * j_t
    
    if a and not a.is_static:
        a.vel = a.vel + impulse_t * inv_m_a
        a.ang_vel = a.ang_vel + I_inv_a * r_a.cross(impulse_t)
    
    if b and not b.is_static:
        b.vel = b.vel - impulse_t * inv_m_b
        b.ang_vel = b.ang_vel - I_inv_b * r_b.cross(impulse_t)


def resolve_ground(contact, baumgarte=0.2, slop=0.01):
    body = contact.body_a
    n = contact.normal
    
    if body is None or body.is_static:
        return
    
    r = contact.pt - body.pos
    I_inv = body.get_world_inertia_inv()
    
    v_pt = body.vel + body.ang_vel.cross(r)
    v_n = v_pt.dot(n)
    
    if v_n > 0:
        return
    
    e = body.restitution
    
    rn = r.cross(n)
    ang_term = (I_inv * rn).cross(r).dot(n)
    eff_mass = body.inv_mass + ang_term
    
    if eff_mass < 1e-10:
        return
    
    bias = 0.0
    pen = contact.depth - slop
    if pen > 0:
        bias = baumgarte * pen / (1.0/60.0)
    
    j_n = (-(1 + e) * v_n + bias) / eff_mass
    j_n = max(j_n, 0.0)
    
    impulse_n = n * j_n
    
    body.vel = body.vel + impulse_n * body.inv_mass
    body.ang_vel = body.ang_vel + I_inv * r.cross(impulse_n)
    
    v_pt = body.vel + body.ang_vel.cross(r)
    v_t = v_pt - n * v_pt.dot(n)
    
    if v_t.mag_sq() < 1e-10:
        return
    
    t = v_t.normalized()
    rt = r.cross(t)
    ang_term_t = (I_inv * rt).cross(r).dot(t)
    eff_mass_t = body.inv_mass + ang_term_t
    
    if eff_mass_t < 1e-10:
        return
    
    mu = body.friction
    j_t = -v_t.mag() / eff_mass_t
    j_t = clamp(j_t, -mu * j_n, mu * j_n)
    
    impulse_t = t * j_t
    
    body.vel = body.vel + impulse_t * body.inv_mass
    body.ang_vel = body.ang_vel + I_inv * r.cross(impulse_t)


def position_correction(contact, correction_pct=0.8, slop=0.01):
    a = contact.body_a
    b = contact.body_b
    n = contact.normal
    
    pen = contact.depth - slop
    if pen <= 0:
        return
    
    inv_m_a = a.inv_mass if a and not a.is_static else 0.0
    inv_m_b = b.inv_mass if b and not b.is_static else 0.0
    
    total_inv = inv_m_a + inv_m_b
    if total_inv < 1e-10:
        return
    
    correction = n * (pen * correction_pct / total_inv)
    
    if a and not a.is_static:
        a.pos = a.pos + correction * inv_m_a
    
    if b and not b.is_static:
        b.pos = b.pos - correction * inv_m_b


def position_correction_ground(contact, correction_pct=0.8, slop=0.01):
    body = contact.body_a
    n = contact.normal
    
    pen = contact.depth - slop
    if pen <= 0 or body is None or body.is_static:
        return
    
    body.pos = body.pos + n * (pen * correction_pct)


class ContactSolver:
    def __init__(self, iterations=10):
        self.iterations = iterations
        self.contacts = []
        self.ground_contacts = []
    
    def add_contact(self, contact):
        if contact.body_b is None:
            self.ground_contacts.append(contact)
        else:
            self.contacts.append(contact)
    
    def clear(self):
        self.contacts.clear()
        self.ground_contacts.clear()
    
    def solve(self):
        for _ in range(self.iterations):
            for c in self.contacts:
                resolve_collision(c)
            for c in self.ground_contacts:
                resolve_ground(c)
        
        for c in self.contacts:
            position_correction(c)
        for c in self.ground_contacts:
            position_correction_ground(c)


if __name__ == "__main__":
    from math_util import Quat
    from body import RigidBody, World, integrate_rk4
    from collision import SphereCollider, BoxCollider, GroundPlane, detect_collision, detect_ground_collision
    
    print("=== Bouncing Ball Test ===")
    
    world = World()
    ball = RigidBody(mass=1.0, pos=Vec3(0, 5, 0))
    ball.set_sphere_inertia(0.5)
    ball.restitution = 0.7
    world.add_body(ball)
    
    ball_col = SphereCollider(ball, 0.5)
    ground = GroundPlane(y=0)
    solver = ContactSolver(iterations=8)
    
    dt = 1.0 / 60.0
    
    for frame in range(300):
        integrate_rk4(ball, dt, world.gravity)
        
        contact = detect_ground_collision(ball_col, ground)
        solver.clear()
        if contact:
            solver.add_contact(contact)
        solver.solve()
        
        if frame % 60 == 0:
            print(f"t={frame*dt:.2f}s  y={ball.pos.y:.3f}  vel_y={ball.vel.y:.3f}")
    
    print("\n=== Spinning Box on Ground ===")
    
    box = RigidBody(mass=2.0, pos=Vec3(0, 0.5, 0))
    box.set_box_inertia(1, 1, 1)
    box.ang_vel = Vec3(0, 10, 0)
    box.friction = 0.5
    
    box_col = BoxCollider(box, Vec3(0.5, 0.5, 0.5))
    solver2 = ContactSolver(iterations=8)
    
    print(f"Initial ang_vel: {box.ang_vel}")
    
    for frame in range(180):
        integrate_rk4(box, dt, Vec3.zero())
        
        contact = detect_ground_collision(box_col, ground)
        solver2.clear()
        if contact:
            solver2.add_contact(contact)
        solver2.solve()
    
    print(f"Final ang_vel after 3s with friction: {box.ang_vel}")
    
    print("\n=== Two Spheres Collision ===")
    
    s1 = RigidBody(mass=1.0, pos=Vec3(-2, 1, 0))
    s1.vel = Vec3(2, 0, 0)
    s1.set_sphere_inertia(0.5)
    
    s2 = RigidBody(mass=1.0, pos=Vec3(2, 1, 0))
    s2.vel = Vec3(-2, 0, 0)
    s2.set_sphere_inertia(0.5)
    
    col1 = SphereCollider(s1, 0.5)
    col2 = SphereCollider(s2, 0.5)
    
    solver3 = ContactSolver(iterations=8)
    
    print(f"Initial: s1.vel={s1.vel}  s2.vel={s2.vel}")
    
    for frame in range(120):
        integrate_rk4(s1, dt, Vec3.zero())
        integrate_rk4(s2, dt, Vec3.zero())
        
        contact = detect_collision(col1, col2)
        solver3.clear()
        if contact:
            solver3.add_contact(contact)
        solver3.solve()
    
    print(f"After collision: s1.vel={s1.vel}  s2.vel={s2.vel}")
