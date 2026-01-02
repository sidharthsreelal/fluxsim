from math_util import Vec3, Mat3, Quat

class RigidBody:
    def __init__(self, mass=1.0, pos=None, rot=None):
        self.pos = pos if pos else Vec3.zero()
        self.vel = Vec3.zero()
        self.rot = rot if rot else Quat.identity()
        self.ang_vel = Vec3.zero()
        
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        
        self.I_body = Mat3.identity()
        self.I_body_inv = Mat3.identity()
        
        self.force_accum = Vec3.zero()
        self.torque_accum = Vec3.zero()
        
        self.is_static = mass <= 0
        self.restitution = 0.5
        self.friction = 0.4
    
    def set_box_inertia(self, w, h, d):
        m = self.mass
        Ix = (1.0/12.0) * m * (h*h + d*d)
        Iy = (1.0/12.0) * m * (w*w + d*d)
        Iz = (1.0/12.0) * m * (w*w + h*h)
        self.I_body = Mat3.from_diagonal(Vec3(Ix, Iy, Iz))
        self.I_body_inv = Mat3.from_diagonal(Vec3(1.0/Ix, 1.0/Iy, 1.0/Iz))
    
    def set_sphere_inertia(self, radius):
        I = (2.0/5.0) * self.mass * radius * radius
        self.I_body = Mat3.from_diagonal(Vec3(I, I, I))
        self.I_body_inv = Mat3.from_diagonal(Vec3(1.0/I, 1.0/I, 1.0/I))
    
    def get_world_inertia_inv(self):
        R = self.rot.to_mat3()
        return R * self.I_body_inv * R.transpose()
    
    def get_world_inertia(self):
        R = self.rot.to_mat3()
        return R * self.I_body * R.transpose()
    
    def apply_force(self, f, world_pt=None):
        if self.is_static:
            return
        self.force_accum = self.force_accum + f
        if world_pt is not None:
            r = world_pt - self.pos
            self.torque_accum = self.torque_accum + r.cross(f)
    
    def apply_torque(self, t):
        if self.is_static:
            return
        self.torque_accum = self.torque_accum + t
    
    def apply_impulse(self, j, world_pt=None):
        if self.is_static:
            return
        self.vel = self.vel + j * self.inv_mass
        if world_pt is not None:
            r = world_pt - self.pos
            I_inv = self.get_world_inertia_inv()
            d_omega = I_inv * r.cross(j)
            self.ang_vel = self.ang_vel + d_omega
    
    def clear_accumulators(self):
        self.force_accum = Vec3.zero()
        self.torque_accum = Vec3.zero()
    
    def local_to_world(self, local_pt):
        return self.pos + self.rot.rotate_vec(local_pt)
    
    def world_to_local(self, world_pt):
        return self.rot.conjugate().rotate_vec(world_pt - self.pos)
    
    def point_velocity(self, world_pt):
        r = world_pt - self.pos
        return self.vel + self.ang_vel.cross(r)


class State:
    def __init__(self, pos, vel, rot, ang_vel):
        self.pos = pos
        self.vel = vel
        self.rot = rot
        self.ang_vel = ang_vel


class Deriv:
    def __init__(self, d_pos, d_vel, d_rot, d_ang_vel):
        self.d_pos = d_pos
        self.d_vel = d_vel
        self.d_rot = d_rot
        self.d_ang_vel = d_ang_vel


def compute_deriv(body, state, dt_frac, gravity):
    d_pos = state.vel
    
    acc = gravity + body.force_accum * body.inv_mass
    d_vel = acc
    
    d_rot = state.rot.derivative(state.ang_vel)
    
    I_inv = body.get_world_inertia_inv()
    tau = body.torque_accum
    gyro = state.ang_vel.cross(body.get_world_inertia() * state.ang_vel)
    d_ang_vel = I_inv * (tau - gyro)
    
    return Deriv(d_pos, d_vel, d_rot, d_ang_vel)


def state_add(s, d, dt):
    new_pos = s.pos + d.d_pos * dt
    new_vel = s.vel + d.d_vel * dt
    new_rot = (s.rot + d.d_rot * dt).normalized()
    new_ang_vel = s.ang_vel + d.d_ang_vel * dt
    return State(new_pos, new_vel, new_rot, new_ang_vel)


def integrate_rk4(body, dt, gravity=None):
    if body.is_static:
        return
    
    if gravity is None:
        gravity = Vec3(0, -9.81, 0)
    
    s0 = State(body.pos.copy(), body.vel.copy(), body.rot.copy(), body.ang_vel.copy())
    
    k1 = compute_deriv(body, s0, 0, gravity)
    
    s1 = state_add(s0, k1, dt * 0.5)
    k2 = compute_deriv(body, s1, dt * 0.5, gravity)
    
    s2 = state_add(s0, k2, dt * 0.5)
    k3 = compute_deriv(body, s2, dt * 0.5, gravity)
    
    s3 = state_add(s0, k3, dt)
    k4 = compute_deriv(body, s3, dt, gravity)
    
    def combine_vec(v, d1, d2, d3, d4):
        return v + (d1 + d2 * 2 + d3 * 2 + d4) * (dt / 6.0)
    
    body.pos = combine_vec(s0.pos, k1.d_pos, k2.d_pos, k3.d_pos, k4.d_pos)
    body.vel = combine_vec(s0.vel, k1.d_vel, k2.d_vel, k3.d_vel, k4.d_vel)
    
    d_rot_avg = (k1.d_rot + k2.d_rot * 2 + k3.d_rot * 2 + k4.d_rot) * (dt / 6.0)
    body.rot = (s0.rot + d_rot_avg).normalized()
    
    body.ang_vel = combine_vec(s0.ang_vel, k1.d_ang_vel, k2.d_ang_vel, k3.d_ang_vel, k4.d_ang_vel)
    
    body.clear_accumulators()


def integrate_semi_implicit(body, dt, gravity=None):
    if body.is_static:
        return
    
    if gravity is None:
        gravity = Vec3(0, -9.81, 0)
    
    acc = gravity + body.force_accum * body.inv_mass
    body.vel = body.vel + acc * dt
    body.pos = body.pos + body.vel * dt
    
    I_inv = body.get_world_inertia_inv()
    tau = body.torque_accum
    I_w = body.get_world_inertia()
    gyro = body.ang_vel.cross(I_w * body.ang_vel)
    ang_acc = I_inv * (tau - gyro)
    
    body.ang_vel = body.ang_vel + ang_acc * dt
    
    dq = body.rot.derivative(body.ang_vel)
    body.rot = (body.rot + dq * dt).normalized()
    
    body.clear_accumulators()


class World:
    def __init__(self):
        self.bodies = []
        self.gravity = Vec3(0, -9.81, 0)
        self.dt = 1.0 / 60.0
        self.use_rk4 = True
    
    def add_body(self, body):
        self.bodies.append(body)
        return body
    
    def remove_body(self, body):
        if body in self.bodies:
            self.bodies.remove(body)
    
    def step(self, dt=None):
        if dt is None:
            dt = self.dt
        
        for b in self.bodies:
            if self.use_rk4:
                integrate_rk4(b, dt, self.gravity)
            else:
                integrate_semi_implicit(b, dt, self.gravity)
    
    def get_kinetic_energy(self):
        ke = 0.0
        for b in self.bodies:
            if b.is_static:
                continue
            ke += 0.5 * b.mass * b.vel.mag_sq()
            I_w = b.get_world_inertia()
            omega = b.ang_vel
            rot_ke = 0.5 * omega.dot(I_w * omega)
            ke += rot_ke
        return ke
    
    def get_potential_energy(self, ref_y=0.0):
        pe = 0.0
        for b in self.bodies:
            if b.is_static:
                continue
            pe += b.mass * abs(self.gravity.y) * (b.pos.y - ref_y)
        return pe


if __name__ == "__main__":
    print("=== RigidBody Tests ===")
    
    world = World()
    world.gravity = Vec3(0, -9.81, 0)
    
    box = RigidBody(mass=2.0, pos=Vec3(0, 10, 0))
    box.set_box_inertia(1, 1, 1)
    world.add_body(box)
    
    print(f"Initial pos: {box.pos}")
    print(f"Initial vel: {box.vel}")
    
    total_time = 0.0
    dt = 1.0 / 60.0
    
    for i in range(60):
        world.step(dt)
        total_time += dt
    
    print(f"\nAfter 1 second of free fall (RK4):")
    print(f"Position: {box.pos}")
    print(f"Velocity: {box.vel}")
    print(f"Expected y-vel: {-9.81 * total_time:.4f}")
    print(f"Expected y-pos: {10 - 0.5 * 9.81 * total_time * total_time:.4f}")
    
    print("\n=== Rotation Test ===")
    spinner = RigidBody(mass=1.0, pos=Vec3(0, 0, 0))
    spinner.set_box_inertia(1, 0.2, 1)
    spinner.ang_vel = Vec3(0, 5, 0)
    
    print(f"Initial ang_vel: {spinner.ang_vel}")
    print(f"Initial rot: {spinner.rot}")
    
    for i in range(120):
        integrate_rk4(spinner, dt, Vec3.zero())
    
    print(f"\nAfter 2 seconds spinning:")
    print(f"Rotation: {spinner.rot}")
    print(f"Ang velocity (should be constant, no torque): {spinner.ang_vel}")
    
    print("\n=== Energy Conservation Test ===")
    pendulum = RigidBody(mass=1.0, pos=Vec3(5, 5, 0))
    pendulum.set_sphere_inertia(0.5)
    world2 = World()
    world2.add_body(pendulum)
    
    e0 = world2.get_kinetic_energy() + world2.get_potential_energy()
    print(f"Initial total energy: {e0:.4f}")
    
    for _ in range(300):
        world2.step(dt)
    
    e1 = world2.get_kinetic_energy() + world2.get_potential_energy()
    print(f"Final total energy: {e1:.4f}")
    print(f"Energy drift: {abs(e1 - e0):.6f}")
