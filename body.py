from __future__ import annotations
from typing import Optional, List
from math_util import Vec3, Mat3, Quat

class RigidBody:
    def __init__(self, mass: float = 1.0, pos: Optional[Vec3] = None, rot: Optional[Quat] = None) -> None:
        self.pos = pos if pos else Vec3.zero()
        self.vel = Vec3.zero()
        self.rot = rot if rot else Quat.identity()
        self.ang_vel = Vec3.zero()
        
        self.mass = float(mass)
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        
        self.I_body = Mat3.identity()
        self.I_body_inv = Mat3.identity()
        
        self.force_accum = Vec3.zero()
        self.torque_accum = Vec3.zero()
        
        self.is_static = mass <= 0
        self.restitution = 0.5
        self.friction = 0.4
        
        self._I_world_inv_cache = None
        self._I_world_cache = None
        self._rot_cache_w = None
        self._rot_cache_x = None
        self._rot_cache_y = None
        self._rot_cache_z = None
    
    def set_box_inertia(self, w: float, h: float, d: float) -> None:
        m = self.mass
        Ix = (1.0/12.0) * m * (h*h + d*d)
        Iy = (1.0/12.0) * m * (w*w + d*d)
        Iz = (1.0/12.0) * m * (w*w + h*h)
        self.I_body = Mat3.from_diagonal(Vec3(Ix, Iy, Iz))
        self.I_body_inv = Mat3.from_diagonal(Vec3(1.0/Ix, 1.0/Iy, 1.0/Iz))
    
    def set_sphere_inertia(self, radius: float) -> None:
        I = (2.0/5.0) * self.mass * radius * radius
        self.I_body = Mat3.from_diagonal(Vec3(I, I, I))
        self.I_body_inv = Mat3.from_diagonal(Vec3(1.0/I, 1.0/I, 1.0/I))
    
    def get_world_inertia_inv(self) -> Mat3:
        r = self.rot
        if (self._I_world_inv_cache is None or
            self._rot_cache_w != r.w or self._rot_cache_x != r.x or
            self._rot_cache_y != r.y or self._rot_cache_z != r.z):
            R = r.to_mat3()
            self._I_world_inv_cache = R * self.I_body_inv * R.transpose()
            self._I_world_cache = R * self.I_body * R.transpose()
            self._rot_cache_w = r.w
            self._rot_cache_x = r.x
            self._rot_cache_y = r.y
            self._rot_cache_z = r.z
        return self._I_world_inv_cache
    
    def get_world_inertia(self) -> Mat3:
        self.get_world_inertia_inv()
        return self._I_world_cache
    
    def apply_force(self, f: Vec3, world_pt: Optional[Vec3] = None) -> None:
        if self.is_static:
            return
        self.force_accum = self.force_accum + f
        if world_pt is not None:
            r = world_pt - self.pos
            self.torque_accum = self.torque_accum + r.cross(f)
    
    def apply_torque(self, t: Vec3) -> None:
        if self.is_static:
            return
        self.torque_accum = self.torque_accum + t
    
    def apply_impulse(self, j: Vec3, world_pt: Optional[Vec3] = None) -> None:
        if self.is_static:
            return
        self.vel = self.vel + j * self.inv_mass
        if world_pt is not None:
            r = world_pt - self.pos
            I_inv = self.get_world_inertia_inv()
            d_omega = I_inv * r.cross(j)
            self.ang_vel = self.ang_vel + d_omega
    
    def clear_accumulators(self) -> None:
        self.force_accum = Vec3.zero()
        self.torque_accum = Vec3.zero()
    
    def local_to_world(self, local_pt: Vec3) -> Vec3:
        return self.pos + self.rot.rotate_vec(local_pt)
    
    def world_to_local(self, world_pt: Vec3) -> Vec3:
        return self.rot.conjugate().rotate_vec(world_pt - self.pos)
    
    def point_velocity(self, world_pt: Vec3) -> Vec3:
        r = world_pt - self.pos
        return self.vel + self.ang_vel.cross(r)


class State:
    def __init__(self, pos: Vec3, vel: Vec3, rot: Quat, ang_vel: Vec3) -> None:
        self.pos = pos
        self.vel = vel
        self.rot = rot
        self.ang_vel = ang_vel


class Deriv:
    def __init__(self, d_pos: Vec3, d_vel: Vec3, d_rot: Quat, d_ang_vel: Vec3) -> None:
        self.d_pos = d_pos
        self.d_vel = d_vel
        self.d_rot = d_rot
        self.d_ang_vel = d_ang_vel


def compute_deriv(body: RigidBody, state: State, gravity: Vec3) -> Deriv:
    d_pos = state.vel
    
    acc = gravity + body.force_accum * body.inv_mass
    d_vel = acc
    
    d_rot = state.rot.derivative(state.ang_vel)
    
    I_inv = body.get_world_inertia_inv()
    tau = body.torque_accum
    gyro = state.ang_vel.cross(body.get_world_inertia() * state.ang_vel)
    d_ang_vel = I_inv * (tau - gyro)
    
    return Deriv(d_pos, d_vel, d_rot, d_ang_vel)


def state_add(s: State, d: Deriv, dt: float) -> State:
    new_pos = s.pos + d.d_pos * dt
    new_vel = s.vel + d.d_vel * dt
    new_rot = (s.rot + d.d_rot * dt).normalized()
    new_ang_vel = s.ang_vel + d.d_ang_vel * dt
    return State(new_pos, new_vel, new_rot, new_ang_vel)


def integrate_rk4(body: RigidBody, dt: float, gravity: Optional[Vec3] = None) -> None:
    if body.is_static:
        return
    
    if gravity is None:
        gravity = Vec3(0, -9.81, 0)
    
    s0 = State(body.pos.copy(), body.vel.copy(), body.rot.copy(), body.ang_vel.copy())
    
    k1 = compute_deriv(body, s0, gravity)
    
    s1 = state_add(s0, k1, dt * 0.5)
    k2 = compute_deriv(body, s1, gravity)
    
    s2 = state_add(s0, k2, dt * 0.5)
    k3 = compute_deriv(body, s2, gravity)
    
    s3 = state_add(s0, k3, dt)
    k4 = compute_deriv(body, s3, gravity)
    
    def combine_vec(v: Vec3, d1: Vec3, d2: Vec3, d3: Vec3, d4: Vec3) -> Vec3:
        return v + (d1 + d2 * 2 + d3 * 2 + d4) * (dt / 6.0)
    
    body.pos = combine_vec(s0.pos, k1.d_pos, k2.d_pos, k3.d_pos, k4.d_pos)
    body.vel = combine_vec(s0.vel, k1.d_vel, k2.d_vel, k3.d_vel, k4.d_vel)
    
    d_rot_avg = (k1.d_rot + k2.d_rot * 2 + k3.d_rot * 2 + k4.d_rot) * (dt / 6.0)
    body.rot = (s0.rot + d_rot_avg).normalized()
    
    body.ang_vel = combine_vec(s0.ang_vel, k1.d_ang_vel, k2.d_ang_vel, k3.d_ang_vel, k4.d_ang_vel)
    
    body.clear_accumulators()


def integrate_semi_implicit(body: RigidBody, dt: float, gravity: Optional[Vec3] = None) -> None:
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
    def __init__(self) -> None:
        self.bodies: List[RigidBody] = []
        self.gravity = Vec3(0, -9.81, 0)
        self.dt = 1.0 / 60.0
        self.use_rk4 = True
    
    def add_body(self, body: RigidBody) -> RigidBody:
        self.bodies.append(body)
        return body
    
    def remove_body(self, body: RigidBody) -> None:
        if body in self.bodies:
            self.bodies.remove(body)
    
    def step(self, dt: Optional[float] = None) -> None:
        if dt is None:
            dt = self.dt
        
        for b in self.bodies:
            if self.use_rk4:
                integrate_rk4(b, dt, self.gravity)
            else:
                integrate_semi_implicit(b, dt, self.gravity)
    
    def get_kinetic_energy(self) -> float:
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
    
    def get_potential_energy(self, ref_y: float = 0.0) -> float:
        pe = 0.0
        for b in self.bodies:
            if b.is_static:
                continue
            pe += b.mass * abs(self.gravity.y) * (b.pos.y - ref_y)
        return pe
