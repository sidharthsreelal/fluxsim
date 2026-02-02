from __future__ import annotations
from typing import Optional, List
from math_util import Vec3, clamp, Mat3
from collision import Contact
from body import RigidBody

def resolve_collision(contact: Contact, baumgarte: float = 0.2, slop: float = 0.01) -> None:
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


def apply_friction(contact: Contact, j_n: float, inv_m_a: float, inv_m_b: float, I_inv_a: Optional[Mat3], I_inv_b: Optional[Mat3], r_a: Vec3, r_b: Vec3) -> None:
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


def resolve_ground(contact: Contact, baumgarte: float = 0.2, slop: float = 0.01) -> None:
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


def position_correction(contact: Contact, correction_pct: float = 0.8, slop: float = 0.01) -> None:
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


def position_correction_ground(contact: Contact, correction_pct: float = 0.8, slop: float = 0.01) -> None:
    body = contact.body_a
    n = contact.normal
    
    pen = contact.depth - slop
    if pen <= 0 or body is None or body.is_static:
        return
    
    body.pos = body.pos + n * (pen * correction_pct)


class ContactSolver:
    def __init__(self, iterations: int = 10) -> None:
        self.iterations = iterations
        self.contacts: List[Contact] = []
        self.ground_contacts: List[Contact] = []
    
    def add_contact(self, contact: Contact) -> None:
        if contact.body_b is None:
            self.ground_contacts.append(contact)
        else:
            self.contacts.append(contact)
    
    def clear(self) -> None:
        self.contacts.clear()
        self.ground_contacts.clear()
    
    def solve(self) -> None:
        for _ in range(self.iterations):
            for c in self.contacts:
                resolve_collision(c)
            for c in self.ground_contacts:
                resolve_ground(c)
        
        for c in self.contacts:
            position_correction(c)
        for c in self.ground_contacts:
            position_correction_ground(c)
