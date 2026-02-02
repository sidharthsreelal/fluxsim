from __future__ import annotations
from typing import List, Optional, Tuple, Union
from math_util import Vec3
from body import RigidBody

class Collider:
    def __init__(self, body: RigidBody) -> None:
        self.body = body
    
    def support(self, d: Vec3) -> Vec3:
        raise NotImplementedError
    
    def bounding_radius(self) -> float:
        raise NotImplementedError


class SphereCollider(Collider):
    def __init__(self, body: RigidBody, radius: float) -> None:
        super().__init__(body)
        self.radius = float(radius)
    
    def support(self, d: Vec3) -> Vec3:
        d_norm = d.normalized()
        return self.body.pos + d_norm * self.radius
    
    def bounding_radius(self) -> float:
        return self.radius


class BoxCollider(Collider):
    def __init__(self, body: RigidBody, half_extents: Vec3) -> None:
        super().__init__(body)
        self.half = half_extents
    
    def support(self, d: Vec3) -> Vec3:
        local_d = self.body.rot.conjugate().rotate_vec(d)
        
        local_pt = Vec3(
            self.half.x if local_d.x >= 0 else -self.half.x,
            self.half.y if local_d.y >= 0 else -self.half.y,
            self.half.z if local_d.z >= 0 else -self.half.z
        )
        
        return self.body.local_to_world(local_pt)
    
    def bounding_radius(self) -> float:
        h = self.half
        return (h.x * h.x + h.y * h.y + h.z * h.z) ** 0.5


class ConvexHullCollider(Collider):
    def __init__(self, body: RigidBody, local_verts: List[Vec3]) -> None:
        super().__init__(body)
        self.local_verts = local_verts
    
    def support(self, d: Vec3) -> Vec3:
        local_d = self.body.rot.conjugate().rotate_vec(d)
        
        best = self.local_verts[0]
        best_dot = local_d.dot(best)
        
        for v in self.local_verts[1:]:
            proj = local_d.dot(v)
            if proj > best_dot:
                best_dot = proj
                best = v
        
        return self.body.local_to_world(best)
    
    def bounding_radius(self) -> float:
        max_dist_sq = 0.0
        for v in self.local_verts:
            d = v.mag_sq()
            if d > max_dist_sq:
                max_dist_sq = d
        return max_dist_sq ** 0.5


def minkowski_support(c1: Collider, c2: Collider, d: Vec3) -> Vec3:
    return c1.support(d) - c2.support(-d)


class Simplex:
    def __init__(self) -> None:
        self.pts: List[Vec3] = []
    
    def add(self, pt: Vec3) -> None:
        self.pts.insert(0, pt)
    
    def __len__(self) -> int:
        return len(self.pts)
    
    def __getitem__(self, i: int) -> Vec3:
        return self.pts[i]
    
    def set(self, pts: List[Vec3]) -> None:
        self.pts = list(pts)


def triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    return b * a.dot(c) - c * a.dot(b)


def do_simplex(simplex: Simplex, d: Vec3) -> bool:
    if len(simplex) == 2:
        return line_case(simplex, d)
    elif len(simplex) == 3:
        return triangle_case(simplex, d)
    elif len(simplex) == 4:
        return tetrahedron_case(simplex, d)
    return False


def line_case(s: Simplex, d: Vec3) -> bool:
    a, b = s[0], s[1]
    ab = b - a
    ao = -a
    
    if ab.dot(ao) > 0:
        new_d = triple_product(ab, ao, ab)
        d.x, d.y, d.z = new_d.x, new_d.y, new_d.z
    else:
        s.set([a])
        d.x, d.y, d.z = ao.x, ao.y, ao.z
    
    return False


def triangle_case(s: Simplex, d: Vec3) -> bool:
    a, b, c = s[0], s[1], s[2]
    ab = b - a
    ac = c - a
    ao = -a
    
    abc = ab.cross(ac)
    
    if abc.cross(ac).dot(ao) > 0:
        if ac.dot(ao) > 0:
            s.set([a, c])
            new_d = triple_product(ac, ao, ac)
            d.x, d.y, d.z = new_d.x, new_d.y, new_d.z
        else:
            s.set([a, b])
            return line_case(s, d)
    else:
        if ab.cross(abc).dot(ao) > 0:
            s.set([a, b])
            return line_case(s, d)
        else:
            if abc.dot(ao) > 0:
                d.x, d.y, d.z = abc.x, abc.y, abc.z
            else:
                s.set([a, c, b])
                d.x, d.y, d.z = -abc.x, -abc.y, -abc.z
    
    return False


def tetrahedron_case(s: Simplex, d: Vec3) -> bool:
    a, b, c, dd = s[0], s[1], s[2], s[3]
    
    ab = b - a
    ac = c - a
    ad = dd - a
    ao = -a
    
    abc = ab.cross(ac)
    acd = ac.cross(ad)
    adb = ad.cross(ab)
    
    if abc.dot(ao) > 0:
        s.set([a, b, c])
        d.x, d.y, d.z = abc.x, abc.y, abc.z
        return triangle_case(s, d)
    
    if acd.dot(ao) > 0:
        s.set([a, c, dd])
        d.x, d.y, d.z = acd.x, acd.y, acd.z
        return triangle_case(s, d)
    
    if adb.dot(ao) > 0:
        s.set([a, dd, b])
        d.x, d.y, d.z = adb.x, adb.y, adb.z
        return triangle_case(s, d)
    
    return True


def gjk(c1: Collider, c2: Collider, max_iter: int = 64) -> Tuple[bool, Optional[Simplex]]:
    d = c2.body.pos - c1.body.pos
    if d.mag_sq() < 1e-10:
        d = Vec3(1, 0, 0)
    
    simplex = Simplex()
    support = minkowski_support(c1, c2, d)
    simplex.add(support)
    d = -support
    
    for _ in range(max_iter):
        if d.mag_sq() < 1e-10:
            return True, simplex
        
        a = minkowski_support(c1, c2, d)
        
        if a.dot(d) < 0:
            return False, None
        
        simplex.add(a)
        
        if do_simplex(simplex, d):
            return True, simplex
    
    return False, None


class EPAFace:
    def __init__(self, a: Vec3, b: Vec3, c: Vec3) -> None:
        self.verts = [a, b, c]
        ab = b - a
        ac = c - a
        self.normal = ab.cross(ac).normalized()
        self.dist = self.normal.dot(a)
        
        if self.dist < 0:
            self.normal = -self.normal
            self.dist = -self.dist
            self.verts = [a, c, b]


def vec_eq(a: Vec3, b: Vec3, eps: float = 1e-8) -> bool:
    return (a - b).mag_sq() < eps * eps


def epa(c1: Collider, c2: Collider, simplex: Simplex, max_iter: int = 64, tol: float = 1e-4) -> Tuple[Vec3, float]:
    if len(simplex) < 4:
        dirs = [Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)]
        for d in dirs:
            if len(simplex) >= 4:
                break
            pt = minkowski_support(c1, c2, d)
            is_dup = False
            for existing in simplex.pts:
                if vec_eq(pt, existing):
                    is_dup = True
                    break
            if not is_dup:
                simplex.add(pt)
    
    if len(simplex) < 4:
        n = Vec3(0, 1, 0)
        return n, 0.01
    
    pts = list(simplex.pts)
    
    faces = [
        EPAFace(pts[0], pts[1], pts[2]),
        EPAFace(pts[0], pts[2], pts[3]),
        EPAFace(pts[0], pts[3], pts[1]),
        EPAFace(pts[1], pts[3], pts[2])
    ]
    
    for _ in range(max_iter):
        min_face = min(faces, key=lambda f: f.dist)
        
        support = minkowski_support(c1, c2, min_face.normal)
        support_dist = min_face.normal.dot(support)
        
        if support_dist - min_face.dist < tol:
            return min_face.normal, min_face.dist
        
        edges = {}
        i = 0
        while i < len(faces):
            f = faces[i]
            if f.normal.dot(support - f.verts[0]) > 0:
                for j in range(3):
                    v0 = f.verts[j]
                    v1 = f.verts[(j + 1) % 3]
                    key = (id(v1), id(v0))
                    if key in edges:
                        del edges[key]
                    else:
                        edges[(id(v0), id(v1))] = (v0, v1)
                faces.pop(i)
            else:
                i += 1
        
        for e in edges.values():
            faces.append(EPAFace(support, e[0], e[1]))
        
        if len(faces) == 0:
            break
    
    min_face = min(faces, key=lambda f: f.dist)
    return min_face.normal, min_face.dist


class Contact:
    def __init__(self, pt: Vec3, normal: Vec3, depth: float, body_a: Optional[RigidBody], body_b: Optional[RigidBody]) -> None:
        self.pt = pt
        self.normal = normal
        self.depth = depth
        self.body_a = body_a
        self.body_b = body_b


def find_contact_point(c1: Collider, c2: Collider, normal: Vec3, depth: float) -> Vec3:
    pt_a = c1.support(normal)
    pt_b = c2.support(-normal)
    return (pt_a + pt_b) * 0.5


def detect_collision(c1: Collider, c2: Collider) -> Optional[Contact]:
    dist_sq = (c1.body.pos - c2.body.pos).mag_sq()
    combined_radius = c1.bounding_radius() + c2.bounding_radius()
    if dist_sq > combined_radius * combined_radius:
        return None
    
    hit, simplex = gjk(c1, c2)
    
    if not hit or simplex is None:
        return None
    
    normal, depth = epa(c1, c2, simplex)
    
    contact_pt = find_contact_point(c1, c2, normal, depth)
    
    return Contact(contact_pt, normal, depth, c1.body, c2.body)


class GroundPlane:
    def __init__(self, y: float = 0, normal: Optional[Vec3] = None) -> None:
        self.y = float(y)
        self.normal = normal if normal else Vec3(0, 1, 0)


def detect_ground_collision(collider: Collider, ground: GroundPlane) -> Optional[Contact]:
    support_down = collider.support(-ground.normal)
    
    penetration = ground.y - support_down.dot(ground.normal)
    
    if penetration > 0:
        contact_pt = support_down + ground.normal * (penetration * 0.5)
        return Contact(contact_pt, ground.normal, penetration, collider.body, None)
    
    return None
