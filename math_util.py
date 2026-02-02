from __future__ import annotations
import numpy as np
from math import sqrt, sin, cos, acos
from typing import Union, Iterator, Optional, List

class Vec3:
    __slots__ = ('x', 'y', 'z')
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, o: Vec3) -> Vec3:
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    
    def __sub__(self, o: Vec3) -> Vec3:
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    
    def __mul__(self, s: float) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)
    
    def __rmul__(self, s: float) -> Vec3:
        return self.__mul__(s)
    
    def __truediv__(self, s: float) -> Vec3:
        inv = 1.0 / s
        return Vec3(self.x * inv, self.y * inv, self.z * inv)
    
    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)
    
    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"
    
    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z
    
    def dot(self, o: Vec3) -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z
    
    def cross(self, o: Vec3) -> Vec3:
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x
        )
    
    def mag_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def mag(self) -> float:
        return sqrt(self.mag_sq())
    
    def normalized(self) -> Vec3:
        m = self.mag()
        if m < 1e-10:
            return Vec3(0, 0, 0)
        return self / m
    
    def copy(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> Vec3:
        return Vec3(arr[0], arr[1], arr[2])
    
    @staticmethod
    def zero() -> Vec3:
        return Vec3(0, 0, 0)
    
    @staticmethod
    def up() -> Vec3:
        return Vec3(0, 1, 0)
    
    @staticmethod
    def right() -> Vec3:
        return Vec3(1, 0, 0)
    
    @staticmethod
    def forward() -> Vec3:
        return Vec3(0, 0, 1)


class Mat3:
    __slots__ = ('m',)
    
    def __init__(self, data: Optional[Union[np.ndarray, List[List[float]]]] = None) -> None:
        if data is None:
            self.m = np.eye(3, dtype=np.float64)
        else:
            self.m = np.array(data, dtype=np.float64).reshape(3, 3)
    
    def __mul__(self, other: Union[Mat3, Vec3, float]) -> Union[Mat3, Vec3]:
        if isinstance(other, Mat3):
            return Mat3(self.m @ other.m)
        elif isinstance(other, Vec3):
            r = self.m @ np.array([other.x, other.y, other.z])
            return Vec3(r[0], r[1], r[2])
        else:
            return Mat3(self.m * other)
    
    def __rmul__(self, s: float) -> Mat3:
        return Mat3(self.m * s)
    
    def __add__(self, o: Mat3) -> Mat3:
        return Mat3(self.m + o.m)
    
    def __repr__(self) -> str:
        return f"Mat3(\n{self.m}\n)"
    
    def transpose(self) -> Mat3:
        return Mat3(self.m.T)
    
    def T(self) -> Mat3:
        return self.transpose()
    
    def inverse(self) -> Mat3:
        try:
            return Mat3(np.linalg.inv(self.m))
        except np.linalg.LinAlgError:
            return Mat3()
    
    def det(self) -> float:
        return float(np.linalg.det(self.m))
    
    def copy(self) -> Mat3:
        return Mat3(self.m.copy())
    
    @staticmethod
    def identity() -> Mat3:
        return Mat3()
    
    @staticmethod
    def zero() -> Mat3:
        return Mat3(np.zeros((3, 3)))
    
    @staticmethod
    def from_diagonal(d: Vec3) -> Mat3:
        m = np.diag([d.x, d.y, d.z])
        return Mat3(m)
    
    @staticmethod
    def rotation_x(angle: float) -> Mat3:
        c, s = cos(angle), sin(angle)
        return Mat3([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ])
    
    @staticmethod
    def rotation_y(angle: float) -> Mat3:
        c, s = cos(angle), sin(angle)
        return Mat3([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def rotation_z(angle: float) -> Mat3:
        c, s = cos(angle), sin(angle)
        return Mat3([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])


class Quat:
    __slots__ = ('w', 'x', 'y', 'z')
    
    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def __mul__(self, other: Union[Quat, float]) -> Quat:
        if isinstance(other, Quat):
            return Quat(
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            )
        else:
            return Quat(self.w * other, self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, s: float) -> Quat:
        return Quat(self.w * s, self.x * s, self.y * s, self.z * s)
    
    def __add__(self, o: Quat) -> Quat:
        return Quat(self.w + o.w, self.x + o.x, self.y + o.y, self.z + o.z)
    
    def __repr__(self) -> str:
        return f"Quat({self.w:.4f}, {self.x:.4f}, {self.y:.4f}, {self.z:.4f})"
    
    def conjugate(self) -> Quat:
        return Quat(self.w, -self.x, -self.y, -self.z)
    
    def mag_sq(self) -> float:
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    
    def mag(self) -> float:
        return sqrt(self.mag_sq())
    
    def normalized(self) -> Quat:
        m = self.mag()
        if m < 1e-10:
            return Quat()
        inv_m = 1.0 / m
        return Quat(self.w * inv_m, self.x * inv_m, self.y * inv_m, self.z * inv_m)
    
    def inverse(self) -> Quat:
        m_sq = self.mag_sq()
        if m_sq < 1e-10:
            return Quat()
        inv_m_sq = 1.0 / m_sq
        return Quat(self.w * inv_m_sq, -self.x * inv_m_sq, -self.y * inv_m_sq, -self.z * inv_m_sq)
    
    def rotate_vec(self, v: Vec3) -> Vec3:
        qv = Quat(0, v.x, v.y, v.z)
        result = self * qv * self.conjugate()
        return Vec3(result.x, result.y, result.z)
    
    def derivative(self, omega: Vec3) -> Quat:
        omega_q = Quat(0, omega.x, omega.y, omega.z)
        dq = omega_q * self * 0.5
        return dq
    
    def to_mat3(self) -> Mat3:
        w, x, y, z = self.w, self.x, self.y, self.z
        
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        
        return Mat3([
            [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
            [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
            [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
        ])
    
    def copy(self) -> Quat:
        return Quat(self.w, self.x, self.y, self.z)
    
    @staticmethod
    def identity() -> Quat:
        return Quat(1, 0, 0, 0)
    
    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> Quat:
        axis = axis.normalized()
        half = angle * 0.5
        s = sin(half)
        return Quat(cos(half), axis.x * s, axis.y * s, axis.z * s)
    
    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> Quat:
        cr, sr = cos(roll * 0.5), sin(roll * 0.5)
        cp, sp = cos(pitch * 0.5), sin(pitch * 0.5)
        cy, sy = cos(yaw * 0.5), sin(yaw * 0.5)
        
        return Quat(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )
    
    @staticmethod
    def from_mat3(m: Mat3) -> Quat:
        trace = m.m[0, 0] + m.m[1, 1] + m.m[2, 2]
        
        if trace > 0:
            s = sqrt(trace + 1.0) * 2
            return Quat(
                0.25 * s,
                (m.m[2, 1] - m.m[1, 2]) / s,
                (m.m[0, 2] - m.m[2, 0]) / s,
                (m.m[1, 0] - m.m[0, 1]) / s
            )
        elif m.m[0, 0] > m.m[1, 1] and m.m[0, 0] > m.m[2, 2]:
            s = sqrt(1.0 + m.m[0, 0] - m.m[1, 1] - m.m[2, 2]) * 2
            return Quat(
                (m.m[2, 1] - m.m[1, 2]) / s,
                0.25 * s,
                (m.m[0, 1] + m.m[1, 0]) / s,
                (m.m[0, 2] + m.m[2, 0]) / s
            )
        elif m.m[1, 1] > m.m[2, 2]:
            s = sqrt(1.0 + m.m[1, 1] - m.m[0, 0] - m.m[2, 2]) * 2
            return Quat(
                (m.m[0, 2] - m.m[2, 0]) / s,
                (m.m[0, 1] + m.m[1, 0]) / s,
                0.25 * s,
                (m.m[1, 2] + m.m[2, 1]) / s
            )
        else:
            s = sqrt(1.0 + m.m[2, 2] - m.m[0, 0] - m.m[1, 1]) * 2
            return Quat(
                (m.m[1, 0] - m.m[0, 1]) / s,
                (m.m[0, 2] + m.m[2, 0]) / s,
                (m.m[1, 2] + m.m[2, 1]) / s,
                0.25 * s
            )
    
    @staticmethod
    def slerp(a: Quat, b: Quat, t: float) -> Quat:
        dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z
        
        if dot < 0:
            b = Quat(-b.w, -b.x, -b.y, -b.z)
            dot = -dot
        
        if dot > 0.9995:
            result = a + (Quat(b.w - a.w, b.x - a.x, b.y - a.y, b.z - a.z) * t)
            return result.normalized()
        
        theta_0 = acos(dot)
        theta = theta_0 * t
        
        sin_theta = sin(theta)
        sin_theta_0 = sin(theta_0)
        
        s0 = cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return Quat(
            s0 * a.w + s1 * b.w,
            s0 * a.x + s1 * b.x,
            s0 * a.y + s1 * b.y,
            s0 * a.z + s1 * b.z
        )

def lerp(a: Union[float, Vec3], b: Union[float, Vec3], t: float) -> Union[float, Vec3]:
    return a + (b - a) * t

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def nearly_zero(v: float, eps: float = 1e-10) -> bool:
    return abs(v) < eps

def nearly_equal(a: float, b: float, eps: float = 1e-10) -> bool:
    return abs(a - b) < eps
