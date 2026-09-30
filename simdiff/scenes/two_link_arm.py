"""2링크 팔 (이중 물리진자) 장면.

두 링크 모두 균일 막대: 질량 m, 길이 L, 질량중심은 관절에서 L/2, 관성 m L^2 / 12.
각도 규약: q1 = 어깨 절대각(아래로 매달린 상태 0), q2 = 팔꿈치 상대각. 회전축 y.
큰 진폭에서는 혼돈 운동이라 아주 작은 오차도 지수적으로 불어난다.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ArmParams:
    m1: float = 1.0
    m2: float = 1.0
    L1: float = 0.5
    L2: float = 0.5
    gravity: float = 9.81

    @property
    def lc1(self):
        return self.L1 / 2

    @property
    def lc2(self):
        return self.L2 / 2

    @property
    def I1(self):
        return self.m1 * self.L1 ** 2 / 12

    @property
    def I2(self):
        return self.m2 * self.L2 ** 2 / 12


def mass_matrix(p: ArmParams, q2):
    c2 = np.cos(q2)
    m11 = p.I1 + p.I2 + p.m1 * p.lc1 ** 2 + p.m2 * (p.L1 ** 2 + p.lc2 ** 2 + 2 * p.L1 * p.lc2 * c2)
    m12 = p.I2 + p.m2 * (p.lc2 ** 2 + p.L1 * p.lc2 * c2)
    m22 = p.I2 + p.m2 * p.lc2 ** 2
    return m11, m12, m22


def rhs(p: ArmParams):
    g = p.gravity

    def f(_, y):
        q1, q2, w1, w2 = y
        m11, m12, m22 = mass_matrix(p, q2)
        h = p.m2 * p.L1 * p.lc2 * np.sin(q2)
        c1 = -h * w2 * (2 * w1 + w2)
        c2 = h * w1 ** 2
        g1 = (p.m1 * p.lc1 + p.m2 * p.L1) * g * np.sin(q1) + p.m2 * p.lc2 * g * np.sin(q1 + q2)
        g2 = p.m2 * p.lc2 * g * np.sin(q1 + q2)
        b1, b2 = -(c1 + g1), -(c2 + g2)
        det = m11 * m22 - m12 ** 2
        a1 = (m22 * b1 - m12 * b2) / det
        a2 = (m11 * b2 - m12 * b1) / det
        return [w1, w2, a1, a2]

    return f


def reference(p: ArmParams, q0, t, rtol=1e-12):
    sol = solve_ivp(rhs(p), (t[0], t[-1]), [q0[0], q0[1], 0.0, 0.0], t_eval=t,
                    method="DOP853", rtol=rtol, atol=rtol)
    return sol.y.T  # (N, 4) = q1, q2, w1, w2


def energy(p: ArmParams, q1, q2, w1, w2):
    m11, m12, m22 = mass_matrix(p, q2)
    T = 0.5 * (m11 * w1 ** 2 + 2 * m12 * w1 * w2 + m22 * w2 ** 2)
    V = -p.m1 * p.gravity * p.lc1 * np.cos(q1) - p.m2 * p.gravity * (p.L1 * np.cos(q1) + p.lc2 * np.cos(q1 + q2))
    return T + V


def tip(p: ArmParams, q1, q2):
    """팔 끝 좌표 (x, z). 엔진의 y축 회전 규약과 부호를 맞춤."""
    x = -(p.L1 * np.sin(q1) + p.L2 * np.sin(q1 + q2))
    z = -(p.L1 * np.cos(q1) + p.L2 * np.cos(q1 + q2))
    return x, z


def mjcf(p: ArmParams, timestep, integrator):
    return f"""
<mujoco model="two_link_arm">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 {-p.gravity}"/>
  <worldbody>
    <body name="link1">
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="0" armature="0"/>
      <inertial pos="0 0 {-p.lc1}" mass="{p.m1}" diaginertia="{p.I1} {p.I1} {p.I1}"/>
      <body name="link2" pos="0 0 {-p.L1}">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0" armature="0"/>
        <inertial pos="0 0 {-p.lc2}" mass="{p.m2}" diaginertia="{p.I2} {p.I2} {p.I2}"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def urdf(p: ArmParams):
    def inertial(m, lc, I):
        return (f'<inertial><origin xyz="0 0 {-lc}"/><mass value="{m}"/>'
                f'<inertia ixx="{I}" ixy="0" ixz="0" iyy="{I}" iyz="0" izz="{I}"/></inertial>')

    return f"""<?xml version="1.0"?>
<robot name="two_link_arm">
  <link name="base"><inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
  <link name="link1">{inertial(p.m1, p.lc1, p.I1)}</link>
  <link name="link2">{inertial(p.m2, p.lc2, p.I2)}</link>
  <joint name="shoulder" type="continuous"><parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/><dynamics damping="0" friction="0"/></joint>
  <joint name="elbow" type="continuous"><parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 {-p.L1}"/><axis xyz="0 1 0"/><dynamics damping="0" friction="0"/></joint>
</robot>
"""
