"""단진자(물리 진자) 장면.

세 가지 구현이 '같은 물리'를 공유하도록 파라미터를 한 곳에서 관리한다.
  - reference: scipy 고정밀 ODE 해 (기준 해, 사실상 '정답')
  - mujoco:    MJCF 모델
  - pybullet:  URDF 모델

각도 규약: theta = 0 이 아래로 매달린 상태, 회전축은 y축.
운동방정식: (I_com + m L^2) * theta'' = -m g L sin(theta) - b * theta'
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PendulumParams:
    mass: float = 1.0          # kg
    length: float = 0.5        # m, 힌지~질량중심 거리
    i_com: float = 0.002       # kg m^2, 질량중심 기준 관성모멘트
    damping: float = 0.0       # N m s/rad, 관절 감쇠
    gravity: float = 9.81      # m/s^2

    @property
    def i_pivot(self) -> float:
        return self.i_com + self.mass * self.length ** 2

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- 기준 해
def reference(params: PendulumParams, theta0: float, t: np.ndarray) -> np.ndarray:
    """고정밀(rtol=1e-12) 적분으로 만든 기준 궤적. shape (len(t), 2) = [theta, omega]."""
    p = params

    def f(_, y):
        th, om = y
        return [om, (-p.mass * p.gravity * p.length * np.sin(th) - p.damping * om) / p.i_pivot]

    sol = solve_ivp(f, (t[0], t[-1]), [theta0, 0.0], t_eval=t,
                    method="DOP853", rtol=1e-12, atol=1e-12)
    return sol.y.T


def energy(params: PendulumParams, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """총 역학적 에너지 (위치에너지 기준: 매달린 상태 = 0)."""
    p = params
    return 0.5 * p.i_pivot * omega ** 2 + p.mass * p.gravity * p.length * (1 - np.cos(theta))


# ---------------------------------------------------------------- MuJoCo
def mjcf(params: PendulumParams, timestep: float, integrator: str) -> str:
    p = params
    return f"""
<mujoco model="pendulum">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 {-p.gravity}"/>
  <worldbody>
    <body name="link" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="{p.damping}" armature="0"/>
      <inertial pos="0 0 {-p.length}" mass="{p.mass}" diaginertia="{p.i_com} {p.i_com} {p.i_com}"/>
    </body>
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------- PyBullet
def urdf(params: PendulumParams) -> str:
    p = params
    return f"""<?xml version="1.0"?>
<robot name="pendulum">
  <link name="base">
    <inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
  </link>
  <link name="link">
    <inertial>
      <origin xyz="0 0 {-p.length}"/>
      <mass value="{p.mass}"/>
      <inertia ixx="{p.i_com}" ixy="0" ixz="0" iyy="{p.i_com}" iyz="0" izz="{p.i_com}"/>
    </inertial>
  </link>
  <joint name="hinge" type="continuous">
    <parent link="base"/>
    <child link="link"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="{p.damping}" friction="0"/>
  </joint>
</robot>
"""
