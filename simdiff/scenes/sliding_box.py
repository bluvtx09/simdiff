"""미끄러지는 상자 (접촉 + 마찰) 장면.

바닥 위 정지한 상자에 수평 속도 v0를 주면 쿨롱 마찰로 감속해 멈춘다.
해석해:  정지 거리 d = v0^2 / (2 mu g),  정지 시간 t = v0 / (mu g)
상자는 정육면체(반변 0.05 m), mu <= 0.8 이면 넘어지지 않는다 (넘어짐 조건 mu > 1).

절차 (두 엔진 공통):
  1) 상자를 바닥에 올려 settle_time 동안 가만히 둬서 접촉을 안정시킨다
  2) x 방향 속도 v0를 준다
  3) 해석적 정지 시간 + margin 까지 시뮬레이션하고 이동 거리를 잰다
"""
import os
import tempfile
from dataclasses import dataclass

import numpy as np

HALF = 0.05
MASS = 1.0
G = 9.81
SETTLE = 0.3
MARGIN = 0.7


@dataclass
class BoxCase:
    mu: float = 0.4
    v0: float = 2.0

    @property
    def d_true(self):
        return self.v0 ** 2 / (2 * self.mu * G)

    @property
    def t_stop(self):
        return self.v0 / (self.mu * G)


def tilt_deg(qx, qy):
    """몸체 z축이 연직에서 기운 각도 (요잉 회전은 제외, 순수하게 넘어진 정도)."""
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (qx ** 2 + qy ** 2), -1, 1))))


def _set_mujoco_v(data, v0):
    data.qvel[:] = 0
    data.qvel[0] = v0


def _inertia(half_z):
    """x, y, z축 관성 (바닥 반변 HALF, 높이 반변 half_z)."""
    ixy = MASS * ((2 * HALF) ** 2 + (2 * half_z) ** 2) / 12
    iz = MASS * (2 * HALF) ** 2 / 6
    return ixy, ixy, iz


def run_mujoco(case: BoxCase, dt=0.002, integrator="Euler", cone="pyramidal", impratio=1.0,
               timeconst=0.02, dampratio=1.0, iterations=100, noslip_iterations=0, half_z=HALF):
    """half_z: 상자 높이의 절반. 기본은 정육면체. 넘어짐 여유 = mu * half_z / HALF (1 미만이면 강체 이론상 안 넘어짐)."""
    import mujoco

    Ix, Iy, Iz = _inertia(half_z)
    xml = f"""
<mujoco>
  <option timestep="{dt}" integrator="{integrator}" cone="{cone}" impratio="{impratio}"
          iterations="{int(iterations)}" noslip_iterations="{int(noslip_iterations)}" gravity="0 0 {-G}"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1" friction="{case.mu} 0 0"
          solref="{timeconst} {dampratio}"/>
    <body name="box" pos="0 0 {half_z}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{MASS}" diaginertia="{Ix} {Iy} {Iz}"/>
      <geom type="box" size="{HALF} {HALF} {half_z}" friction="{case.mu} 0 0"
            solref="{timeconst} {dampratio}" condim="3"/>
    </body>
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(int(round(SETTLE / dt))):
        mujoco.mj_step(model, data)
    x0 = data.qpos[0]
    _set_mujoco_v(data, case.v0)
    n = int(round((case.t_stop + MARGIN) / dt))
    xs = np.empty(n + 1); xs[0] = 0.0
    for i in range(1, n + 1):
        mujoco.mj_step(model, data)
        xs[i] = data.qpos[0] - x0
    return {"d_sim": float(xs[-1]), "x": xs, "dt": dt,
            "tilt_deg": tilt_deg(data.qpos[4], data.qpos[5]),
            "z_final": float(data.qpos[2])}


def run_pybullet(case: BoxCase, dt=0.002, iterations=50, erp=0.2, contact_erp=0.2,
                 friction_erp=0.2, friction_combine="correct", zero_damping=True, half_z=HALF):
    """friction_combine:
         correct: 바닥 마찰 1.0, 상자 마찰 mu → PyBullet은 두 값을 곱하므로 실효 mu
         naive:   바닥과 상자 모두 mu (MuJoCo 방식대로 적으면 실효 mu^2 이 됨)
    """
    import pybullet as p

    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -G, physicsClientId=cid)
        p.setPhysicsEngineParameter(fixedTimeStep=dt, numSubSteps=0, numSolverIterations=int(iterations),
                                    erp=erp, contactERP=contact_erp, frictionERP=friction_erp,
                                    physicsClientId=cid)
        plane = p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid),
                                  physicsClientId=cid)
        box = p.createMultiBody(MASS, p.createCollisionShape(p.GEOM_BOX, halfExtents=[HALF, HALF, half_z],
                                                             physicsClientId=cid),
                                basePosition=[0, 0, half_z], physicsClientId=cid)
        floor_mu = 1.0 if friction_combine == "correct" else case.mu
        p.changeDynamics(plane, -1, lateralFriction=floor_mu, restitution=0, physicsClientId=cid)
        p.changeDynamics(box, -1, lateralFriction=case.mu, restitution=0, physicsClientId=cid)
        if zero_damping:
            p.changeDynamics(box, -1, linearDamping=0, angularDamping=0, physicsClientId=cid)
        for _ in range(int(round(SETTLE / dt))):
            p.stepSimulation(physicsClientId=cid)
        x0 = p.getBasePositionAndOrientation(box, physicsClientId=cid)[0][0]
        p.resetBaseVelocity(box, [case.v0, 0, 0], [0, 0, 0], physicsClientId=cid)
        n = int(round((case.t_stop + MARGIN) / dt))
        xs = np.empty(n + 1); xs[0] = 0.0
        for i in range(1, n + 1):
            p.stepSimulation(physicsClientId=cid)
            xs[i] = p.getBasePositionAndOrientation(box, physicsClientId=cid)[0][0] - x0
        pos, orn = p.getBasePositionAndOrientation(box, physicsClientId=cid)
        return {"d_sim": float(xs[-1]), "x": xs, "dt": dt,
                "tilt_deg": tilt_deg(orn[0], orn[1]),
                "z_final": float(pos[2])}
    finally:
        p.disconnect(cid)
