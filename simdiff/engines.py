"""엔진별 실행기. 모두 같은 형식 (t, theta, omega) 을 돌려준다."""
import os
import tempfile

import numpy as np

from .scenes import pendulum as pend


def run_mujoco(params, theta0, duration, timestep, integrator="Euler"):
    import mujoco

    model = mujoco.MjModel.from_xml_string(pend.mjcf(params, timestep, integrator))
    data = mujoco.MjData(model)
    data.qpos[0] = theta0
    mujoco.mj_forward(model, data)

    n = int(round(duration / timestep))
    out = np.empty((n + 1, 3))
    out[0] = (0.0, data.qpos[0], data.qvel[0])
    for i in range(1, n + 1):
        mujoco.mj_step(model, data)
        out[i] = (data.time, data.qpos[0], data.qvel[0])
    return out


# PyBullet 모드별 설정.
#   default: URDF를 그대로 불러오고 PyBullet 기본값을 건드리지 않음 (초보자가 흔히 하는 방식)
#   clean:   관성은 파일값 사용, 기본 링크 감쇠 0, 관절 모터 해제
PYBULLET_MODES = {
    "default": dict(use_file_inertia=False, zero_link_damping=False),
    "file_inertia_only": dict(use_file_inertia=True, zero_link_damping=False),
    "damping_only": dict(use_file_inertia=False, zero_link_damping=True),
    "clean": dict(use_file_inertia=True, zero_link_damping=True),
}


def run_pybullet(params, theta0, duration, timestep, mode="clean", solver_iterations=50):
    import pybullet as p

    cfg = PYBULLET_MODES[mode]
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -params.gravity, physicsClientId=cid)
        p.setPhysicsEngineParameter(fixedTimeStep=timestep, numSubSteps=0,
                                    numSolverIterations=solver_iterations, physicsClientId=cid)

        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(pend.urdf(params))
            path = f.name
        flags = p.URDF_USE_INERTIA_FROM_FILE if cfg["use_file_inertia"] else 0
        body = p.loadURDF(path, useFixedBase=True, flags=flags, physicsClientId=cid)
        os.unlink(path)

        # 관절 모터는 기본적으로 속도 0을 유지하려 하므로 반드시 꺼야 한다 (모든 모드 공통).
        p.setJointMotorControl2(body, 0, p.VELOCITY_CONTROL, force=0, physicsClientId=cid)
        if cfg["zero_link_damping"]:
            p.changeDynamics(body, 0, linearDamping=0, angularDamping=0, physicsClientId=cid)
        p.resetJointState(body, 0, theta0, 0.0, physicsClientId=cid)

        n = int(round(duration / timestep))
        out = np.empty((n + 1, 3))
        th, om = p.getJointState(body, 0, physicsClientId=cid)[:2]
        out[0] = (0.0, th, om)
        for i in range(1, n + 1):
            p.stepSimulation(physicsClientId=cid)
            th, om = p.getJointState(body, 0, physicsClientId=cid)[:2]
            out[i] = (i * timestep, th, om)
        return out
    finally:
        p.disconnect(cid)


def pybullet_effective_inertia(params, mode):
    """PyBullet이 실제로 쓰는 링크 관성값을 확인 (기본값 함정 진단용)."""
    import pybullet as p

    cfg = PYBULLET_MODES[mode]
    cid = p.connect(p.DIRECT)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(pend.urdf(params))
            path = f.name
        flags = p.URDF_USE_INERTIA_FROM_FILE if cfg["use_file_inertia"] else 0
        body = p.loadURDF(path, useFixedBase=True, flags=flags, physicsClientId=cid)
        os.unlink(path)
        info = p.getDynamicsInfo(body, 0, physicsClientId=cid)
        return {"mass": info[0], "local_inertia_diag": info[2]}
    finally:
        p.disconnect(cid)


# ================================================================ 다관절 체인 공용 실행기
# 반환 형식: (N+1, 1 + 2*nq) = [t, q_1..q_n, qd_1..qd_n]

def run_mujoco_chain(xml, q0, duration):
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nq = model.nq
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    n = int(round(duration / dt))
    out = np.empty((n + 1, 1 + 2 * nq))
    out[0] = np.concatenate([[0.0], data.qpos, data.qvel])
    for i in range(1, n + 1):
        mujoco.mj_step(model, data)
        out[i] = np.concatenate([[data.time], data.qpos, data.qvel])
    return out


def run_pybullet_chain(urdf_text, q0, duration, timestep, gravity=9.81, mode="clean",
                       solver_iterations=50):
    import pybullet as p

    cfg = PYBULLET_MODES[mode]
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -gravity, physicsClientId=cid)
        p.setPhysicsEngineParameter(fixedTimeStep=timestep, numSubSteps=0,
                                    numSolverIterations=solver_iterations, physicsClientId=cid)
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(urdf_text)
            path = f.name
        flags = p.URDF_USE_INERTIA_FROM_FILE if cfg["use_file_inertia"] else 0
        body = p.loadURDF(path, useFixedBase=True, flags=flags, physicsClientId=cid)
        os.unlink(path)
        nj = p.getNumJoints(body, physicsClientId=cid)
        for j in range(nj):
            p.setJointMotorControl2(body, j, p.VELOCITY_CONTROL, force=0, physicsClientId=cid)
            if cfg["zero_link_damping"]:
                p.changeDynamics(body, j, linearDamping=0, angularDamping=0, physicsClientId=cid)
            p.resetJointState(body, j, q0[j], 0.0, physicsClientId=cid)

        def state():
            s = p.getJointStates(body, list(range(nj)), physicsClientId=cid)
            return [x[0] for x in s], [x[1] for x in s]

        n = int(round(duration / timestep))
        out = np.empty((n + 1, 1 + 2 * nj))
        q, qd = state()
        out[0] = [0.0, *q, *qd]
        for i in range(1, n + 1):
            p.stepSimulation(physicsClientId=cid)
            q, qd = state()
            out[i] = [i * timestep, *q, *qd]
        return out
    finally:
        p.disconnect(cid)
