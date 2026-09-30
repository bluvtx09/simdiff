"""SimDiff 2단계-A 실행: 2링크 팔의 예측 가능 시간(horizon) 측정.

python run_arm.py  →  results/arm_runs.csv, results/arm_ref_check.csv, results/arm_traj.npz
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import solve_ivp

from simdiff import engines
from simdiff.scenes import two_link_arm as arm

ROOT = Path(__file__).parent


def dense_reference(p, q0, T, rtol):
    y0 = [q0[0], q0[1], 0.0, 0.0]
    return solve_ivp(arm.rhs(p), (0, T), y0, method="DOP853", rtol=rtol, atol=rtol,
                     dense_output=True).sol


def horizon(t, qa, qb, thr):
    """두 관절각 궤적의 차이가 thr을 처음 넘는 시각. 끝까지 안 넘으면 inf."""
    d = np.max(np.abs(qa - qb), axis=1)
    idx = np.argmax(d > thr)
    return float(t[idx]) if d[idx] > thr else float("inf")


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    p = arm.ArmParams()
    T, thr = cfg["duration"], cfg["threshold_rad"]

    ics = [dict(name=c["name"], q=np.deg2rad(c["q_deg"])) for c in cfg["initial_conditions"]]
    rng = np.random.default_rng(cfg["random_chaotic"]["seed"])
    rc = cfg["random_chaotic"]
    for i in range(rc["n"]):
        q = np.deg2rad([rng.uniform(*rc["q1_deg"]), rng.uniform(*rc["q2_deg"])])
        ics.append(dict(name=f"rand{i:02d}", q=q))

    rows, ref_rows, keep = [], [], {}
    t0 = time.time()
    for ic in ics:
        q0 = ic["q"]
        ref = dense_reference(p, q0, T, 1e-12)
        ref_lo = dense_reference(p, q0, T, 1e-10)
        ref_pert = dense_reference(p, q0 + cfg["butterfly_perturb_rad"], T, 1e-12)
        tt = np.linspace(0, T, 20001)
        R = ref(tt).T
        E = arm.energy(p, *R.T)
        ref_rows.append({
            "ic": ic["name"], "q1_deg": float(np.rad2deg(q0[0])), "q2_deg": float(np.rad2deg(q0[1])),
            # 기준 해 자체가 믿을 수 있는 시간: 허용오차 1e-12 와 1e-10 해가 1e-3 rad 이상 벌어지기 전까지
            "ref_valid_until": horizon(tt, R[:, :2], ref_lo(tt).T[:, :2], 1e-3),
            # 나비효과 한계: 초기각을 1e-9 rad 흔들었을 때의 예측 가능 시간 (물리 자체의 한계)
            "butterfly_horizon": horizon(tt, R[:, :2], ref_pert(tt).T[:, :2], thr),
            "ref_energy_drift": float(np.ptp(E) / max(abs(E[0]), 1e-9)),
        })
        for dt in cfg["timesteps"]:
            n = int(round(T / dt)); t = np.arange(n + 1) * dt
            Rq = ref(t).T[:, :2]
            trajs = {}
            for c in cfg["configs"]:
                if c["engine"] == "mujoco":
                    a = engines.run_mujoco_chain(arm.mjcf(p, dt, c["integrator"]), q0, T)
                else:
                    a = engines.run_pybullet_chain(arm.urdf(p), q0, T, dt, mode=c["mode"])
                trajs[c["label"]] = a
                Ea = arm.energy(p, *a[:, 1:5].T)
                rows.append({
                    "ic": ic["name"], "dt": dt, "config": c["label"],
                    "horizon_vs_ref": horizon(t, a[:, 1:3], Rq, thr),
                    "err_at_1s": float(np.max(np.abs(a[int(round(1 / dt)), 1:3] - Rq[int(round(1 / dt))]))),
                    "energy_drift_rel": float((Ea[-1] - Ea[0]) / abs(E[0])) if abs(E[0]) > 1e-9 else float("nan"),
                })
            # 엔진 간 예측 가능 시간 (MuJoCo Euler vs PyBullet clean)
            h = horizon(t, trajs["MuJoCo Euler"][:, 1:3], trajs["PyBullet clean"][:, 1:3], thr)
            rows.append({"ic": ic["name"], "dt": dt, "config": "엔진 간 (MuJoCo Euler vs PyBullet)",
                         "horizon_vs_ref": h, "err_at_1s": float(np.max(np.abs(
                             trajs["MuJoCo Euler"][int(round(1 / dt)), 1:3]
                             - trajs["PyBullet clean"][int(round(1 / dt)), 1:3]))),
                         "energy_drift_rel": float("nan")})
            if ic["name"] == "high" and dt == 0.001:
                keep = {k: v[:, :3] for k, v in trajs.items()}
                keep["reference"] = np.column_stack([t, Rq])
        print(f"  {ic['name']} 완료 ({time.time() - t0:.0f}s)", flush=True)

    for fname, data in [("arm_runs.csv", rows), ("arm_ref_check.csv", ref_rows)]:
        with open(out / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)
    np.savez(out / "arm_traj.npz", **keep)
    print(f"{len(rows)}개 행 완료 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "experiments" / "arm_sweep.yaml")
