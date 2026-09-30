"""SimDiff 1단계 실행 스크립트: 진자 재현성 감사.

사용법:
    python run_pendulum.py                       # 기본 설정
    python run_pendulum.py experiments/xxx.yaml  # 다른 설정 파일

결과:
    results/pendulum_runs.csv     조건별 지표
    results/pendulum_traj.npz     그래프용 대표 궤적
    results/env.json              실행 환경 (엔진 버전 등)
"""
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from simdiff import engines
from simdiff.scenes import pendulum as pend

ROOT = Path(__file__).parent


def run_config(cfg, params, theta0, duration, dt):
    if cfg["engine"] == "mujoco":
        return engines.run_mujoco(params, theta0, duration, dt, cfg["integrator"])
    return engines.run_pybullet(params, theta0, duration, dt, cfg["mode"])


def metrics(params, traj, ref):
    err = traj[:, 1] - ref[:, 0]
    e_sim = pend.energy(params, traj[:, 1], traj[:, 2])
    e_ref = pend.energy(params, ref[:, 0], ref[:, 1])
    e0 = e_ref[0]
    return {
        "rmse_rad": float(np.sqrt(np.mean(err ** 2))),
        "max_err_rad": float(np.max(np.abs(err))),
        # 기준 해 대비 에너지 오차 (초기 에너지로 정규화). 감쇠가 있어도 공정하게 비교 가능.
        "energy_err_final": float((e_sim[-1] - e_ref[-1]) / e0),
        "energy_err_maxabs": float(np.max(np.abs(e_sim - e_ref)) / e0),
    }


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    import mujoco
    import pybullet
    import scipy
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "pybullet_api_version": pybullet.getAPIVersion(),
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "config": cfg,
        "pybullet_effective_inertia": {
            m: engines.pybullet_effective_inertia(pend.PendulumParams(**cfg["params"]), m)
            for m in engines.PYBULLET_MODES
        },
    }

    rows, keep = [], {}
    t0 = time.time()
    for damping in cfg["dampings"]:
        params = pend.PendulumParams(**cfg["params"], damping=damping)
        for th_deg in cfg["theta0_deg"]:
            theta0 = np.deg2rad(th_deg)
            for dt in cfg["timesteps"]:
                n = int(round(cfg["duration"] / dt))
                t = np.arange(n + 1) * dt
                ref = pend.reference(params, theta0, t)
                for c in cfg["configs"]:
                    runs = [run_config(c, params, theta0, cfg["duration"], dt)
                            for _ in range(cfg["repeat_for_determinism"])]
                    deterministic = all(np.array_equal(runs[0], r) for r in runs[1:])
                    m = metrics(params, runs[0], ref)
                    rows.append({"engine": c["engine"], "config": c["label"],
                                 "damping": damping, "theta0_deg": th_deg, "dt": dt,
                                 "deterministic": deterministic, **m})
                    # 그래프용 대표 궤적: 감쇠 없음, 90도, dt=0.01
                    if damping == 0.0 and th_deg == 90 and dt == 0.01:
                        keep[c["label"]] = runs[0]
                        keep["reference"] = np.column_stack([t, ref])
    elapsed = time.time() - t0

    with open(out_dir / "pendulum_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(out_dir / "pendulum_traj.npz", **{k: v for k, v in keep.items()})
    env["runs"] = len(rows)
    env["elapsed_sec"] = round(elapsed, 1)
    (out_dir / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)}개 조건 완료 ({elapsed:.1f}s) → results/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "experiments" / "pendulum_sweep.yaml")
