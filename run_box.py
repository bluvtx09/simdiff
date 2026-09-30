"""SimDiff 2단계-B 실행: 미끄러지는 상자 (접촉·마찰).

  (a) 기본값 함정: PyBullet 마찰 결합 방식, 기본 감쇠
  (b) 격자 실험: 물리 조건 9개 x 타임스텝 5개 x 엔진 설정 → 뒤집힘 지수 재료

python run_box.py  →  results/box_traps.csv, results/box_grid.csv
"""
import csv
import itertools
import time
from pathlib import Path

import numpy as np

from simdiff.scenes import sliding_box as sb

ROOT = Path(__file__).parent
OUT = ROOT / "results"

MUS = [0.2, 0.4, 0.6]
V0S = [1.0, 2.0, 3.0]
DTS = [0.0005, 0.001, 0.002, 0.005, 0.01]


def rel_err(r, case):
    return (r["d_sim"] / case.d_true - 1) * 100


def row(engine, setting, case, dt, r):
    return {"engine": engine, "setting": setting, "mu": case.mu, "v0": case.v0, "dt": dt,
            "d_true": case.d_true, "d_sim": r["d_sim"], "err_pct": rel_err(r, case),
            "z_final": r["z_final"]}


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    # (a) 함정
    traps = []
    for mu, v0 in itertools.product(MUS, V0S):
        c = sb.BoxCase(mu, v0)
        for label, kw in [("PyBullet 올바른 설정", {}),
                          ("PyBullet 마찰을 양쪽에 mu로 입력", dict(friction_combine="naive")),
                          ("PyBullet 기본 감쇠 유지", dict(zero_damping=False))]:
            traps.append(row("pybullet", label, c, 0.002, sb.run_pybullet(c, dt=0.002, **kw)))
        traps.append(row("mujoco", "MuJoCo 기본값", c, 0.002, sb.run_mujoco(c, dt=0.002)))

    # (b) 격자
    grid = []
    mj_settings = {"MuJoCo pyramidal(기본)": dict(cone="pyramidal"),
                   "MuJoCo elliptic": dict(cone="elliptic")}
    pb_settings = {"PyBullet iter 50(기본)": dict(iterations=50),
                   "PyBullet iter 10": dict(iterations=10)}
    for mu, v0 in itertools.product(MUS, V0S):
        c = sb.BoxCase(mu, v0)
        for dt in DTS:
            for name, kw in mj_settings.items():
                grid.append(row("mujoco", name, c, dt, sb.run_mujoco(c, dt=dt, **kw)))
            for name, kw in pb_settings.items():
                grid.append(row("pybullet", name, c, dt, sb.run_pybullet(c, dt=dt, **kw)))

    for fname, data in [("box_traps.csv", traps), ("box_grid.csv", grid)]:
        with open(OUT / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)
    print(f"함정 {len(traps)}행, 격자 {len(grid)}행 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
