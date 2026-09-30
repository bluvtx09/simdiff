"""SimDiff 2단계-C: Sobol 민감도 분석 (미끄러지는 상자).

"접촉 결과(정지 거리 오차)를 가장 크게 흔드는 것은 무엇인가?"를 분산 기반으로 분해한다.
물리 조건(mu, v0)과 엔진 설정(타임스텝, 솔버 등)을 한 번에 넣어서,
오차가 '어떤 장면이냐'에서 오는지 '어떻게 설정했냐'에서 오는지 비교한다.

범주형 설정(cone, integrator)은 [0,1] 값을 반으로 나눠 두 범주에 대응시킨다.

python run_sobol.py [N]   (기본 N=256, 엔진당 N*(2D+2)회 실행)
→ results/sobol_mujoco.csv, results/sobol_pybullet.csv, results/sobol_samples_*.npz
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from simdiff.scenes import sliding_box as sb

ROOT = Path(__file__).parent
OUT = ROOT / "results"

PROBLEMS = {
    "mujoco": {
        "num_vars": 7,
        "names": ["log10_dt", "mu", "v0", "log10_timeconst", "impratio", "cone", "integrator"],
        "labels": ["타임스텝", "마찰계수 μ", "초기 속도", "접촉 강성(timeconst)", "impratio",
                   "마찰 원뿔 종류", "적분기"],
        "kind": ["설정", "물리", "물리", "설정", "설정", "설정", "설정"],
        "bounds": [[-3.3, -2.0], [0.1, 0.8], [0.5, 3.0], [-2.3, -1.3], [1.0, 10.0], [0, 1], [0, 1]],
    },
    "pybullet": {
        "num_vars": 6,
        "names": ["log10_dt", "mu", "v0", "log10_iterations", "contact_erp", "friction_erp"],
        "labels": ["타임스텝", "마찰계수 μ", "초기 속도", "솔버 반복 횟수", "contactERP", "frictionERP"],
        "kind": ["설정", "물리", "물리", "설정", "설정", "설정"],
        "bounds": [[-3.3, -2.0], [0.1, 0.8], [0.5, 3.0], [0.0, 2.0], [0.05, 0.8], [0.05, 0.8]],
    },
}


def eval_mujoco(x):
    ldt, mu, v0, ltc, imp, cone, integ = x
    c = sb.BoxCase(mu, v0)
    r = sb.run_mujoco(c, dt=10 ** ldt, timeconst=10 ** ltc, impratio=imp,
                      cone="pyramidal" if cone < 0.5 else "elliptic",
                      integrator="Euler" if integ < 0.5 else "implicitfast")
    return (r["d_sim"] / c.d_true - 1) * 100


def eval_pybullet(x):
    ldt, mu, v0, lit, cerp, ferp = x
    c = sb.BoxCase(mu, v0)
    r = sb.run_pybullet(c, dt=10 ** ldt, iterations=max(1, int(round(10 ** lit))),
                        contact_erp=cerp, friction_erp=ferp)
    return (r["d_sim"] / c.d_true - 1) * 100


def main(N):
    OUT.mkdir(exist_ok=True)
    for engine, fn in [("mujoco", eval_mujoco), ("pybullet", eval_pybullet)]:
        prob = {k: PROBLEMS[engine][k] for k in ("num_vars", "names", "bounds")}
        X = sobol_sample.sample(prob, N, calc_second_order=True, seed=0)
        t0 = time.time()
        with Pool(2) as pool:
            Y = np.array(pool.map(fn, X, chunksize=32))
        el = time.time() - t0
        Yabs = np.abs(Y)
        Si = sobol_analyze.analyze(prob, Yabs, calc_second_order=True, seed=0, print_to_console=False)
        df = pd.DataFrame({"name": prob["names"], "label": PROBLEMS[engine]["labels"],
                           "kind": PROBLEMS[engine]["kind"],
                           "S1": Si["S1"], "S1_conf": Si["S1_conf"],
                           "ST": Si["ST"], "ST_conf": Si["ST_conf"]}).sort_values("ST", ascending=False)
        df.to_csv(OUT / f"sobol_{engine}.csv", index=False)
        np.savez(OUT / f"sobol_samples_{engine}.npz", X=X, Y=Y)
        print(f"[{engine}] {len(X)}회 실행 ({el:.0f}s), |오차| 평균 {Yabs.mean():.2f}%, 최대 {Yabs.max():.2f}%")
        print(df[["label", "S1", "ST", "ST_conf"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
