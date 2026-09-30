"""SimDiff 3단계-A3: PyBullet 최악 조건의 '입력 민감도' 측정.

퍼징이 찾은 PyBullet 최악 조건에서는 입력을 1e-12만 바꿔도 결과가 크게 달라졌다.
같은 조건을 아주 작게(1e-12, 정규화 공간) 흔들어 200번 돌려서, 결과가 사실상 무작위인지 확인한다.
대조군: 규칙 v2를 지키는 평범한 조건.

python pb_chaos.py  →  results/pb_chaos.json, results/pb_chaos_samples.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fuzz

ROOT = Path(__file__).parent
OUT = ROOT / "results"


def spread(z, rules, n=200, eps=1e-12, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fuzz.evaluate(("pybullet", np.clip(z + rng.uniform(-eps, eps, len(z)), 0, 1), rules))[0]
            for _ in range(n)]
    return np.array(vals)


def main():
    h = pd.read_csv(OUT / "fuzz_history.csv", float_precision="round_trip")
    d = h[h.target == "pybullet"]
    names = fuzz.USES["pybullet"]
    worst = d.loc[d.value.idxmax(), names].to_numpy(dtype=float)
    # 대조군: dt 2 ms 근처, mu 0.4, v0 2, 기본 iteration 50, contactERP 0.2
    lo = {n: fuzz.BOUNDS[n] for n in names}
    ctrl_x = {"log10_dt": np.log10(0.002), "mu": 0.4, "v0": 2.0, "pb_log10_iterations": np.log10(50),
              "pb_contact_erp": 0.2}
    ctrl = np.array([(ctrl_x[n] - lo[n][0]) / (lo[n][1] - lo[n][0]) for n in names])
    res, samples = {}, []
    for name, z in [("최악 조건", worst), ("평범한 조건", ctrl)]:
        v = spread(z, "v1")
        c, dt, x = fuzz.decode(z, names, "v1")
        res[name] = {"params": {k: float(val) for k, val in x.items()}, "dt_ms": dt * 1000,
                     "n": len(v), "perturbation": 1e-12,
                     "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
                     "std": float(v.std()), "share_over_1pct": float((v > 1).mean())}
        samples += [{"condition": name, "abs_err_pct": float(a)} for a in v]
    (OUT / "pb_chaos.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(samples).to_csv(OUT / "pb_chaos_samples.csv", index=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
