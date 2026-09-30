"""SimDiff 3단계-A: 퍼징 — 시뮬레이터가 가장 크게 틀리는 조건 자동 탐색.

2단계에서 얻은 실용 규칙("정지까지 300스텝 이상이면 오차 1% 미만")의 반례를 찾는다.
모든 후보는 규칙을 지키도록 만든다: dt = min(샘플한 dt, 정지시간 / 300).

목표 3가지:
  mujoco    : MuJoCo 정지 거리 |오차| 최대화
  pybullet  : PyBullet 정지 거리 |오차| 최대화
  disagree  : 같은 물리·같은 dt에서 두 엔진 결과 차이 최대화

탐색기 2가지 (같은 평가 예산):
  CMA-ES    : 진화 전략 (좋은 후보 주변을 집중 탐색)
  random    : 균일 무작위 탐색 (기준선)

python fuzz.py [budget] [seeds]                 →  results/fuzz_history.csv, fuzz_best.json
python fuzz.py [budget] [seeds] --rules v1_mu  →  넘어짐 여유 규칙 추가 검증, *_refined 파일
python fuzz.py [budget] [seeds] --rules v2     →  2차 보완 규칙 검증, *_v2 파일
python fuzz.py [budget] [seeds] --rules v3     →  최종 규칙 검증, *_v3 파일
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cma
import numpy as np
import pandas as pd

from simdiff.scenes import sliding_box as sb

ROOT = Path(__file__).parent
OUT = ROOT / "results"
MIN_STEPS = 300

# 공통 물리·시간 변수 + 엔진별 설정. 모두 [0,1]로 정규화해서 탐색.
SPACE = [
    ("log10_dt", -3.3, -2.0),
    ("mu", 0.1, 0.8),
    ("v0", 0.5, 3.0),
    ("mj_log10_timeconst", -2.3, -1.3),
    ("mj_impratio", 1.0, 10.0),
    ("mj_cone", 0.0, 1.0),
    ("pb_log10_iterations", 0.0, 2.0),
    ("pb_contact_erp", 0.05, 0.8),
]
USES = {  # 각 목표가 실제로 쓰는 변수 (안 쓰는 변수는 탐색에서 뺀다)
    "mujoco": ["log10_dt", "mu", "v0", "mj_log10_timeconst", "mj_impratio", "mj_cone"],
    "pybullet": ["log10_dt", "mu", "v0", "pb_log10_iterations", "pb_contact_erp"],
    "disagree": [s[0] for s in SPACE],
}
BOUNDS = {n: (lo, hi) for n, lo, hi in SPACE}

# 검증할 규칙 버전. 모든 후보를 규칙 안으로 잘라 넣어서(clip) 규칙을 지키는 조건만 탐색한다.
#   v1     : 정지까지 300스텝 이상 (2단계 규칙)
#   v1_mu  : v1 + 넘어짐 여유 tau = mu*h/b <= 0.6 (1차 퍼징 후 보완)
#   v2     : v1_mu + MuJoCo timeconst <= 0.2 * 정지시간, PyBullet dt <= 5 ms, contactERP <= 0.6
#   v3     : v2 + MuJoCo timeconst >= 10 * dt
RULES = {"v1": dict(tau_max=None, tc_frac=None, dt_max=None, erp_max=None, tc_min_dt=None),
         "v1_mu": dict(tau_max=0.6, tc_frac=None, dt_max=None, erp_max=None, tc_min_dt=None),
         "v2": dict(tau_max=0.6, tc_frac=0.2, dt_max=0.005, erp_max=0.6, tc_min_dt=None),
         "v3": dict(tau_max=0.6, tc_frac=0.2, dt_max=0.005, erp_max=0.6, tc_min_dt=10)}
TAGS = {"v1": "", "v1_mu": "_refined", "v2": "_v2", "v3": "_v3"}


def decode(z, names, rules="v1"):
    R = RULES[rules]
    z = np.clip(z, 0, 1)
    x = {n: BOUNDS[n][0] + zi * (BOUNDS[n][1] - BOUNDS[n][0]) for n, zi in zip(names, z)}
    if R["tau_max"] is not None:  # 정육면체라 tau = mu
        x["mu"] = BOUNDS["mu"][0] + z[names.index("mu")] * (R["tau_max"] - BOUNDS["mu"][0])
    c = sb.BoxCase(x["mu"], x["v0"])
    dt = min(10 ** x["log10_dt"], c.t_stop / MIN_STEPS)  # 규칙을 항상 만족시키기
    if R["dt_max"] is not None and ("pb_contact_erp" in x):
        dt = min(dt, R["dt_max"])
    if R["tc_frac"] is not None and "mj_log10_timeconst" in x:
        x["mj_log10_timeconst"] = min(x["mj_log10_timeconst"], np.log10(R["tc_frac"] * c.t_stop))
    if R["tc_min_dt"] is not None and "mj_log10_timeconst" in x:
        x["mj_log10_timeconst"] = max(x["mj_log10_timeconst"], np.log10(R["tc_min_dt"] * dt))
    if R["erp_max"] is not None and "pb_contact_erp" in x:
        x["pb_contact_erp"] = min(x["pb_contact_erp"], R["erp_max"])
    return c, dt, x


def run_mj(c, dt, x):
    return sb.run_mujoco(c, dt=dt, timeconst=10 ** x.get("mj_log10_timeconst", np.log10(0.02)),
                         impratio=x.get("mj_impratio", 1.0),
                         cone="pyramidal" if x.get("mj_cone", 0.0) < 0.5 else "elliptic")["d_sim"]


def run_pb(c, dt, x):
    return sb.run_pybullet(c, dt=dt, iterations=max(1, int(round(10 ** x.get("pb_log10_iterations", np.log10(50))))),
                           contact_erp=x.get("pb_contact_erp", 0.2))["d_sim"]


def evaluate(args):
    target, z, rules = args
    names = USES[target]
    c, dt, x = decode(np.asarray(z), names, rules)
    if target == "mujoco":
        val = abs(run_mj(c, dt, x) / c.d_true - 1) * 100
    elif target == "pybullet":
        val = abs(run_pb(c, dt, x) / c.d_true - 1) * 100
    else:
        val = abs(run_mj(c, dt, x) - run_pb(c, dt, x)) / c.d_true * 100
    return val, dt, c.t_stop / dt


def search(target, method, budget, seed, pool):
    names = USES[target]
    d = len(names)
    rng = np.random.default_rng(seed)
    hist = []

    def record(Z, res):
        for z, (v, dt, steps) in zip(Z, res):
            hist.append(dict(target=target, method=method, seed=seed, eval=len(hist) + 1, value=v,
                             dt=dt, steps=steps, **{n: float(np.clip(zi, 0, 1)) for n, zi in zip(names, z)}))

    if method == "random":
        Z = rng.random((budget, d))
        record(Z, pool.map(evaluate, [(target, z, RULE) for z in Z], chunksize=8))
    else:
        es = cma.CMAEvolutionStrategy(rng.random(d), 0.3,
                                      {"bounds": [0, 1], "seed": seed + 1, "verbose": -9, "popsize": 10})
        while len(hist) < budget:
            Z = es.ask()
            Z = Z[: budget - len(hist)]
            res = pool.map(evaluate, [(target, z, RULE) for z in Z])
            record(Z, res)
            if len(Z) == es.popsize:
                es.tell(Z, [-r[0] for r in res])  # 최대화 → 부호 반전
    return hist


def robustness(target, z_best, pool, n=40, scale=0.02, seed=0):
    """찾은 최악 조건이 칼날 위의 한 점인지, 넓은 영역인지: 주변을 ±2% 흔들어 재평가."""
    rng = np.random.default_rng(seed)
    Z = np.clip(np.asarray(z_best) + rng.normal(0, scale, (n, len(z_best))), 0, 1)
    vals = [r[0] for r in pool.map(evaluate, [(target, z, RULE) for z in Z])]
    return {"neighbors": n, "scale": scale, "median": float(np.median(vals)),
            "share_over_1pct": float(np.mean(np.array(vals) > 1))}


RULE = "v1"


def main(budget=400, seeds=5, rules="v1"):
    global RULE
    RULE = rules
    OUT.mkdir(exist_ok=True)
    tag = TAGS[rules]
    allh, best = [], {}
    t0 = time.time()
    with Pool(2) as pool:
        for target in ["mujoco", "pybullet", "disagree"]:
            for method in ["cmaes", "random"]:
                for s in range(seeds):
                    allh += search(target, method, budget, s, pool)
                print(f"  {target}/{method} 완료 ({time.time() - t0:.0f}s)", flush=True)
            df = pd.DataFrame([h for h in allh if h["target"] == target])
            top = df.loc[df.value.idxmax()]
            names = USES[target]
            c, dt, x = decode(np.array([top[n] for n in names]), names, rules)
            best[target] = {
                "value_pct": float(top.value), "found_by": top.method, "seed": int(top.seed),
                "params": {k: float(v) for k, v in x.items()}, "dt": float(dt),
                "steps_to_stop": float(c.t_stop / dt), "d_true": c.d_true,
                "robustness": robustness(target, [top[n] for n in names], pool),
                "share_over_1pct_cmaes": float((df[df.method == "cmaes"].value > 1).mean()),
                "share_over_1pct_random": float((df[df.method == "random"].value > 1).mean()),
            }
    pd.DataFrame(allh).to_csv(OUT / f"fuzz_history{tag}.csv", index=False)
    (OUT / f"fuzz_best{tag}.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=1))
    print(f"총 {len(allh)}회 평가 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = sys.argv[1:]
    rules = "v1"
    if "--rules" in a:
        i = a.index("--rules"); rules = a[i + 1]; a = a[:i] + a[i + 2:]
    main(int(a[0]) if a else 400, int(a[1]) if len(a) > 1 else 5, rules)
