"""SimDiff 3단계-B: 불일치 지도 — 어떤 '불일치'가 실제 오차를 예측하나?

정답을 모르는 실제 상황에서 쓸 수 있는 오차 신호 두 가지를 비교한다.
  엔진 간 불일치 : 같은 조건을 MuJoCo와 PyBullet으로 돌린 결과 차이
  자기 불일치    : 같은 엔진을 dt와 dt/2로 돌린 결과 차이 (타임스텝 반감)
평가: 실제 오차(해석해·기준 해 대비)와의 순위상관(Spearman), 1% 넘는 실패를 골라내는 능력(AUC),
     자기 불일치 x 2 가 실제 오차를 얼마나 잘 맞히는지(1차 방법의 리처드슨 추정).

python disagreement_map.py  →  results/dmap_box.csv, results/dmap_arm.csv, results/dmap_summary.json
"""
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from simdiff import engines
from simdiff.scenes import sliding_box as sb
from simdiff.scenes import two_link_arm as arm

ROOT = Path(__file__).parent
OUT = ROOT / "results"
N_BOX = 1500


def box_sample(i):
    rng = np.random.default_rng(1000 + i)
    mu, v0 = rng.uniform(0.1, 0.8), rng.uniform(0.5, 3.0)
    dt = 10 ** rng.uniform(-3.3, -2.0)
    tc = 10 ** rng.uniform(-2.3, -1.3)
    imp = rng.uniform(1, 10)
    cone = "pyramidal" if rng.random() < 0.5 else "elliptic"
    erp = rng.uniform(0.05, 0.8)
    c = sb.BoxCase(mu, v0)
    mj = lambda h: sb.run_mujoco(c, dt=h, timeconst=tc, impratio=imp, cone=cone)["d_sim"]
    pb = lambda h: sb.run_pybullet(c, dt=h, contact_erp=erp)["d_sim"]
    m1, m2, p1, p2 = mj(dt), mj(dt / 2), pb(dt), pb(dt / 2)
    pct = lambda a: abs(a) / c.d_true * 100
    return dict(mu=mu, v0=v0, dt=dt, timeconst=tc, impratio=imp, cone=cone, contact_erp=erp,
                steps=c.t_stop / dt,
                err_mj=pct(m1 - c.d_true), err_pb=pct(p1 - c.d_true),
                cross=pct(m1 - p1), self_mj=pct(m1 - m2), self_pb=pct(p1 - p2))


def arm_sample(args):
    ic_name, q0, dt = args
    p = arm.ArmParams()
    T = 2.0
    ref = arm.reference(p, q0, np.array([0.0, 0.5, 1.0, 2.0]))
    mj1 = engines.run_mujoco_chain(arm.mjcf(p, dt, "Euler"), q0, T)
    mj2 = engines.run_mujoco_chain(arm.mjcf(p, dt / 2, "Euler"), q0, T)
    pb1 = engines.run_pybullet_chain(arm.urdf(p), q0, T, dt)
    rows = []
    for k, t in enumerate([0.5, 1.0, 2.0], start=1):
        i1, i2 = int(round(t / dt)), int(round(t / (dt / 2)))
        a, b, c = mj1[i1, 1:3], mj2[i2, 1:3], pb1[i1, 1:3]
        rows.append(dict(ic=ic_name, dt=dt, t=t,
                         err_mj=float(np.max(np.abs(a - ref[k, :2]))),
                         cross=float(np.max(np.abs(a - c))),
                         self_mj=float(np.max(np.abs(a - b)))))
    return rows


def score(df, err, pred, fail_thr):
    """fail_thr: 실패로 보는 실제 오차 기준 (상자 1%, 2링크 팔 0.1 rad)."""
    y = df[err].to_numpy()
    x = df[pred].to_numpy()
    rho = spearmanr(x, y).correlation
    fail = y > fail_thr
    auc = roc_auc_score(fail, x) if 0 < fail.mean() < 1 else float("nan")
    return float(rho), float(auc), float(fail.mean())


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    with Pool(2) as pool:
        box = pd.DataFrame(pool.map(box_sample, range(N_BOX), chunksize=10))
    box.to_csv(OUT / "dmap_box.csv", index=False)
    print(f"상자 {len(box)}조건 ({time.time() - t0:.0f}s)", flush=True)

    ref = pd.read_csv(OUT / "arm_ref_check.csv")
    ics = [(r.ic, np.deg2rad([r.q1_deg, r.q2_deg])) for r in ref.itertuples()]
    jobs = [(n, q, dt) for n, q in ics for dt in [0.0005, 0.001, 0.002, 0.005]]
    with Pool(2) as pool:
        armdf = pd.DataFrame([r for rows in pool.map(arm_sample, jobs) for r in rows])
    armdf.to_csv(OUT / "dmap_arm.csv", index=False)
    print(f"2링크 팔 {len(armdf)}행 ({time.time() - t0:.0f}s)", flush=True)

    summ = {"box": {}, "arm": {}}
    for eng, err, selfc in [("MuJoCo", "err_mj", "self_mj"), ("PyBullet", "err_pb", "self_pb")]:
        for pred, lab in [("cross", "엔진 간 불일치"), (selfc, "자기 불일치 (dt vs dt/2)")]:
            rho, auc, fr = score(box, err, pred, 1.0)
            summ["box"][f"{eng} 오차 ← {lab}"] = {"spearman": round(rho, 3), "auc_fail_1pct": round(auc, 3),
                                                   "fail_rate": round(fr, 3)}
        ratio = (box[err] / (2 * box[selfc])).replace([np.inf, -np.inf], np.nan).dropna()
        summ["box"][f"{eng} 실제오차 / (2 x 자기불일치) 중앙값"] = round(float(ratio.median()), 3)
        summ["box"][f"{eng} 2x자기불일치가 실제오차의 0.5~2배 안에 드는 비율"] = round(
            float(((ratio > 0.5) & (ratio < 2)).mean()), 3)
    for pred, lab in [("cross", "엔진 간 불일치"), ("self_mj", "자기 불일치 (dt vs dt/2)")]:
        rho, auc, fr = score(armdf, "err_mj", pred, 0.1)
        summ["arm"][f"MuJoCo 오차 ← {lab}"] = {"spearman": round(rho, 3), "auc_fail_0.1rad": round(auc, 3),
                                              "fail_rate": round(fr, 3)}
    ratio = armdf.err_mj / (2 * armdf.self_mj)
    summ["arm"]["실제오차 / (2 x 자기불일치) 중앙값"] = round(float(ratio.median()), 3)
    summ["arm"]["엔진 간 불일치 중앙값 (rad)"] = float(armdf.cross.median())
    summ["arm"]["실제 오차 중앙값 (rad)"] = float(armdf.err_mj.median())
    (OUT / "dmap_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
