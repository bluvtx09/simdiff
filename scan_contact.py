"""SimDiff 3단계-A2: 퍼징이 찾은 반례의 원인 규명.

퍼징의 MuJoCo 최악 조건(오차 26%)을 들여다보니, 상자가 튀어 오른 뒤 앞으로 굴러 넘어졌다.
강체 쿨롱 마찰 이론에서 상자는 '넘어짐 여유' tau = mu * h / b 가 1 이상일 때만 넘어진다
(h: 높이의 절반, b: 바닥 폭의 절반). 정육면체에서는 tau = mu 이므로 mu < 1 이면 안 넘어져야 한다.

가설: 오차의 두 번째 원인은 tau 이고, 상자 모양을 바꿔도 tau 로 정리된다.
검증: 상자 높이 3종 x mu 14단계 x v0 3개 x dt 3단계 x 엔진 설정 3종.

python scan_contact.py  →  results/contact_scan.csv
"""
import csv
import itertools
import time
from pathlib import Path

import numpy as np

from simdiff.scenes import sliding_box as sb

OUT = Path(__file__).parent / "results"
ASPECTS = [0.5, 1.0, 1.5]                     # h / b
MUS = np.round(np.arange(0.30, 0.951, 0.05), 2)
V0S = [1.0, 2.0, 3.0]
DT_FACTORS = [1.0, 0.5, 0.25]                 # 규칙을 만족하는 dt에 곱하는 값 (더 작게)
SETTINGS = {
    "MuJoCo 기본값": lambda c, dt, hz: sb.run_mujoco(c, dt=dt, half_z=hz),
    "MuJoCo 딱딱한 접촉": lambda c, dt, hz: sb.run_mujoco(c, dt=dt, half_z=hz, timeconst=6 * dt,
                                                    cone="elliptic", impratio=5.0),
    "PyBullet": lambda c, dt, hz: sb.run_pybullet(c, dt=dt, half_z=hz),
}


def main():
    OUT.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    for aspect, mu, v0, f in itertools.product(ASPECTS, MUS, V0S, DT_FACTORS):
        c = sb.BoxCase(float(mu), v0)
        hz = sb.HALF * aspect
        dt = min(0.002, c.t_stop / 300) * f
        for name, fn in SETTINGS.items():
            r = fn(c, dt, hz)
            rows.append(dict(setting=name, aspect=aspect, mu=float(mu), v0=v0, dt=dt,
                             steps=c.t_stop / dt, tau=float(mu) * aspect,
                             err_pct=(r["d_sim"] / c.d_true - 1) * 100,
                             tilt_deg=r["tilt_deg"], tumbled=r["tilt_deg"] > 5))
    with open(OUT / "contact_scan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)}회 완료 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
