"""1단계 분석: 핵심 수치 요약 + 그래프 3장.

python analyze.py  →  results/summary.json, figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from simdiff import engines
from simdiff.scenes import pendulum as pend

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- 스타일 (라이트 모드 정적 이미지)
for fp in font_manager.findSystemFonts():
    if "NotoSansCJK-Regular" in fp:
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 22,
})

cfg = yaml.safe_load((ROOT / "experiments" / "pendulum_sweep.yaml").read_text(encoding="utf-8"))
df = pd.read_csv(RES / "pendulum_runs.csv")
D = cfg["duration"]


def label_end(ax, x, y, text, color_ink=INK, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=color_ink)


# ======================================================== 1. 엔진 불일치 vs 실제 오차
rows = []
for damping in cfg["dampings"]:
    params = pend.PendulumParams(**cfg["params"], damping=damping)
    for th in cfg["theta0_deg"]:
        for dt in cfg["timesteps"]:
            n = int(round(D / dt)); t = np.arange(n + 1) * dt
            ref = pend.reference(params, np.deg2rad(th), t)
            mj = engines.run_mujoco(params, np.deg2rad(th), D, dt, "Euler")
            pb = engines.run_pybullet(params, np.deg2rad(th), D, dt, "clean")
            rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
            rows.append(dict(damping=damping, theta0_deg=th, dt=dt,
                             disagreement=rmse(mj[:, 1], pb[:, 1]),
                             err_mujoco=rmse(mj[:, 1], ref[:, 0]),
                             err_pybullet=rmse(pb[:, 1], ref[:, 0])))
dis = pd.DataFrame(rows)
dis["err_worst"] = dis[["err_mujoco", "err_pybullet"]].max(axis=1)
dis.to_csv(RES / "disagreement_vs_error.csv", index=False)

rho_all, p_all = spearmanr(dis.disagreement, dis.err_worst)
damped = dis[dis.damping > 0]
rho_d, p_d = spearmanr(damped.disagreement, damped.err_worst)
undamped = dis[dis.damping == 0]

# ======================================================== 2. 핵심 수치
q = lambda **k: df.loc[np.logical_and.reduce([df[c] == v for c, v in k.items()])]
base = dict(damping=0.0, theta0_deg=90, dt=0.001)
summary = {
    "runs": int(len(df)),
    "all_deterministic": bool(df.deterministic.all()),
    "euler_first_order_slope": float(np.polyfit(
        np.log(q(damping=0.0, theta0_deg=90, config="MuJoCo Euler (기본값)").dt),
        np.log(q(damping=0.0, theta0_deg=90, config="MuJoCo Euler (기본값)").rmse_rad), 1)[0]),
    "rk4_slope": float(np.polyfit(
        np.log(q(damping=0.0, theta0_deg=90, config="MuJoCo RK4").dt),
        np.log(q(damping=0.0, theta0_deg=90, config="MuJoCo RK4").rmse_rad), 1)[0]),
    "at_dt_1ms_theta90_undamped": {
        r.config: {"rmse_rad": r.rmse_rad, "energy_err_final": r.energy_err_final}
        for r in q(**base).itertuples()},
    "mujoco_euler_equals_implicit_all_conditions": bool(np.allclose(
        q(config="MuJoCo Euler (기본값)").rmse_rad.values,
        q(config="MuJoCo implicit").rmse_rad.values, rtol=1e-9, atol=0)),
    "undamped_max_disagreement_rad": float(undamped.disagreement.max()),
    "undamped_min_error_rad": float(undamped.err_worst.min()),
    "undamped_max_error_rad": float(undamped.err_worst.max()),
    "damped_pybullet_over_mujoco_error_ratio_mean": float((damped.err_pybullet / damped.err_mujoco).mean()),
    "spearman_disagreement_vs_error_all": {"rho": float(rho_all), "p": float(p_all), "n": int(len(dis))},
    "spearman_disagreement_vs_error_damped_only": {"rho": float(rho_d), "p": float(p_d), "n": int(len(damped))},
    "pybullet_effective_inertia": json.loads((RES / "env.json").read_text(encoding="utf-8"))["pybullet_effective_inertia"],
}
(RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

# ======================================================== 3. 그림 1: 타임스텝 vs 오차
fig, ax = plt.subplots(figsize=(7.2, 4.4))
series = [
    ("MuJoCo Euler (기본값)", S1, "반암시적 오일러\n(MuJoCo 기본값 = PyBullet clean)"),
    ("MuJoCo RK4", S3, "MuJoCo RK4"),
    ("PyBullet 기본값", S2, "PyBullet 기본값"),
]
for cfgname, col, lab in series:
    x = q(damping=0.0, theta0_deg=90, config=cfgname).sort_values("dt")
    ax.loglog(x.dt * 1000, x.rmse_rad, color=col, marker="o", markersize=5,
              markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab.replace("\n", " "))
    label_end(ax, x.dt.iloc[-1] * 1000, x.rmse_rad.iloc[-1], lab)
ax.set_xlabel("타임스텝 (ms)")
ax.set_ylabel("각도 RMSE vs 기준 해 (rad)")
ax.set_title("타임스텝보다 기본 설정이 더 큰 오차를 만든다")
ax.text(0, 1.02, "진자 90°에서 출발, 감쇠 없음, 10초", transform=ax.transAxes, color=INK2, fontsize=9)
ax.set_xlim(0.4, 80)
ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig1_timestep_vs_error.png", dpi=180)
plt.close(fig)

# ======================================================== 4. 그림 2: 에너지 보존
z = np.load(RES / "pendulum_traj.npz")
params0 = pend.PendulumParams(**cfg["params"], damping=0.0)
E0 = pend.energy(params0, z["reference"][0, 1], z["reference"][0, 2])
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ref = z["reference"]
ax.plot(ref[:, 0], pend.energy(params0, ref[:, 1], ref[:, 2]) / E0, color=INK2, lw=1.5, ls="--",
        label="기준 해", zorder=5)
for key, col, lab in [("MuJoCo Euler (기본값)", S1, "MuJoCo 기본값"),
                      ("PyBullet 감쇠만 수정", S3, "PyBullet 감쇠만 수정"),
                      ("PyBullet 기본값", S2, "PyBullet 기본값")]:
    a = z[key]
    e = pend.energy(params0, a[:, 1], a[:, 2]) / E0
    ax.plot(a[:, 0], e, color=col, label=lab)
    label_end(ax, a[-1, 0], e[-1], f"{lab} ({e[-1]:.2f})", dy={"MuJoCo 기본값": 9, "PyBullet 감쇠만 수정": -9}.get(lab, 0))
ax.set_xlabel("시간 (s)")
ax.set_ylabel("에너지 / 초기 에너지")
ax.set_title("PyBullet 기본값은 10초 만에 에너지 68%를 잃는다")
ax.text(0, 1.02, "감쇠 없는 진자라 이상적으로는 1.0 유지, dt = 10 ms", transform=ax.transAxes, color=INK2, fontsize=9)
ax.set_xlim(0, 14.5); ax.set_ylim(0, 1.15)
ax.legend(loc="lower left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig2_energy.png", dpi=180)
plt.close(fig)

# ======================================================== 5. 그림 3: 불일치 vs 오차 (소형 다중)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
for ax, damping, title in [(axes[0], 0.0, "감쇠 없음"), (axes[1], 0.05, "감쇠 있음 (0.05)")]:
    x = dis[(dis.damping == damping) & (dis.theta0_deg == 90)].sort_values("dt")
    ax.loglog(x.dt * 1000, x.err_worst, color=S1, marker="o", markersize=5,
              markeredgecolor=SURFACE, markeredgewidth=1.5, label="실제 오차 (엔진 vs 기준 해)")
    ax.loglog(x.dt * 1000, np.maximum(x.disagreement, 1e-16), color=S2, marker="s", markersize=5,
              markeredgecolor=SURFACE, markeredgewidth=1.5, label="엔진 간 불일치 (MuJoCo vs PyBullet)")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("타임스텝 (ms)")
axes[0].set_ylabel("각도 RMSE (rad)")
axes[0].annotate("두 엔진이 소수점 14자리까지 일치\n하지만 둘 다 똑같이 틀림",
                 (1, 1e-14), xytext=(1.2, 1e-9), fontsize=8.5, color=INK2,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
axes[1].legend(loc="center right", frameon=False, fontsize=8.5)
fig.suptitle("엔진끼리 합의해도 정답이라는 보장은 없다", x=0.02, ha="left", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig3_disagreement_vs_error.png", dpi=180)
plt.close(fig)

print(json.dumps({k: v for k, v in summary.items() if k != "at_dt_1ms_theta90_undamped"},
                 ensure_ascii=False, indent=1))
