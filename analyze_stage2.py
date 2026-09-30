"""2단계 분석: 2링크 팔, 미끄러지는 상자, Sobol, 뒤집힘 지수.

python analyze_stage2.py  →  results/summary_stage2.json, figures/fig4~fig7
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_style import INK, INK2, SURFACE, S1, S2, S3, S4, apply_style

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
apply_style()
TICKS = ([0.25, 0.5, 1, 2, 5], ["0.25", "0.5", "1", "2", "5"])
summary = {}

# ============================================================== A. 2링크 팔
runs = pd.read_csv(RES / "arm_runs.csv")
ref = pd.read_csv(RES / "arm_ref_check.csv")
chaotic = ref[np.isfinite(ref.ref_valid_until)].ic.tolist()
runs = runs.merge(ref[["ic", "ref_valid_until", "butterfly_horizon"]], on="ic")
# 기준 해가 믿을 수 있는 시간을 넘는 값은 '그 이상'으로만 알 수 있음 → 하한값으로 잘라서 사용
runs["h_capped"] = np.minimum(runs.horizon_vs_ref, runs.ref_valid_until)
runs["censored"] = runs.horizon_vs_ref > runs.ref_valid_until
ch = runs[runs.ic.isin(chaotic)]
med = ch.pivot_table(index="dt", columns="config", values="h_capped", aggfunc="median")
eng_med = ch[ch.config.str.startswith("엔진 간")].groupby("dt").horizon_vs_ref.median()
bf = ref[ref.ic.isin(chaotic)].butterfly_horizon.median()
euler = med["MuJoCo Euler"]
slope = np.polyfit(np.log10(euler.index), euler.values, 1)[0]
same = ch[ch.config.isin(["MuJoCo Euler", "PyBullet clean"])].pivot_table(
    index=["ic", "dt"], columns="config", values="horizon_vs_ref")
summary["arm"] = {
    "n_chaotic_ics": len(chaotic),
    "median_horizon_s": {c: {str(k): round(v, 2) for k, v in med[c].items()} for c in med.columns
                         if not c.startswith("엔진 간")},
    "median_engine_vs_engine_horizon_s": {str(k): (round(v, 2) if np.isfinite(v) else "20초 내내 일치")
                                          for k, v in eng_med.items()},
    "median_butterfly_horizon_s": round(float(bf), 2),
    "euler_horizon_gain_per_10x_smaller_dt_s": round(float(-slope), 2),
    "rk4_censored_fraction": round(float(ch[ch.config == "MuJoCo RK4"].censored.mean()), 3),
    "mujoco_euler_eq_pybullet_horizon_all": bool(np.allclose(same["MuJoCo Euler"], same["PyBullet clean"])),
    "nonchaotic_ics": [i for i in ref.ic if i not in chaotic],
}

fig, ax = plt.subplots(figsize=(7.4, 5.0))
x = np.array(med.index) * 1000
ax.axhspan(bf * 0.97, bf * 1.03, color=INK2, alpha=0.10, lw=0)
ax.axhline(bf, color=INK2, lw=1.2, ls="--")
ax.annotate(f"나비효과 한계 (초기각 1e-9 rad 차이): {bf:.1f}초", (0.25, bf), xytext=(0, 6),
            textcoords="offset points", fontsize=8.5, color=INK2)
lines = [(eng_med.replace(np.inf, 20.0).values, S2, "s", "엔진끼리 비교 (MuJoCo vs PyBullet)"),
         (med["MuJoCo RK4"].values, S3, "o", "MuJoCo RK4 (하한값)"),
         (euler.values, S1, "o", "반암시적 오일러 (MuJoCo 기본값 = PyBullet)")]
for y, col, mk, lab in lines:
    ax.semilogx(x, y, color=col, marker=mk, markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=lab)
    ax.annotate(lab, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK)
ax.set_xticks(*TICKS)
ax.set_xlim(0.2, 60)
ax.set_ylim(0, 21)
ax.set_xlabel("타임스텝 (ms)")
ax.set_ylabel("예측 가능 시간 (s)\n관절각 오차 0.1 rad 도달까지")
ax.set_title("엔진끼리는 17초간 일치하지만, 정답과는 1초 남짓에 갈라진다")
ax.text(0, 1.02, f"2링크 팔 혼돈 영역 초기자세 {len(chaotic)}개의 중앙값", transform=ax.transAxes,
        color=INK2, fontsize=9)
ax.legend(loc="upper center", frameon=False, fontsize=8.5, bbox_to_anchor=(0.45, -0.18), ncol=2)
fig.tight_layout()
fig.savefig(FIG / "fig4_arm_horizon.png", dpi=180)
plt.close(fig)

# ============================================================== B. 상자: 함정
traps = pd.read_csv(RES / "box_traps.csv")
summary["box_traps"] = traps.groupby("setting").err_pct.agg(["mean", "min", "max"]).round(2).to_dict("index")
summary["box_never_tipped"] = bool((traps.z_final.sub(0.05).abs() < 0.001).all())

# ============================================================== C. Sobol + 무차원 수
sob = {}
for eng in ["mujoco", "pybullet"]:
    s = pd.read_csv(RES / f"sobol_{eng}.csv")
    z = np.load(RES / f"sobol_samples_{eng}.npz")
    X, Y = z["X"], np.abs(z["Y"])
    dt, mu, v0 = 10 ** X[:, 0], X[:, 1], X[:, 2]
    steps = v0 / (mu * 9.81 * dt)
    # 무차원 수 하나(정지까지의 스텝 수)가 log|오차| 분산을 얼마나 설명하나
    ly, lx = np.log10(Y + 1e-4), np.log10(steps)
    r2 = np.corrcoef(lx, ly)[0, 1] ** 2
    bins = [0, 20, 50, 100, 300, np.inf]
    b = pd.cut(steps, bins)
    by = pd.Series(Y).groupby(b, observed=True)
    sob[eng] = dict(table=s, steps=steps, Y=Y)
    summary[f"sobol_{eng}"] = {
        "n_runs": int(len(Y)),
        "mean_abs_err_pct": round(float(Y.mean()), 3), "max_abs_err_pct": round(float(Y.max()), 2),
        "share_abs_err_over_1pct": round(float((Y > 1).mean()), 3),
        "indices": s[["label", "kind", "S1", "ST", "ST_conf"]].round(3).to_dict("records"),
        "sum_S1": round(float(s.S1.clip(lower=0).sum()), 3),
        "r2_log_steps_vs_log_err": round(float(r2), 3),
        "mean_abs_err_by_steps_to_stop": {str(k): round(float(v), 3) for k, v in by.mean().items()},
        "share_under_1pct_by_steps_to_stop": {str(k): round(float(v), 3) for k, v in
                                              by.apply(lambda s: (s < 1).mean()).items()},
    }

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
for ax, eng, title in [(axes[0], "mujoco", "MuJoCo"), (axes[1], "pybullet", "PyBullet")]:
    t = sob[eng]["table"].sort_values("ST")
    yy = np.arange(len(t))
    ax.barh(yy + 0.18, t.ST.clip(lower=0), height=0.34, color=S1, label="전체 영향 (ST, 상호작용 포함)")
    ax.barh(yy - 0.18, t.S1.clip(lower=0), height=0.34, color=S2, label="단독 영향 (S1)")
    ax.errorbar(t.ST.clip(lower=0), yy + 0.18, xerr=t.ST_conf, fmt="none", ecolor=INK2, lw=0.8, capsize=2)
    ax.set_yticks(yy, [f"{l}  ·{k}" for l, k in zip(t.label, t.kind)])
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Sobol 지수")
    ax.grid(axis="y", visible=False)
axes[0].tick_params(axis="y", labelsize=8.5)
axes[1].tick_params(axis="y", labelleft=True, labelsize=8.5)
axes[1].legend(loc="lower right", frameon=False, fontsize=8.5)
fig.suptitle("정지 거리 오차는 한 가지 설정이 아니라 설정 간 상호작용에서 나온다",
             x=0.02, ha="left", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig5_sobol.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.4))
for eng, col, lab in [("mujoco", S1, "MuJoCo"), ("pybullet", S2, "PyBullet")]:
    st, Y = sob[eng]["steps"], sob[eng]["Y"]
    ax.scatter(st, np.maximum(Y, 1e-3), s=3, color=col, alpha=0.12, lw=0, rasterized=True)
    edges = np.logspace(np.log10(st.min()), np.log10(st.max()), 18)
    idx = np.digitize(st, edges)
    cx, cy = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() > 50:
            cx.append(np.sqrt(edges[i - 1] * edges[i])); cy.append(np.percentile(Y[m], 90))
    ax.loglog(cx, cy, color=col, lw=2, label=f"{lab} (상위 10% 오차선)")
ax.axvline(300, color=INK2, lw=1, ls="--")
ax.annotate("300스텝", (300, 30), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK2)
ax.set_xlabel("정지까지 걸리는 스텝 수 = v0 / (μ g Δt)")
ax.set_ylabel("정지 거리 |오차| (%)")
ax.set_title("오차를 가장 잘 설명하는 건 '멈출 때까지 몇 스텝인가'다")
ax.text(0, 1.02, "Sobol 표본 전체 (MuJoCo 32,768회, PyBullet 28,672회), 점 하나 = 실행 1회",
        transform=ax.transAxes, color=INK2, fontsize=9)
ax.set_ylim(1e-3, 150)
ax.legend(loc="lower left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig6_steps_to_stop.png", dpi=180)
plt.close(fig)

# ============================================================== D. 뒤집힘 지수
TIE = 0.05  # 두 엔진 |오차| 차이가 0.05%p 이내면 무승부
g = pd.read_csv(RES / "box_grid.csv")
g["abs"] = g.err_pct.abs()
mj = g[g.engine == "mujoco"].rename(columns={"setting": "mj", "abs": "mj_abs"})[["mu", "v0", "dt", "mj", "mj_abs"]]
pb = g[g.engine == "pybullet"].rename(columns={"setting": "pb", "abs": "pb_abs"})[["mu", "v0", "dt", "pb", "pb_abs"]]
m = mj.merge(pb, on=["mu", "v0", "dt"])
m["win"] = np.where(m.mj_abs + TIE < m.pb_abs, "MuJoCo",
                    np.where(m.pb_abs + TIE < m.mj_abs, "PyBullet", "무승부"))
base = m[(m.dt == 0.002) & m.mj.str.contains("기본") & m.pb.str.contains("기본")][["mu", "v0", "win"]]
mm = m.merge(base.rename(columns={"win": "win_base"}), on=["mu", "v0"])
dec = mm[(mm.win != "무승부") & (mm.win_base != "무승부")]
summary["flip_index"] = {
    "definition": "같은 물리 조건에서 기본 설정일 때의 승자와 반대 엔진이 이기는 비율 (둘 다 승부가 난 경우만)",
    "tie_band_pctpt": TIE,
    "value": round(float((dec.win != dec.win_base).mean()), 3),
    "n_decisive_pairs": int(len(dec)),
    "any_change_rate": round(float((mm.win != mm.win_base).mean()), 3),
    "n_pairs": int(len(mm)),
    "winner_counts_by_dt": m.pivot_table(index="dt", columns="win", values="mu", aggfunc="count")
                            .fillna(0).astype(int).to_dict("index"),
}
wc = m.pivot_table(index="dt", columns="win", values="mu", aggfunc="count").fillna(0)
wc = wc[["MuJoCo", "무승부", "PyBullet"]]
wc = wc.div(wc.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(7.4, 3.6))
yy = np.arange(len(wc))[::-1]
left = np.zeros(len(wc))
for col, color in [("MuJoCo", S1), ("무승부", "#c9c8c3"), ("PyBullet", S2)]:
    ax.barh(yy, wc[col], left=left, color=color, height=0.62, edgecolor=SURFACE, lw=2, label=col)
    for yi, l, w in zip(yy, left, wc[col]):
        if w >= 12:
            ax.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=8.5,
                    color=INK if color == "#c9c8c3" else "#ffffff")
    left += wc[col].values
ax.set_yticks(yy, [f"{d * 1000:g} ms" for d in wc.index])
ax.set_xlim(0, 100)
ax.set_xlabel("비교 비율 (%)")
ax.grid(axis="y", visible=False)
ax.set_title(f"'어느 엔진이 더 정확한가'는 설정에 따라 뒤집힌다 (뒤집힘 지수 {summary['flip_index']['value']:.2f})")
ax.text(0, 1.03, "타임스텝별로 더 정확했던 엔진, 물리 조건 9개 x 엔진 설정 4조합", transform=ax.transAxes,
        color=INK2, fontsize=9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig7_flip.png", dpi=180)
plt.close(fig)

(RES / "summary_stage2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                         encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1, default=float)[:6000])
