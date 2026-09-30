"""3단계 분석: 퍼징, 넘어짐 여유, PyBullet 입력 민감도, 불일치 지도.

python analyze_stage3.py  →  results/summary_stage3.json, figures/fig8~fig11
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_style import INK, INK2, SURFACE, S1, S2, S3, apply_style

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"
apply_style()
summary = {}

# ============================================================== A. 규칙 버전별 퍼징 최악값
VERS = [("v1", "", "v1\n300스텝"), ("v1_mu", "_refined", "v1 + 넘어짐\n여유 ≤ 0.6"),
        ("v2", "_v2", "+ 접촉 시간·\nPyBullet dt·ERP"), ("v3", "_v3", "+ timeconst\n≥ 10 dt")]
TARGETS = [("mujoco", "MuJoCo 오차", S1), ("pybullet", "PyBullet 오차", S2), ("disagree", "두 엔진 차이", S3)]
worst = {}
for v, tag, _ in VERS:
    b = json.loads((RES / f"fuzz_best{tag}.json").read_text(encoding="utf-8"))
    h = pd.read_csv(RES / f"fuzz_history{tag}.csv")
    worst[v] = {}
    for t, _, _ in TARGETS:
        hh = h[h.target == t]
        worst[v][t] = {
            "max_pct": round(b[t]["value_pct"], 2), "found_by": b[t]["found_by"],
            "neighbors_median_pct": round(b[t]["robustness"]["median"], 2),
            "share_over_1pct_cmaes": round(b[t]["share_over_1pct_cmaes"], 4),
            "share_over_1pct_random": round(b[t]["share_over_1pct_random"], 4),
            "n_evals": int(len(hh)),
        }
summary["fuzz_worst_by_rule"] = worst

fig, ax = plt.subplots(figsize=(7.6, 4.6))
xs = np.arange(len(VERS))
for t, lab, col in TARGETS:
    ys = [worst[v][t]["max_pct"] for v, _, _ in VERS]
    ax.plot(xs, ys, color=col, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=f"{lab} ({ys[0]:.1f}% → {ys[-1]:.1f}%)")
ax.axhline(1, color=INK2, lw=1, ls="--")
ax.annotate("오차 1%", (0, 1), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK2)
ax.set_xticks(xs, [l for _, _, l in VERS], fontsize=8.5)
ax.set_xlim(-0.3, 3.3)
ax.set_ylabel("퍼징이 찾은 최악 |오차| (%)")
ax.set_title("퍼징 → 원인 규명 → 규칙 보완을 3번 반복하자 최악 오차가 26% → 2%")
ax.text(0, 1.02, "규칙 버전마다 CMA-ES·무작위 탐색 각 8회 x 600회 = 목표당 9,600회 평가",
        transform=ax.transAxes, color=INK2, fontsize=9)
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig8_fuzz_rules.png", dpi=180)
plt.close(fig)

# ============================================================== B. 탐색 효율 (v1)
h = pd.read_csv(RES / "fuzz_history.csv")
h["best"] = h.groupby(["target", "method", "seed"]).value.cummax()
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8))
eff = {}
for ax, (t, lab, _) in zip(axes, TARGETS):
    for m, col, mlab in [("cmaes", S1, "CMA-ES"), ("random", S2, "무작위")]:
        g = h[(h.target == t) & (h.method == m)].pivot_table(index="eval", columns="seed", values="best")
        med = g.median(axis=1)
        ax.fill_between(g.index, g.quantile(0.25, axis=1), g.quantile(0.75, axis=1), color=col, alpha=0.15, lw=0)
        ax.plot(g.index, med, color=col, label=mlab)
        eff.setdefault(t, {})[m] = {"median_best_at_100": round(float(med.loc[100]), 2),
                                    "median_best_at_600": round(float(med.iloc[-1]), 2)}
    ax.set_title(lab, fontsize=11, pad=8)
    ax.set_xlabel("평가 횟수")
axes[0].set_ylabel("지금까지 찾은 최악 |오차| (%)")
axes[0].legend(loc="lower right", frameon=False, fontsize=8.5)
fig.suptitle("CMA-ES는 MuJoCo 약점은 빨리 찾지만, PyBullet의 뾰족한 약점에는 무작위가 낫다",
             x=0.02, ha="left", fontweight="bold", fontsize=12)
fig.text(0.02, 0.86, "규칙 v1(300스텝) 아래, 시드 8개의 중앙값과 25~75% 구간", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(FIG / "fig9_fuzz_efficiency.png", dpi=180)
plt.close(fig)
summary["fuzz_efficiency_v1"] = eff

# ============================================================== C. 넘어짐 여유
cs = pd.read_csv(RES / "contact_scan.csv")
cs["abs"] = cs.err_pct.abs()
bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5]
cs["tau_bin"] = pd.cut(cs.tau, bins)
tum = cs.pivot_table(index="tau_bin", columns="setting", values="tumbled", aggfunc="mean", observed=True)
mx = cs[cs.tau < 1].pivot_table(index="tau_bin", columns="setting", values="abs", aggfunc="max", observed=True)
summary["tipping_margin"] = {
    "tumble_rate": {str(k): {c: round(float(v), 3) for c, v in r.items()} for k, r in tum.iterrows()},
    "max_abs_err_pct_tau_lt_1": {str(k): {c: round(float(v), 2) for c, v in r.items()} for k, r in mx.iterrows()},
    "n_runs": int(len(cs)),
    "mujoco_default_tumble_by_mu_aspect": cs[cs.setting == "MuJoCo 기본값"].pivot_table(
        index="mu", columns="aspect", values="tumbled", aggfunc="mean").round(2).to_dict(),
}
fig, ax = plt.subplots(figsize=(7.4, 4.4))
centers = [(b.left + b.right) / 2 for b in tum.index]
for setting, col in [("MuJoCo 기본값", S1), ("MuJoCo 딱딱한 접촉", S3), ("PyBullet", S2)]:
    ax.plot(centers, tum[setting] * 100, color=col, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=setting)
ax.axvspan(1.0, 1.5, color=INK2, alpha=0.08, lw=0)
ax.annotate("강체 이론상\n넘어지는 영역", (1.25, 45), ha="center", fontsize=8.5, color=INK2)
ax.axvline(0.6, color=INK2, lw=1, ls="--")
ax.annotate("τ = 0.6", (0.6, 95), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK2)
ax.set_xlabel("넘어짐 여유 τ = μ · (높이/2) / (바닥 폭/2)")
ax.set_ylabel("상자가 넘어진 비율 (%)")
ax.set_ylim(-3, 100)
ax.set_title("MuJoCo는 이론상 안 넘어질 상자도 τ ≥ 0.6부터 넘어뜨린다")
ax.text(0, 1.02, "상자 높이 3종 x μ 14단계 x v0 3개 x dt 3단계 (설정별 378회)", transform=ax.transAxes,
        color=INK2, fontsize=9)
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig10_tipping.png", dpi=180)
plt.close(fig)

# ============================================================== D. PyBullet 입력 민감도
summary["pybullet_input_chaos"] = json.loads((RES / "pb_chaos.json").read_text(encoding="utf-8"))

# ============================================================== E. 불일치 지도
dm = json.loads((RES / "dmap_summary.json").read_text(encoding="utf-8"))
box = pd.read_csv(RES / "dmap_box.csv")
armd = pd.read_csv(RES / "dmap_arm.csv")
for e in ["mj", "pb"]:
    r = box[f"err_{e}"] / box.cross
    dm["box"][f"{e} 실제오차 / 엔진간불일치 중앙값"] = round(float(r.median()), 3)
    dm["box"][f"{e} 엔진간불일치가 실제오차의 0.5~2배 안에 드는 비율"] = round(float(((r > 0.5) & (r < 2)).mean()), 3)
dm["arm"]["실제오차 / 엔진간불일치 중앙값"] = float((armd.err_mj / armd.cross).median())
summary["disagreement_map"] = dm

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
lims = (1e-4, 1e2)
for ax, pred, lab, col in [(axes[0], box.cross, "엔진 간 불일치 (MuJoCo vs PyBullet)", S2),
                           (axes[1], 2 * box.self_mj, "2 x 자기 불일치 (dt vs dt/2)", S1)]:
    ax.scatter(np.clip(pred, *lims), np.clip(box.err_mj, *lims), s=5, color=col, alpha=0.35, lw=0)
    ax.plot(lims, lims, color=INK2, lw=1, ls="--")
    ax.plot(lims, [l * 2 for l in lims], color=INK2, lw=0.6, ls=":")
    ax.plot(lims, [l / 2 for l in lims], color=INK2, lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel(f"{lab} (%)")
    ax.set_title(lab.split(" (")[0], fontsize=11, pad=8)
axes[0].set_ylabel("MuJoCo 실제 오차 (%)")
axes[0].annotate("점선: 예측 = 실제\n점점선: 2배 범위", (1.5e-4, 20), fontsize=8.5, color=INK2)
k0 = dm["box"]["MuJoCo 오차 ← 엔진 간 불일치"]["spearman"]
k1 = dm["box"]["MuJoCo 오차 ← 자기 불일치 (dt vs dt/2)"]["spearman"]
axes[0].text(0.97, 0.04, f"순위상관 {k0:.2f}", transform=axes[0].transAxes, ha="right", fontsize=9, color=INK)
axes[1].text(0.97, 0.04, f"순위상관 {k1:.2f}", transform=axes[1].transAxes, ha="right", fontsize=9, color=INK)
fig.suptitle("정답 없이 오차를 추정하려면, 다른 엔진보다 타임스텝을 반으로 줄여 보는 게 낫다",
             x=0.02, ha="left", fontweight="bold", fontsize=12)
fig.text(0.02, 0.87, "미끄러지는 상자 무작위 조건 1,500개", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(FIG / "fig11_disagreement_map.png", dpi=180)
plt.close(fig)

(RES / "summary_stage3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                         encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["fuzz_worst_by_rule", "fuzz_efficiency_v1"]}, ensure_ascii=False, indent=1))
