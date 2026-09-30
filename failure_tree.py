"""SimDiff 3단계-A4: 실패 원인을 의사결정나무로 독립 확인.

2단계 Sobol 표본(무작위에 가까운 균일 표본)에 깊이 3짜리 의사결정나무를 학습시켜
'|오차| > 1%'를 가르는 변수를 찾는다. 퍼징과 독립적으로 같은 원인이 나오는지 교차 확인하는 용도.

python failure_tree.py  →  results/failure_tree.txt, results/failure_tree.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

RES = Path(__file__).parent / "results"
G = 9.81


def features(eng):
    z = np.load(RES / f"sobol_samples_{eng}.npz")
    X, Y = z["X"], np.abs(z["Y"])
    dt, mu, v0 = 10 ** X[:, 0], X[:, 1], X[:, 2]
    ts = v0 / (mu * G)
    F = pd.DataFrame({"log10_steps": np.log10(ts / dt), "tau": mu, "log10_dt_ms": np.log10(dt * 1000)})
    if eng == "mujoco":
        tc = 10 ** X[:, 3]
        F["log10_tc_over_tstop"] = np.log10(tc / ts)
        F["log10_tc_over_dt"] = np.log10(tc / dt)
        F["impratio"] = X[:, 4]
        F["elliptic"] = (X[:, 5] >= 0.5).astype(int)
    else:
        F["iterations"] = np.round(10 ** X[:, 3])
        F["contact_erp"] = X[:, 4]
    return F, (Y > 1).astype(int)


def main():
    text, js = [], {}
    for eng in ["mujoco", "pybullet"]:
        F, y = features(eng)
        cv = {}
        for depth in [2, 3, 4]:
            t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=300, random_state=0)
            cv[depth] = float(cross_val_score(t, F, y, cv=5).mean())
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=300, random_state=0).fit(F, y)
        imp = dict(sorted(zip(F.columns, tree.feature_importances_), key=lambda kv: -kv[1]))
        js[eng] = {"cv_accuracy_by_depth": cv, "baseline_accuracy": float(1 - y.mean()),
                   "feature_importance_depth3": {k: round(float(v), 3) for k, v in imp.items()}}
        text.append(f"== {eng} (실패율 {y.mean():.3f}, 깊이3 교차검증 정확도 {cv[3]:.3f}) ==\n")
        text.append(export_text(tree, feature_names=list(F.columns), show_weights=True, decimals=2))
    (RES / "failure_tree.txt").write_text("\n".join(text), encoding="utf-8")
    (RES / "failure_tree.json").write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(js, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
