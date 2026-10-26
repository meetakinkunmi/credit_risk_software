"""Step 10: fairness audit + one mitigation (reweighing) with the accuracy/fairness trade-off."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config as C
from src.cost import CostOptimizer


class FairnessAuditor:
    """Compares decisions across groups. 'Positive outcome' = being APPROVED.

    approval_rate_among_good  -> equal-opportunity view (creditworthy people treated alike?)
    approval_rate_among_bad   -> error view (risky people slipping through, per group)
    disparate_impact_ratio    -> min approval rate / max approval rate (the 'four-fifths rule' flags < 0.8)"""

    def __init__(self, attrs=C.PROTECTED_ATTRS, cost_opt: CostOptimizer = None, min_group=30):
        self.attrs, self.cost_opt, self.min_group = attrs, cost_opt or CostOptimizer(), min_group

    def group_table(self, df, y_bad, approve, attr) -> pd.DataFrame:
        d = pd.DataFrame({"g": df[attr].values, "y": np.asarray(y_bad).astype(int),
                          "a": np.asarray(approve).astype(int)})
        rows = []
        for g, s in d.groupby("g"):
            good, bad = s[s.y == 0], s[s.y == 1]
            rows.append({"attribute": attr, "group": g, "n": len(s), "actual_bad_rate": s.y.mean(),
                         "approval_rate": s.a.mean(),
                         "approval_rate_among_good": good.a.mean() if len(good) else np.nan,
                         "approval_rate_among_bad": bad.a.mean() if len(bad) else np.nan,
                         "cost_per_applicant": self.cost_opt.total_cost(s.y, s.a) / len(s),
                         "small_group_warning": len(s) < self.min_group})
        return pd.DataFrame(rows)

    def summary(self, df, y_bad, approve) -> pd.DataFrame:
        rows = []
        for attr in self.attrs:
            t = self.group_table(df, y_bad, approve, attr)
            t = t[~t["small_group_warning"]] if (~t["small_group_warning"]).sum() >= 2 else t
            ar = t["approval_rate"]
            rows.append({"attribute": attr,
                         "disparate_impact_ratio": ar.min() / ar.max() if ar.max() > 0 else np.nan,
                         "approval_rate_gap": ar.max() - ar.min(),
                         "equal_opportunity_gap": t["approval_rate_among_good"].max() - t["approval_rate_among_good"].min(),
                         "false_approval_gap": t["approval_rate_among_bad"].max() - t["approval_rate_among_bad"].min()})
        out = pd.DataFrame(rows)
        out["passes_four_fifths"] = out["disparate_impact_ratio"] >= 0.8
        return out

    def audit(self, df, y_bad, approve) -> dict:
        return {a: self.group_table(df, y_bad, approve, a) for a in self.attrs}


class ReweighingMitigator:
    """Pre-processing mitigation (Kamiran & Calders): weight each (group, label) cell by
    P(group) * P(label) / P(group, label) so the label becomes independent of the group in training."""

    def __init__(self, attr: str):
        self.attr = attr

    def weights(self, df: pd.DataFrame, y_bad) -> np.ndarray:
        a, y = df[self.attr].values, np.asarray(y_bad).astype(int)
        w = np.ones(len(y))
        for g in np.unique(a):
            for yv in (0, 1):
                m = (a == g) & (y == yv)
                if m.any():
                    w[m] = (a == g).mean() * (y == yv).mean() / m.mean()
        return w


class FairnessMitigationExperiment:
    """Baseline vs reweighed models. Each variant gets its OWN cost-optimal threshold from
    out-of-fold training predictions, then is judged once on the hold-out set."""

    def __init__(self, trainer, factory, cost_opt: CostOptimizer, auditor: FairnessAuditor):
        self.trainer, self.factory, self.cost_opt, self.auditor = trainer, factory, cost_opt, auditor

    def run(self, train, y_train, test, y_test) -> pd.DataFrame:
        variants = {"baseline (no mitigation)": None}
        for a in self.auditor.attrs:
            variants[f"reweighing on {a}"] = ReweighingMitigator(a)
        rows = []
        for name, mit in variants.items():
            wfn = mit.weights if mit else None
            oof = self.trainer.oof_predict(self.factory.logreg, train, y_train, weight_fn=wfn)
            thr = self.cost_opt.optimal_threshold(y_train, oof)
            model = self.factory.logreg().fit(train, y_train, sample_weight=wfn(train, y_train) if wfn else None)
            p = model.predict_pd(test)
            approve = p < thr
            row = {"variant": name, "threshold": thr, "test_auc": roc_auc_score(y_test, p),
                   "approval_rate": approve.mean(),
                   "cost_per_applicant": self.cost_opt.total_cost(y_test, approve) / len(y_test)}
            for _, r in self.auditor.summary(test, y_test, approve).iterrows():
                row[f"DI_{r['attribute']}"] = r["disparate_impact_ratio"]
                row[f"EOgap_{r['attribute']}"] = r["equal_opportunity_gap"]
            rows.append(row)
        return pd.DataFrame(rows)
