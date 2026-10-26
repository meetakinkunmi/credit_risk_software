"""Step 4: cost-sensitive decision threshold using the dataset's 5:1 cost matrix."""
import numpy as np
import pandas as pd

from src import config as C


class CostOptimizer:
    """Decision rule: APPROVE when PD < threshold (or score >= cutoff), otherwise REJECT.

    False approval (approve a bad customer) costs `cost_false_approve`;
    false rejection (reject a good customer) costs `cost_false_reject`."""

    def __init__(self, cost_false_approve=C.COST_FALSE_APPROVE, cost_false_reject=C.COST_FALSE_REJECT):
        self.c_fa, self.c_fr = cost_false_approve, cost_false_reject

    @property
    def theoretical_threshold(self) -> float:
        """For PERFECTLY calibrated PDs: approve iff c_fa*PD < c_fr*(1-PD)  =>  PD < c_fr/(c_fa+c_fr)."""
        return self.c_fr / (self.c_fa + self.c_fr)

    def breakdown(self, y_bad, approve) -> dict:
        y, a = np.asarray(y_bad).astype(bool), np.asarray(approve).astype(bool)
        fa, fr = int((a & y).sum()), int((~a & ~y).sum())
        return {"approved": int(a.sum()), "rejected": int((~a).sum()),
                "false_approvals": fa, "false_rejections": fr,
                "true_approvals": int((a & ~y).sum()), "true_rejections": int((~a & y).sum()),
                "cost_false_approvals": self.c_fa * fa, "cost_false_rejections": self.c_fr * fr,
                "total_cost": self.c_fa * fa + self.c_fr * fr}

    def total_cost(self, y_bad, approve) -> float:
        return self.breakdown(y_bad, approve)["total_cost"]

    def curve(self, y_bad, pd_, grid=None) -> pd.DataFrame:
        grid = np.round(np.arange(0.02, 0.98, 0.005), 3) if grid is None else grid
        pd_ = np.asarray(pd_)
        rows = []
        for t in grid:
            b = self.breakdown(y_bad, pd_ < t)
            rows.append({"threshold": t, "approval_rate": b["approved"] / len(pd_), **b})
        out = pd.DataFrame(rows)
        out["cost_per_applicant"] = out["total_cost"] / len(pd_)
        return out

    def optimal_threshold(self, y_bad, pd_, grid=None) -> float:
        """Argmin of total cost. Ties (very common on small data) -> median of the tied thresholds."""
        cv = self.curve(y_bad, pd_, grid)
        return float(cv.loc[cv["total_cost"] == cv["total_cost"].min(), "threshold"].median())

    def optimal_score_cutoff(self, y_bad, scores) -> float:
        """Approve when score >= cutoff. Candidates are midpoints between sorted unique scores."""
        s = np.asarray(scores, dtype=float)
        u = np.unique(s)
        cands = np.concatenate([[u[0] - 1], (u[:-1] + u[1:]) / 2, [u[-1] + 1]])
        costs = np.array([self.total_cost(y_bad, s >= c) for c in cands])
        return float(np.median(cands[costs == costs.min()]))
