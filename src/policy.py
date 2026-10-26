"""Step 7: compare lending policies on the hold-out set (cost matrix AND money)."""
import numpy as np
import pandas as pd

from src import config as C
from src.cost import CostOptimizer


class PolicySimulator:
    def __init__(self, cost_opt: CostOptimizer, margin_rate=C.MARGIN_RATE, lgd=C.LOSS_GIVEN_DEFAULT,
                 n_boot=1000, seed=C.RANDOM_STATE):
        self.cost_opt, self.margin, self.lgd = cost_opt, margin_rate, lgd
        self.n_boot, self.rng = n_boot, np.random.default_rng(seed)

    def evaluate(self, approve, y_bad, amount) -> dict:
        a, y, amt = (np.asarray(v) for v in (approve, y_bad, amount))
        a, y = a.astype(bool), y.astype(bool)
        b = self.cost_opt.breakdown(y, a)
        profit = float((self.margin * amt[a & ~y]).sum() - self.lgd * amt[a & y].sum())
        n = len(y)
        return {"approval_rate": b["approved"] / n,
                "bad_rate_among_approved": float(y[a].mean()) if a.any() else np.nan,
                "false_approvals": b["false_approvals"], "false_rejections": b["false_rejections"],
                "matrix_cost": b["total_cost"], "matrix_cost_per_applicant": b["total_cost"] / n,
                "expected_profit_DM": profit, "profit_per_applicant_DM": profit / n}

    def bootstrap_ci(self, approve, y_bad, level=0.95) -> tuple:
        """Cost per applicant is noisy on ~250 test rows: report an interval, not just a number."""
        a, y = np.asarray(approve).astype(bool), np.asarray(y_bad).astype(bool)
        vals = []
        for _ in range(self.n_boot):
            i = self.rng.integers(0, len(y), len(y))
            vals.append(self.cost_opt.total_cost(y[i], a[i]) / len(y))
        lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
        return float(lo), float(hi)

    def compare(self, policies: dict, y_bad, amount) -> pd.DataFrame:
        rows = []
        for name, approve in policies.items():
            r = self.evaluate(approve, y_bad, amount)
            r["cost_ci_low"], r["cost_ci_high"] = self.bootstrap_ci(approve, y_bad)
            rows.append({"policy": name, **r})
        return pd.DataFrame(rows)
