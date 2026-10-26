"""Step 12: drift monitoring with the Population Stability Index (PSI)."""
import numpy as np
import pandas as pd


class DriftMonitor:
    """PSI < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def fit(self, df: pd.DataFrame, pd_scores, num_cols, cat_cols):
        self.ref_ = {}
        for c in num_cols:
            self.ref_[c] = self._num_ref(df[c].values)
        for c in cat_cols:
            self.ref_[c] = {"kind": "cat", "freq": df[c].value_counts(normalize=True).to_dict()}
        self.ref_["__pd__"] = self._num_ref(np.asarray(pd_scores))
        return self

    def _num_ref(self, x):
        edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        return {"kind": "num", "edges": edges, "freq": np.histogram(x, edges)[0] / len(x)}

    @staticmethod
    def psi(expected, actual, eps=1e-4) -> float:
        e, a = np.clip(np.asarray(expected, float), eps, None), np.clip(np.asarray(actual, float), eps, None)
        return float(np.sum((a - e) * np.log(a / e)))

    def report(self, df: pd.DataFrame, pd_scores) -> pd.DataFrame:
        rows = []
        for c, ref in self.ref_.items():
            if c == "__pd__":
                cur = np.histogram(np.asarray(pd_scores), ref["edges"])[0] / len(pd_scores)
                val = self.psi(ref["freq"], cur)
                name = "predicted_pd"
            elif ref["kind"] == "num":
                cur = np.histogram(df[c].values, ref["edges"])[0] / len(df)
                val, name = self.psi(ref["freq"], cur), c
            else:
                cats = sorted(set(ref["freq"]) | set(df[c].unique()))
                cur_f = df[c].value_counts(normalize=True)
                val = self.psi([ref["freq"].get(k, 0) for k in cats], [cur_f.get(k, 0) for k in cats])
                name = c
            rows.append({"feature": name, "psi": val,
                         "status": "stable" if val < 0.10 else "moderate" if val < 0.25 else "significant"})
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
