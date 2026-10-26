"""Step 3: Weight-of-Evidence binning, Information Value and the points-based scorecard.

Conventions
-----------
* y_bad: 1 = bad (default), 0 = good.
* WoE = ln(%good / %bad): positive WoE => the bin is safer than average.
* The logistic regression models the odds of GOOD, so coefficients are positive and a higher
  score means a safer customer.  Score = offset + factor * ln(odds_good).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config as C
from src.schema import FRIENDLY


class WoEBinner:
    """Learns bins + WoE per column on TRAIN data only, then maps any frame to WoE values."""

    def __init__(self, n_bins=C.N_BINS, min_bin_frac=C.MIN_BIN_FRACTION,
                 max_unique_numeric=6, smoothing=0.5):
        self.n_bins, self.min_bin_frac = n_bins, min_bin_frac
        self.max_unique_numeric, self.smoothing = max_unique_numeric, smoothing

    # ---- fitting -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y_bad, cols=None):
        y = np.asarray(y_bad).astype(int)
        self.cols_ = list(cols if cols is not None else X.columns)
        self.specs_, self.tables_ = {}, {}
        for c in self.cols_:
            spec = self._make_spec(X[c])
            labels = self._assign(X[c], spec)
            self.specs_[c] = spec
            self.tables_[c] = self._woe_table(labels, y, spec)
        self.iv_ = pd.Series({c: t["iv"].sum() for c, t in self.tables_.items()}).sort_values(ascending=False)
        return self

    def _make_spec(self, s: pd.Series) -> dict:
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > self.max_unique_numeric:
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            cuts = np.unique(np.quantile(s.dropna(), qs))
            return {"kind": "numeric", "edges": [-np.inf, *cuts.tolist(), np.inf],
                    "labels": self._interval_labels(cuts)}
        freq = s.astype(str).value_counts(normalize=True)
        return {"kind": "categorical", "levels": freq[freq >= self.min_bin_frac].index.tolist()}

    @staticmethod
    def _interval_labels(cuts):
        if len(cuts) == 0:
            return ["ALL"]
        labs = [f"<= {cuts[0]:g}"]
        labs += [f"({cuts[i]:g}, {cuts[i + 1]:g}]" for i in range(len(cuts) - 1)]
        labs.append(f"> {cuts[-1]:g}")
        return labs

    @staticmethod
    def _assign(s: pd.Series, spec: dict) -> pd.Series:
        if spec["kind"] == "numeric":
            out = pd.cut(s, bins=spec["edges"], labels=spec["labels"], include_lowest=True)
            out = out.astype(object)
            out = out.where(~s.isna(), "MISSING")
            return pd.Series(out.values, index=s.index, dtype=object)
        lab = s.astype(str)
        return lab.where(lab.isin(spec["levels"]), "OTHER").astype(object)

    def _woe_table(self, labels: pd.Series, y: np.ndarray, spec: dict) -> pd.DataFrame:
        df = pd.DataFrame({"bin": labels.values, "bad": y})
        g = df.groupby("bin")["bad"].agg(n="size", bad="sum")
        g["good"] = g["n"] - g["bad"]
        sg, sb = g["good"] + self.smoothing, g["bad"] + self.smoothing
        g["dist_good"], g["dist_bad"] = sg / sg.sum(), sb / sb.sum()
        g["bad_rate"] = g["bad"] / g["n"]
        g["woe"] = np.log(g["dist_good"] / g["dist_bad"])
        g["iv"] = (g["dist_good"] - g["dist_bad"]) * g["woe"]
        if spec["kind"] == "numeric":
            order = [l for l in spec["labels"] + ["MISSING"] if l in g.index]
            g = g.loc[order]
        else:
            g = g.sort_values("woe")
        return g.reset_index()

    # ---- transforming ------------------------------------------------------
    def transform_labels(self, X: pd.DataFrame, cols=None) -> pd.DataFrame:
        cols = list(cols if cols is not None else self.cols_)
        return pd.DataFrame({c: self._assign(X[c], self.specs_[c]) for c in cols}, index=X.index)

    def transform(self, X: pd.DataFrame, cols=None) -> pd.DataFrame:
        cols = list(cols if cols is not None else self.cols_)
        labels = self.transform_labels(X, cols)
        out = {}
        for c in cols:
            mapping = self.tables_[c].set_index("bin")["woe"]
            out[c] = labels[c].map(mapping).astype(float).fillna(0.0)  # unseen bin -> neutral
        return pd.DataFrame(out, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        t = self.iv_.rename("iv").reset_index().rename(columns={"index": "variable"})
        t.columns = ["variable", "iv"]
        t["strength"] = pd.cut(t["iv"], [-1, 0.02, 0.1, 0.3, 0.5, np.inf],
                               labels=["useless", "weak", "medium", "strong", "suspiciously strong"]).astype(str)
        return t


class Scorecard:
    """WoE binning -> variable selection -> logistic regression -> points.

    Exposes the same fit()/predict_pd() interface as every other model in the project."""
    name = "scorecard"

    def __init__(self, columns=None, iv_min=C.IV_MIN, corr_max=C.CORR_MAX, C_reg=1.0,
                 base_score=C.BASE_SCORE, base_odds=C.BASE_ODDS, pdo=C.PDO):
        self.columns, self.iv_min, self.corr_max, self.C_reg = columns, iv_min, corr_max, C_reg
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    # ---- training ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y_bad, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("Scorecard does not support sample weights.")
        cols = self.columns or list(X.columns)
        self.input_cols_ = cols
        y_bad = np.asarray(y_bad).astype(int)
        self.binner_ = WoEBinner().fit(X, y_bad, cols)
        iv = self.binner_.iv_
        candidates = [c for c in iv.index if iv[c] >= self.iv_min]
        if len(candidates) < 2:                       # safety net for tiny / noisy data
            candidates = list(iv.index[:3])
        woe = self.binner_.transform(X, candidates)
        keep = self._drop_correlated(woe, iv)
        y_good = 1 - y_bad
        self.dropped_negative_ = []
        while True:
            lr = LogisticRegression(C=self.C_reg, max_iter=2000).fit(woe[keep], y_good)
            coefs = lr.coef_[0]
            if (coefs >= 0).all() or len(keep) <= 1:
                break
            worst = keep[int(np.argmin(coefs))]      # sign flip = multicollinearity -> remove
            self.dropped_negative_.append(worst)
            keep.remove(worst)
        self.lr_, self.features_ = lr, keep
        self._build_points()
        return self

    def _drop_correlated(self, woe: pd.DataFrame, iv: pd.Series) -> list:
        corr, keep = woe.corr().abs(), []
        for c in sorted(woe.columns, key=lambda c: -iv[c]):
            if all(corr.loc[c, k] < self.corr_max for k in keep):
                keep.append(c)
        return keep

    def _build_points(self):
        n = len(self.features_)
        base = (self.offset + self.factor * self.lr_.intercept_[0]) / n
        self.base_points_ = base
        rows = []
        for c, b in zip(self.features_, self.lr_.coef_[0]):
            t = self.binner_.tables_[c]
            rows.append(pd.DataFrame({
                "feature": c, "bin": t["bin"], "n": t["n"], "bad_rate": t["bad_rate"],
                "woe": t["woe"], "coef": b, "points": self.factor * b * t["woe"] + base}))
        self.points_table_ = pd.concat(rows, ignore_index=True)

    # ---- inference ---------------------------------------------------------
    def _log_odds_good(self, X: pd.DataFrame) -> np.ndarray:
        woe = self.binner_.transform(X, self.features_)
        return self.lr_.decision_function(woe[self.features_])

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(self._log_odds_good(X)))   # P(bad) = 1 - sigmoid(z)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.offset + self.factor * self._log_odds_good(X)

    def pd_to_score(self, pd_):
        p = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
        return self.offset + self.factor * np.log((1 - p) / p)

    def points_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points contributed by each retained variable for each applicant (rows sum to the score)."""
        labels = self.binner_.transform_labels(X, self.features_)
        out = {}
        for c in self.features_:
            m = self.points_table_[self.points_table_.feature == c].set_index("bin")["points"]
            out[c] = labels[c].map(m).astype(float).fillna(self.base_points_)
        return pd.DataFrame(out, index=X.index)

    def reason_codes(self, X: pd.DataFrame, n: int = 3) -> list:
        """Adverse-action style reasons: variables where the applicant lost the most points
        compared with the best possible bin of that variable."""
        P = self.points_matrix(X)
        best = self.points_table_.groupby("feature")["points"].max()
        labels = self.binner_.transform_labels(X, self.features_)
        reasons = []
        for i in range(len(X)):
            loss = (best[self.features_] - P.iloc[i]).sort_values(ascending=False).head(n)
            reasons.append([{"factor": FRIENDLY.get(c, c), "applicant_value": str(labels.iloc[i][c]),
                             "points_lost": round(float(v), 1)} for c, v in loss.items() if v > 0])
        return reasons
