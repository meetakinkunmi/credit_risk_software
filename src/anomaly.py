"""Step 8: flag unusual applications for manual review (unsupervised + business rules)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config as C


class AnomalyDetector:
    FEATURES = ["log_credit_amount", "duration_months", "monthly_payment", "age",
                "installment_rate", "existing_credits"]

    def __init__(self, contamination=C.ANOMALY_CONTAMINATION, seed=C.RANDOM_STATE):
        self.contamination, self.seed = contamination, seed

    def fit(self, df: pd.DataFrame):
        self.iso_ = IsolationForest(n_estimators=300, contamination=self.contamination,
                                    random_state=self.seed).fit(df[self.FEATURES])
        q = df[["credit_amount", "monthly_payment", "duration_months"]].quantile
        self.thr_ = {"amount_q90": q(0.90)["credit_amount"], "amount_q75": q(0.75)["credit_amount"],
                     "payment_q95": q(0.95)["monthly_payment"], "duration_q25": q(0.25)["duration_months"]}
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        t = self.thr_
        rules = {
            "large amount over short duration": (df["credit_amount"] >= t["amount_q90"]) & (df["duration_months"] <= t["duration_q25"]),
            "very high implied monthly payment": df["monthly_payment"] >= t["payment_q95"],
            "young applicant with large loan": (df["age"] <= 23) & (df["credit_amount"] >= t["amount_q75"]),
        }
        out = pd.DataFrame(index=df.index)
        out["anomaly_score"] = -self.iso_.score_samples(df[self.FEATURES])   # higher = more unusual
        out["statistical_outlier"] = self.iso_.predict(df[self.FEATURES]) == -1
        for name, mask in rules.items():
            out[name] = mask.values
        rule_cols = list(rules)
        out["manual_review"] = out["statistical_outlier"] | out[rule_cols].any(axis=1)

        def why(row):
            r = [c for c in rule_cols if row[c]]
            if row["statistical_outlier"]:
                r.append("statistical outlier (isolation forest)")
            return "; ".join(r)

        out["review_reason"] = out.apply(why, axis=1)
        return out.drop(columns=rule_cols)

    @staticmethod
    def evaluate(flags: pd.DataFrame, y_bad) -> pd.DataFrame:
        """Sanity check: are flagged applications riskier? (A hypothesis to test, not a given.)"""
        d = pd.DataFrame({"flagged": flags["manual_review"].values, "bad": np.asarray(y_bad)})
        return d.groupby("flagged")["bad"].agg(n="size", bad_rate="mean").reset_index()
