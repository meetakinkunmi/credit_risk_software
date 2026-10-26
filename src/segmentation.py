"""Step 6: customer segmentation with k-means or a Gaussian mixture, plus a policy per segment."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src import schema as S


class CustomerSegmenter:
    def __init__(self, method="gmm", k_range=range(2, 8), k=None, seed=C.RANDOM_STATE):
        assert method in ("kmeans", "gmm")
        self.method, self.k_range, self.k, self.seed = method, k_range, k, seed

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["log_amount"] = np.log1p(df["credit_amount"])
        X["duration"], X["age"] = df["duration_months"], df["age"]
        X["installment_rate"], X["existing_credits"] = df["installment_rate"], df["existing_credits"]
        X["employment_level"] = df["employment_since"].map(S.ORDER_EMPLOYMENT)
        X["job_level"] = df["job"].map(S.ORDER_JOB)
        X["savings_level"] = df["savings"].map(S.ORDER_SAVINGS)           # unknown -> NaN -> imputed
        X["savings_unknown"] = (df["savings"] == "unknown / none").astype(int)
        X["checking_level"] = df["checking_status"].map(S.ORDER_CHECKING)  # no account -> NaN -> imputed
        X["no_checking"] = (df["checking_status"] == "no checking account").astype(int)
        X["owns_home"] = (df["housing"] == "own").astype(int)
        return X

    def select_k(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evidence for choosing k: silhouette (k-means, higher better) and BIC (GMM, lower better)."""
        Z = self.prep_.fit_transform(self._matrix(df))
        rows = []
        for k in self.k_range:
            km = KMeans(k, n_init=10, random_state=self.seed).fit(Z)
            gm = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=self.seed).fit(Z)
            rows.append({"k": k, "kmeans_silhouette": silhouette_score(Z, km.labels_), "gmm_bic": gm.bic(Z)})
        return pd.DataFrame(rows)

    def fit(self, df: pd.DataFrame):
        self.prep_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.k_table_ = self.select_k(df)
        if self.k is None:
            self.k_ = int(self.k_table_.loc[self.k_table_["kmeans_silhouette"].idxmax(), "k"]
                          if self.method == "kmeans" else self.k_table_.loc[self.k_table_["gmm_bic"].idxmin(), "k"])
        else:
            self.k_ = self.k
        Z = self.prep_.fit_transform(self._matrix(df))
        self.model_ = (KMeans(self.k_, n_init=10, random_state=self.seed) if self.method == "kmeans"
                       else GaussianMixture(self.k_, covariance_type="diag", n_init=3, random_state=self.seed)).fit(Z)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.prep_.transform(self._matrix(df)))

    def profile(self, df: pd.DataFrame, y_bad) -> pd.DataFrame:
        d = df.copy()
        d["cluster"], d["bad"] = self.predict(df), np.asarray(y_bad)
        g = d.groupby("cluster")
        prof = g.agg(n=("bad", "size"), bad_rate=("bad", "mean"), avg_amount=("credit_amount", "mean"),
                     avg_duration=("duration_months", "mean"), avg_age=("age", "mean"),
                     avg_installment_rate=("installment_rate", "mean"))
        for col in ["checking_status", "savings", "purpose", "housing", "job"]:
            prof[f"typical_{col}"] = g[col].agg(lambda s: s.mode().iat[0])
        prof["share"] = prof["n"] / len(d)
        prof["risk_lift"] = prof["bad_rate"] / d["bad"].mean()
        return prof.reset_index()

    @staticmethod
    def suggest_policy(profile: pd.DataFrame, pd_threshold: float) -> pd.DataFrame:
        """Rule of thumb tied to the cost-optimal threshold t: approving a whole segment pays off
        only if its bad rate is below t. A starting point for a credit officer, not a final policy."""
        def rule(r):
            br = r["bad_rate"]
            if br <= 0.5 * pd_threshold:
                return "Fast-track approval; competitive pricing; higher limits"
            if br <= pd_threshold:
                return "Approve with standard checks; risk-based pricing"
            if br <= 1.5 * pd_threshold:
                return "Conditional approval: lower limit, shorter tenor, guarantor/collateral, or manual review"
            return "Decline by default; exceptions only with strong mitigants (guarantor, collateral, down payment)"
        out = profile.copy()
        out["suggested_policy"] = out.apply(rule, axis=1)
        return out
