"""Feature engineering as a scikit-learn compatible transformer."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds simple, explainable features. Stateless, so it cannot leak information."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["monthly_payment"] = X["credit_amount"] / X["duration_months"].clip(lower=1)
        X["log_credit_amount"] = np.log1p(X["credit_amount"])
        return X
