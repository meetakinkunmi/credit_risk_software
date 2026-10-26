"""Step 2: exploratory data analysis helpers (tables first, plots in viz.py)."""
import numpy as np
import pandas as pd

from src import schema as S


class EDA:
    def __init__(self, df: pd.DataFrame, target: str = S.TARGET):
        self.df, self.target = df, target

    def summary(self) -> dict:
        return {
            "n_rows": int(len(self.df)),
            "bad_rate": float(self.df[self.target].mean()),
            "n_missing_cells": int(self.df[S.ALL_FEATURE_COLS].isna().sum().sum()),
            "duplicated_rows": int(self.df.duplicated().sum()),
        }

    def bad_rate_by(self, col: str, n_bins: int = 5) -> pd.DataFrame:
        """Default rate per category (or per quantile bin for high-cardinality numerics)."""
        s = self.df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
            s = pd.qcut(s, n_bins, duplicates="drop").astype(str)
        g = self.df.groupby(s, observed=True)[self.target].agg(n="size", bad_rate="mean").reset_index()
        g.columns = ["level", "n", "bad_rate"]
        g.insert(0, "variable", col)
        return g

    def bad_rate_tables(self) -> pd.DataFrame:
        cols = S.CATEGORICAL_COLS + S.NUMERIC_COLS + ["monthly_payment"]
        return pd.concat([self.bad_rate_by(c) for c in cols], ignore_index=True)

    def numeric_describe(self) -> pd.DataFrame:
        cols = S.NUMERIC_COLS + ["monthly_payment"]
        return self.df[cols].describe().T.assign(skew=self.df[cols].skew())
