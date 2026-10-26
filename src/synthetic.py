"""Synthetic data with the SAME schema as german.data.

Used ONLY to smoke-test the code (`python run_pipeline.py --synthetic`). It has no relationship
with the real dataset, so never interpret results produced from it."""
import numpy as np
import pandas as pd

from src import schema as S


def make_synthetic_raw(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = {}
    for col in S.COLUMNS:
        if col in S.CODE_MAPS:
            d[col] = rng.choice(list(S.CODE_MAPS[col].keys()), size=n)
    d["duration_months"] = rng.integers(4, 73, n)
    d["credit_amount"] = np.round(rng.lognormal(7.8, 0.8, n)).astype(int)
    d["installment_rate"] = rng.integers(1, 5, n)
    d["residence_since"] = rng.integers(1, 5, n)
    d["age"] = rng.integers(19, 76, n)
    d["existing_credits"] = rng.integers(1, 5, n)
    d["dependents"] = rng.integers(1, 3, n)
    df = pd.DataFrame(d)[S.COLUMNS]
    logit = (-1.6 + 0.03 * df["duration_months"] + 0.0001 * df["credit_amount"]
             + 0.9 * (df["checking_status"] == "A11") + 0.5 * (df["credit_history"] == "A34")
             - 0.02 * (df["age"] - 35))
    p_bad = 1 / (1 + np.exp(-logit))
    df["target"] = np.where(rng.random(n) < p_bad, 2, 1)
    return df
