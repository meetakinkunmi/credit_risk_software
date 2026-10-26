"""Step 13: the single object shipped to production. 
API and dashboard both call ModelBundle.score()."""
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.data import GermanCreditData


@dataclass
class ModelBundle:
    champion_name: str
    champion: object            # anything with predict_pd(df)
    scorecard: object           # used for points + reason codes
    threshold: float            # approve if PD < threshold (cost-optimal, from training OOF)
    score_cutoff: float         # scorecard-points cutoff (cost-optimal, from training OOF)
    segmenter: object
    segment_policies: dict
    anomaly: object
    monitor: object
    version: str = "1.0.0"
    metadata: dict = field(default_factory=dict)

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        """raw = frame with the 20 UCI columns using UCI CODES (A11, A12, ...). 
        Raises ValueError on bad codes."""
        df = GermanCreditData.prepare(raw)
        pd_ = np.asarray(self.champion.predict_pd(df))
        flags = self.anomaly.score(df)
        out = pd.DataFrame({
            "pd": pd_,
            "score": self.scorecard.score(df).round(0),
            "risk_band": pd.cut(pd_, [-np.inf, *C.RISK_BAND_EDGES, np.inf],
                                labels=C.RISK_BAND_LABELS).astype(str),
            "decision": np.where(pd_ < self.threshold, "APPROVE", "DECLINE"),
            "manual_review": flags["manual_review"].values,
            "review_reason": flags["review_reason"].values,
            "segment": self.segmenter.predict(df),
        })
        out["final_action"] = np.where(out["manual_review"], "MANUAL_REVIEW", out["decision"])
        out["segment_policy"] = out["segment"].map(self.segment_policies)
        out["main_risk_factors"] = self.scorecard.reason_codes(df, n=3)
        return out

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        return joblib.load(path)
