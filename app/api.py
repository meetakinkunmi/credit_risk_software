"""FastAPI service.   Run:  uvicorn app.api:app --host 0.0.0.0 --port 8000

Input uses the ORIGINAL UCI codes (A11, A12, ...), exactly like the source dataset."""
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import schema as S
from src.bundle import ModelBundle
from src.config import ROOT

BUNDLE_PATH = Path(__import__("os").environ.get("MODEL_BUNDLE", ROOT / "artifacts" / "model_bundle.joblib"))
LOG_PATH = BUNDLE_PATH.parent / "prediction_log.jsonl"
state = {}


class Applicant(BaseModel):
    checking_status: str = Field(..., examples=["A11"], description="Attribute1: A11-A14")
    duration_months: int = Field(..., ge=1, le=120, examples=[24])
    credit_history: str = Field(..., examples=["A32"], description="Attribute3: A30-A34")
    purpose: str = Field(..., examples=["A43"], description="Attribute4: A40-A49, A410")
    credit_amount: float = Field(..., gt=0, examples=[3500])
    savings: str = Field(..., examples=["A61"], description="Attribute6: A61-A65")
    employment_since: str = Field(..., examples=["A73"], description="Attribute7: A71-A75")
    installment_rate: int = Field(..., ge=1, le=4, examples=[3])
    personal_status_sex: str = Field(..., examples=["A93"], description="Attribute9: A91-A95")
    other_debtors: str = Field(..., examples=["A101"], description="Attribute10: A101-A103")
    residence_since: int = Field(..., ge=1, le=4, examples=[2])
    property: str = Field(..., examples=["A121"], description="Attribute12: A121-A124")
    age: int = Field(..., ge=18, le=100, examples=[35])
    other_installment_plans: str = Field(..., examples=["A143"], description="Attribute14: A141-A143")
    housing: str = Field(..., examples=["A152"], description="Attribute15: A151-A153")
    existing_credits: int = Field(..., ge=1, le=4, examples=[1])
    job: str = Field(..., examples=["A173"], description="Attribute17: A171-A174")
    dependents: int = Field(..., ge=1, le=2, examples=[1])
    telephone: str = Field(..., examples=["A192"], description="Attribute19: A191-A192")
    foreign_worker: str = Field(..., examples=["A201"], description="Attribute20: A201-A202")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not BUNDLE_PATH.exists():
        raise RuntimeError(f"Model bundle not found at {BUNDLE_PATH}. Run `python run_pipeline.py` first.")
    state["bundle"] = ModelBundle.load(BUNDLE_PATH)
    yield


app = FastAPI(title="Credit Risk Scoring API", version="1.0.0", lifespan=lifespan)


def _score(applicants: list) -> pd.DataFrame:
    raw = pd.DataFrame([a.model_dump() for a in applicants])[S.COLUMNS]
    try:
        res = state["bundle"].score(raw)
    except ValueError as e:                       # invalid category codes
        raise HTTPException(status_code=422, detail=str(e))
    with open(LOG_PATH, "a") as f:                # feeds drift monitoring; contains applicant data -> protect it
        for rec, r in zip(raw.to_dict("records"), res["pd"]):
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "pd": float(r), **rec}) + "\n")
    return res


@app.get("/health")
def health():
    return {"status": "ok", "model": state["bundle"].champion_name, "version": state["bundle"].version}


@app.get("/model-info")
def model_info():
    b = state["bundle"]
    return {"champion": b.champion_name, "pd_threshold": b.threshold, "score_cutoff": b.score_cutoff,
            "segments": b.segment_policies, "metadata": b.metadata}


@app.post("/predict")
def predict(applicant: Applicant):
    return _score([applicant]).iloc[0].to_dict() | {"pd_threshold": state["bundle"].threshold}


@app.post("/predict/batch")
def predict_batch(applicants: list[Applicant]):
    if not 0 < len(applicants) <= 1000:
        raise HTTPException(422, "Send between 1 and 1000 applicants.")
    return _score(applicants).to_dict("records")


@app.get("/monitor")
def monitor(min_rows: int = 100):
    """PSI of logged production traffic vs the training reference."""
    if not LOG_PATH.exists():
        raise HTTPException(404, "No predictions logged yet.")
    logs = pd.read_json(LOG_PATH, lines=True)
    if len(logs) < min_rows:
        raise HTTPException(409, f"Need at least {min_rows} logged predictions, have {len(logs)}.")
    from src.data import GermanCreditData
    df = GermanCreditData.prepare(logs[S.COLUMNS])
    return state["bundle"].monitor.report(df, logs["pd"].values).to_dict("records")
