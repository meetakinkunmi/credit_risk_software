"""Usage:
    python run_pipeline.py --download          # fetch UCI data, then run everything
    python run_pipeline.py                     # run with data/german.data already in place
    python run_pipeline.py --synthetic         # PLUMBING TEST ONLY (fake data, separate .smoke/ folder)
"""
import argparse
from pathlib import Path

from src.config import ROOT
from src.data import GermanCreditData
from src.pipeline import CreditRiskProject

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download german.data from UCI first")
    ap.add_argument("--synthetic", action="store_true", help="use fake data to test the code (writes to .smoke/)")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--no-tune", action="store_true", help="skip GBM grid search (faster)")
    a = ap.parse_args()
    if a.download:
        GermanCreditData().download()
    workdir = ROOT / ".smoke" if a.synthetic else ROOT
    CreditRiskProject(workdir=Path(workdir), synthetic=a.synthetic, run_shap=not a.no_shap,
                      tune=not a.no_tune).run_all()
