"""Streamlit dashboard.   Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import schema as S
from src.bundle import ModelBundle

TABLES, FIGS = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
st.set_page_config(page_title="Credit Risk", layout="wide")
st.title("Credit risk dashboard - German Credit")


def table(name, caption=None):
    p = TABLES / f"{name}.csv"
    if p.exists():
        if caption:
            st.caption(caption)
        st.dataframe(pd.read_csv(p), use_container_width=True)
    else:
        st.info(f"{name}.csv not found - run the pipeline first.")


def fig(name):
    p = FIGS / f"{name}.png"
    if p.exists():
        st.image(str(p))


tabs = st.tabs(["Models", "Scorecard", "Cost & policies", "Segments", "Fairness", "Anomalies", "Score an applicant"])

with tabs[0]:
    table("model_comparison_cv_oof", "Cross-validated (out-of-fold) on training data")
    table("model_comparison_test", "Hold-out test set")
    c1, c2 = st.columns(2)
    with c1: fig("roc_test")
    with c2: fig("calibration_test")
    fig("shap_summary")
    table("odds_ratios")

with tabs[1]:
    table("scorecard_iv_table")
    table("scorecard_points_table")
    fig("score_distribution")

with tabs[2]:
    table("cost_threshold_summary")
    fig("cost_curve")
    table("policy_comparison")
    fig("policy_comparison")
    st.caption("Profit figures depend on MARGIN_RATE and LOSS_GIVEN_DEFAULT in src/config.py (assumptions).")

with tabs[3]:
    table("segmentation_k_selection")
    table("segment_profile_train")
    table("segment_profile_test", "Same segments on unseen data - are the default rates stable?")

with tabs[4]:
    table("fairness_summary")
    for a in ["age_group", "sex", "foreign_worker"]:
        table(f"fairness_groups_{a}")
    table("fairness_mitigation_tradeoff")

with tabs[5]:
    table("anomalies_test")
    table("anomaly_flag_vs_bad_rate")

with tabs[6]:
    path = ROOT / "artifacts" / "model_bundle.joblib"
    if not path.exists():
        st.info("Train first: python run_pipeline.py")
    else:
        bundle = ModelBundle.load(path)
        with st.form("applicant"):
            vals = {}
            cols = st.columns(3)
            for i, col in enumerate(S.COLUMNS):
                with cols[i % 3]:
                    if col in S.CODE_MAPS:
                        m = S.CODE_MAPS[col]
                        pick = st.selectbox(col, list(m), format_func=lambda k, m=m: f"{k} - {m[k]}")
                        vals[col] = pick
                    elif col == "credit_amount":
                        vals[col] = st.number_input(col, 250, 20000, 3000)
                    elif col == "age":
                        vals[col] = st.number_input(col, 18, 100, 35)
                    elif col == "duration_months":
                        vals[col] = st.number_input(col, 1, 120, 24)
                    else:
                        hi = 2 if col == "dependents" else 4
                        vals[col] = st.number_input(col, 1, hi, 1)
            go = st.form_submit_button("Score")
        if go:
            res = bundle.score(pd.DataFrame([vals])[S.COLUMNS]).iloc[0]
            c = st.columns(4)
            c[0].metric("Probability of default", f"{res['pd']:.1%}")
            c[1].metric("Score", int(res["score"]))
            c[2].metric("Risk band", res["risk_band"])
            c[3].metric("Action", res["final_action"])
            if res["manual_review"]:
                st.warning(f"Manual review: {res['review_reason']}")
            st.write("Main risk factors:", res["main_risk_factors"])
            st.write("Segment policy:", res["segment_policy"])
