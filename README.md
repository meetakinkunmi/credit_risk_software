# German Credit Risk - class-based project (scorecard -> cost-sensitive policy -> API + dashboard)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --synthetic     # 1) plumbing test on FAKE data (writes to .smoke/, ignore its numbers)
pytest -q                              # 2) unit tests
python run_pipeline.py --download      # 3) real run: downloads UCI german.data, trains, writes reports/ + artifacts/
uvicorn app.api:app --reload           # 4) API  -> http://localhost:8000/docs
streamlit run app/dashboard.py         # 5) dashboard -> http://localhost:8501
docker compose up --build              # 6) both services in containers (after step 3)
```
If `--download` fails (no internet / UCI URL changed), download `german.data` from the UCI "Statlog (German Credit Data)" page and put it in `data/`.

## Project steps -> classes
| Step | Class (file) | Output in `reports/tables/` |
|---|---|---|
| Load, decode, split | `GermanCreditData` (data.py) | - |
| EDA | `EDA` (eda.py) | `eda_*.csv` |
| WoE / IV / scorecard points | `WoEBinner`, `Scorecard` (scorecard.py) | `scorecard_iv_table`, `scorecard_points_table` |
| PD models + CV + calibration + champion | `ModelFactory`, `ModelTrainer`, `MetricSuite` (models.py) | `model_comparison_*` |
| Cost-sensitive cutoff (5:1) | `CostOptimizer` (cost.py) | `cost_curve_train_oof`, `cost_threshold_summary` |
| Segmentation + policy per cluster | `CustomerSegmenter` (segmentation.py) | `segmentation_k_selection`, `segment_profile_*` |
| Risk drivers | `RiskDriverExplainer` (explain.py) | `odds_ratios`, `shap_importance` |
| Fairness audit + mitigation | `FairnessAuditor`, `ReweighingMitigator`, `FairnessMitigationExperiment` (fairness.py) | `fairness_*` |
| Policy comparison | `PolicySimulator` (policy.py) | `policy_comparison` |
| Anomalies | `AnomalyDetector` (anomaly.py) | `anomalies_test`, `anomaly_flag_vs_bad_rate` |
| Drift (PSI) | `DriftMonitor` (monitor.py) | `psi_train_vs_test` |
| Everything shipped | `ModelBundle` (bundle.py) | `artifacts/model_bundle.joblib` |
| Orchestration | `CreditRiskProject` (pipeline.py) | - |

## Design decisions worth understanding (first classification project)
1. **Hold-out discipline.** The 25% test set is used once, for final numbers. All tuning, binning, calibration, clustering and **threshold choice** use the training set (out-of-fold predictions).
2. **Target orientation.** `default` = 1 means BAD (raw file: 2 = bad). The scorecard models odds of GOOD internally so higher score = safer.
3. **Why not 0.5?** With costs 5:1, approving is worth it only if `5*PD < 1*(1-PD)`, i.e. `PD < 1/6`. That is the *theoretical* cutoff for perfectly calibrated PDs; the code also finds the *empirical* one and plots both. This is why calibration (Platt scaling for the boosted model) is part of the pipeline.
4. **Champion rule.** Best CV AUC wins, but the interpretable scorecard is kept if within 0.01 AUC (`CHAMPION_AUC_TOLERANCE`).
5. **Small data warning.** 1,000 rows (~250 test) means wide uncertainty. Policy comparison reports bootstrap intervals; treat small differences as noise. Consider repeated CV for model comparison.
6. **No date column**, so a random stratified split is used. In real lending, validate out-of-time.
7. **Fairness.** Sex is only derivable from `personal_status_sex`, and age/foreign-worker status are sensitive. Many jurisdictions restrict their use in credit decisions - the audit measures outcomes; `EXCLUDE_FROM_MODEL` in `config.py` lets you test "unawareness" (proxies can still leak). Reweighing is the tested mitigation; compare its cost/fairness trade-off in `fairness_mitigation_tradeoff.csv`.
8. **Money assumptions** (`MARGIN_RATE`, `LOSS_GIVEN_DEFAULT`) are placeholders consistent with the 5:1 matrix; edit them.
9. **Dataset age.** It is a 1994 German dataset in DM: great for learning, not for production decisions.
10. **Production refit.** The shipped model is trained on the training split only. Before real use, refit on all data with the chosen settings and re-check calibration.

## API
`POST /predict` with the 20 UCI codes, e.g.
```json
{"checking_status":"A11","duration_months":24,"credit_history":"A32","purpose":"A43","credit_amount":3500,
 "savings":"A61","employment_since":"A73","installment_rate":3,"personal_status_sex":"A93","other_debtors":"A101",
 "residence_since":2,"property":"A121","age":35,"other_installment_plans":"A143","housing":"A152",
 "existing_credits":1,"job":"A173","dependents":1,"telephone":"A192","foreign_worker":"A201"}
```
Returns PD, scorecard points, risk band, APPROVE/DECLINE, manual-review flag + reason, segment and its policy, main risk factors (reason codes). `GET /monitor` computes PSI on logged traffic. The prediction log contains applicant data - protect it.

## Next ideas
Monotonic WoE binning (`optbinning`), LightGBM/XGBoost challengers, repeated CV, reject inference (not possible with this data), survival analysis, rate-based profit optimisation per loan, SHAP reason codes in the API, CI/CD with `pytest`.
