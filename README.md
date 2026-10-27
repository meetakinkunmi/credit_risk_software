# German Credit Risk: cost-sensitive scoring, audit and deployment

Dashboard url - https://tinyurl.com/credit-risk-software

A class-based credit-risk project built on the UCI Statlog German Credit data (1,000 loans, 20 applicant attributes, about 30% bad loan). This project produces a points-based scorecard, a cost-aware approve/decline rule, a fairness audit, a manual-review flag, drift monitoring, a FastAPI service and a Streamlit dashboard.

> **Status of the results in this file.** Every number below comes from runs on the real `german.data` (800 train rows, 200 test rows, random seed 42) unless marked otherwise. The test set is small, so most differences between models are within noise. This is a 1994 dataset in Deutsche Marks. It is a learning and method-demonstration project, not a model to make real lending decisions with.

---

## 1. Problem statement

**The cost of an error is lopsided, and the data shows it.** The dataset documentation assigns a cost of 5 to approving loan to customer which turns out bad and 1 to rejecting loan to customer who would have repaid. On the 200-row test set, this produced the following cost per applicant:

| Policy | Cost per applicant |
|---|---|
| Approve everyone | 1.500 |
| Reject everyone | 0.700 |
| Scorecard approving PD < 1/6 | 0.545 |
| Scorecard approving PD < 0.50 | 0.805 |

A conventional "approve if the model says more likely good than bad" cutoff of 0.5 costs more than rejecting everyone. A model can have a good AUC and still lose money if the cutoff ignores the 5:1 asymmetry.

**A lender has to explain the decision, and the explanation has to come from the model that decided it.** Initially I used the gradient boosting model to produce the PD and the decision, while the scorecard produced the points and the reason. But on a real applicant the boosting model gave PD 0.139 and approved, while the scorecard gave PD 0.286 and would have declined. The reasons shown to that applicant would have described a different model from the one that approved loan.

**A protected attribute can reach the customer through a reason code.** "Foreign worker: yes" appeared as the third reason (26.2 points lost) for the riskiest applicant. Only 31 of 800 training applicants are "no", so that bin was merged into `OTHER`, and the reason really means "you are not in a 31-person group".

**Small data hides real uncertainty.** Model rankings flipped between cross-validation (GBM best) and the hold-out set (logistic regression best), and the champion changed at an AUC tolerance of 0.013. A pipeline that reports single numbers on 200 rows invites decisions on noise.

**The data cannot show who would have been good among the rejected.** Only approved loans are in the file. The probable consequence is that "critical / credits elsewhere" credit history shows a lower default rate than clean histories, and the models learn that pattern (odds ratio 0.50). It reflects who was approved, not what causes repayment.

## 2. How this project addresses it

| Problem | What the project does | Where |
|---|---|---|
| Lopsided costs | Derives the break-even PD from the cost matrix (5 / (5+1) gives approve below 1/6), then compares that with an empirically tuned cutoff on out-of-fold training predictions | `cost.py`, `pipeline.cost_sensitive` |
| Explanation must match the decision | The scorecard is the champion (`CHAMPION_OVERRIDE`), so PD, points and reason codes all come from one model | `config.py`, `pipeline.py`, `scorecard.py` |
| Protected attributes | `foreign_worker` and `personal_status_sex` are excluded from every model through `EXCLUDE_FROM_MODEL`, but stay in the data for the audit | `config.py`, `models.py`, `fairness.py` |
| Small-sample noise | Bootstrap intervals on cost, paired comparisons, out-of-fold selection, and a test set used once | `policy.py`, `models.py` |
| Unusual applications | Isolation Forest plus three rules route applicants to manual review | `anomaly.py` |
| Models going stale | PSI against an out-of-fold training reference, on inputs and on PDs | `monitor.py` |
| Selection bias in credit history | Not fixable with this data. Documented, and the pattern is left visible for reviewers | Section 8 |

## 3. Objectives

Each objective can be checked against a file in `reports/tables/` after a run.

1. **Choose the approve/decline cutoff from the cost matrix, not from convention.** Report cost at 0.5, at the tuned cutoff and at 1/6 (`cost_threshold_summary.csv`).
2. **Beat both naive baselines on cost.** The scorecard at 1/6 cost 0.545 per applicant against 0.700 for rejecting everyone, about 22% lower, and against 1.500 for approving everyone.
3. **Produce one consistent decision record per applicant.** PD, points, risk band, decision, review flag, segment and the three largest point losses, all from one model, in a single `ModelBundle.score()` call.
4. **Keep protected attributes out of the model and out of reason codes, while still auditing outcomes by sex, age group and foreign-worker status.**
5. **Measure the uncertainty of every comparison** instead of ranking on a single 200-row score.
6. **Detect distribution shift** and know the noise floor of the detector (PSI up to about 0.07 on 200 rows from a pure random split).
7. **Be re-runnable.** The full pipeline ran in about 31 seconds on the real data, and eight unit tests pass in about 6 seconds.

**Not objectives.** Legal compliance in any jurisdiction, and causal claims about why applicants default.

---

## 4. Setup

Use one environment and call tools through `python -m`, so the tool always comes from the same environment as `python`. (During this project the notebook and the terminal used different environments, and `streamlit` was "not found" in the terminal.)

```bash
python -m venv .venv
.venv\Scripts\activate                   # Windows. Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt httpx
```

The bundle in this project was trained with **Python 3.12.12 and scikit-learn 1.8.0**. A pickled model only loads reliably under the same library versions. Loading it under scikit-learn 1.9.1 raised `InconsistentVersionWarning` and then `ModuleNotFoundError: No module named '_loss'`.

## 5. Running it

```bash
python run_pipeline.py --synthetic --no-shap --no-tune   # 1) wiring test on FAKE data, writes to .smoke/, ignore numbers
python -m pytest -q                                      # 2) 8 unit tests
python run_pipeline.py --download                        # 3) real run: downloads german.data, trains, writes reports/ and artifacts/
python -m uvicorn app.api:app --port 8000                # 4) API at http://localhost:8000/docs
python -m streamlit run app/dashboard.py                 # 5) dashboard at http://localhost:8501
```

`--download` overwrites an existing `data/german.data`. If the download fails, put the UCI file named `german.data` (not `german.data-numeric`) in `data/`.

Flags: `--no-shap` skips SHAP, `--no-tune` skips the GBM grid search, `--synthetic` uses fake data.

**Docker is verified.** Use `python:3.12-slim`, an API-only `requirements-docker.txt` with exact pins (pandas, numpy, scikit-learn==1.8.0, scipy, joblib, shap, fastapi, uvicorn, pydantic), and pip's `--default-timeout=120 --retries 10`. The dashboard service additionally needs `streamlit` and `matplotlib`.

---

## 6. Step-by-step: what each file does, and what it showed

Files are listed in build order. Each depends only on files above it.

### Stage 1. Foundations

**`config.py`.** The control panel: test size 0.20, 5 folds, seed 42, cost 5 and 1, margin 10% and loss-given-default 50% (ratio 5, consistent with the cost matrix), scorecard scale (600 points at 50:1 odds, 20 points to double the odds), risk band edges 0.10 / 0.20 / 0.35 / 0.55 with labels A to E, protected attributes, anomaly contamination of 5%, and `CHAMPION_AUC_TOLERANCE` of 0.01. After review, these were added: `EXCLUDE_FROM_MODEL = ["foreign_worker", "personal_status_sex"]` and `CHAMPION_OVERRIDE = "scorecard"`.

**`schema.py`.** The data dictionary in code: column names, code-to-label maps (`A11` becomes `< 0 DM`), column groups, the target definition and the age bins. Raw target 1 = good, 2 = bad is flipped to `default` (1 = bad). Check run on the real schema: 13 categorical and 7 numeric columns, every feature has a display name, and all `ORDER_*` labels exist in `CODE_MAPS`.

**`features.py`.** `FeatureEngineer` adds `monthly_payment` (amount divided by duration, duration clipped to at least 1) and `log_credit_amount`. It learns nothing from data, so it cannot leak.

**`data.py`.** `GermanCreditData` downloads, loads and checks the file (must be exactly 1,000 x 21), decodes codes and rejects unknown ones, derives `sex` from `personal_status_sex`, builds `age_group` with `right=False` (so ages 25, 35 and 50 fall in `25-34`, `35-49`, `50+`), builds the target, and splits 80/20 stratified. Train and test both have about 30% bad.

**`synthetic.py`.** Fake data with the same schema and a planted signal (checking status, duration, amount, credit history, age). For wiring tests only.

**`eda.py`.** Summary, bad rate per variable level, numeric description with skew.

Real-data findings from this stage:
- `purpose` has levels with 5 (retraining), 6 (others) and 10 (domestic appliances) training rows, and `education` has 40, exactly 5.0% of train.
- The real file has no applicant coded `vacation` (purpose) or `female single` (personal status). The model therefore sees 63 one-hot columns, and the derived `sex = female` covers only one marital category, so any audit by sex mixes sex with marital status.

### Stage 2. Modelling

**`scorecard.py`.** Weight-of-evidence binning, Information Value, correlation filter, logistic regression on WoE, and a points table.
- WoE is the log of a bin's share of goods over its share of bads. Positive means safer than average. IV sums that over a variable's bins into one strength score.
- Numeric variables with more than 6 distinct values get 5 quantile bins. Categories under 5% of rows merge into `OTHER`. Bins are learned on train only.
- Variables with IV under 0.02 are dropped. Of two WoE features correlated above 0.70, the weaker is dropped. Any negative coefficient removes a variable and refits.
- Points sum exactly to the score (tested), and `reason_codes` lists where an applicant lost the most points against the best bin of each variable.
- Limits: bins are not forced to be monotonic, and "best bin" can be a tiny group (see the foreign-worker case in Section 2).

**`cost.py`.** `CostOptimizer` counts false approvals and false rejections, computes cost over a grid of cutoffs, and picks the cheapest one (median of ties). The theoretical cutoff is 1/6 = 0.167.
- On the scorecard alone, a cutoff tuned on train (0.195) was cheaper than 1/6 on train but more expensive on test (113 against 109 cost units). Across 200 bootstrap resamples of the test set, 1/6 won 142 times, tuned won 45, 13 tied. The 95% range of the cost difference included zero (-9 to +20), so the honest reading is "theory is probably slightly better, not certainly".
- For the full pipeline run with the GBM champion, the tuned cutoff was 0.160, close to 1/6.

**`models.py`.** Three models behind one interface (`fit`, `predict_pd`): the scorecard, a logistic regression, and a gradient boosting model with Platt calibration. Plus metrics (AUC, Gini, KS, PR-AUC, Brier, log loss), calibration tables, out-of-fold prediction, GBM tuning, and the champion rule.

| Model | CV AUC (800 rows, out-of-fold) | Test AUC (200 rows) |
|---|---|---|
| Scorecard | 0.772 | 0.796 |
| Logistic regression | 0.774 | 0.817 |
| GBM | 0.785 | 0.806 |
| GBM calibrated | 0.785 | 0.806 |

- In-sample AUC for the GBM was 0.902 against 0.785 out-of-fold. Models are compared only on out-of-fold predictions.
- Bootstrap of the GBM minus scorecard CV AUC gap: mean +0.013, 95% range -0.008 to +0.030, GBM better in 89% of resamples. The default rule picked the GBM because the scorecard missed the 0.01 tolerance by 0.003, and it flips to the scorecard at a tolerance of 0.013.
- Calibration (test, 5 bands of 40): the scorecard's predicted and observed bad rates were close in bands 2 to 5 (e.g. 0.239 against 0.200, 0.681 against 0.725), with the lowest band underestimating risk (0.047 against 0.100, 4 bad of 40).

### Stage 3. Business and risk layers

**`policy.py`.** `PolicySimulator` scores a list of approve/decline decisions by matrix cost, by profit (10% of amount on good loans, 50% loss on bad loans) and with bootstrap intervals. Test-set results for the scorecard policy at 1/6 (before the model changes below):

| Policy | Approved | Bad rate among approved | Cost per applicant | Profit per applicant (DM) |
|---|---|---|---|---|
| Approve everyone | 100% | 30.0% | 1.500 | -480 |
| Reject everyone | 0% | n/a | 0.700 | 0 |
| Scorecard at 1/6 | 39.5% | 10.1% | 0.545 | +55 |

Cutoff sweep for the scorecard: cost per applicant was 0.575 / 0.545 / 0.540 / 0.600 / 0.805 at 0.10 / 0.167 / 0.25 / 0.35 / 0.50, and profit was +41 / +55 / +3 / -13 / -96. Cost is flat between 0.167 and 0.25, while profit falls sharply, because profit weights loan size. The sweep used the test set and shows the shape of the curve, not a place to pick the final cutoff.

**`anomaly.py`.** Isolation Forest on six numeric features plus three rules (large amount over short duration, top 5% implied monthly payment, age 23 or under with a top-quartile loan). Thresholds are learned on train.
- Rule counts on train: payment 40, outlier 40, young-large 9, large-short 3 (the last is almost dead).
- Flagged applicants default more: 44.1% (68 applicants) against 28.7% on train, 41.7% (24) against 28.4% on test.
- Among applicants the scorecard approves (out-of-fold, train), flagged ones defaulted 15.4% (n=26, 4 defaults) against 9.4% (n=287). Suggestive, not statistically established.

**`monitor.py`.** PSI per feature and for PDs against a training reference. Rules of thumb: under 0.10 stable, 0.10 to 0.25 moderate, over 0.25 significant.
- Same-population baseline (train against test): all stable, highest 0.073 (`purpose`, `age`). On 200 rows, PSI up to about 0.07 is noise.
- Test with every loan amount multiplied by 1.5: `monthly_payment` 0.693 and `credit_amount` 0.497 (both significant), but `predicted_pd` only 0.088. Watch inputs as well as scores.

**`segmentation.py`.** Clusters applicants (Gaussian mixture, k chosen by BIC), profiles each cluster, and maps its bad rate to a suggested policy using the cutoff.
- k = 6 on the real run. Train cluster bad rates ranged from 11% to 46%, driven mainly by checking and savings status.
- **Not stable.** Silhouette is about 0.11 for every k, and with other seeds the chosen k was 7 with agreement (adjusted Rand index) of 0.80, 0.69 and 0.43 against the seed-42 clusters. The BIC fall from 12,159 (k=2) to -7,272 (k=6) is unexplained. Do not let segment policies drive decisions.

**`explain.py`.** Odds ratios from the logistic regression and SHAP values for the GBM.
- Largest logistic-regression effects: no checking account 0.44, critical / credits elsewhere 0.50, education 1.88, overdrawn account 1.84.
- SHAP values are additive (checked). For one-hot columns, a positive SHAP value can mean "lacks a protective level", not "has a risky one": the riskiest test applicant had no "no checking account" status and received +0.43 from that column.
- Odds ratios for rare levels (education, 40 rows) are unstable: the coefficient moved from 0.63 to 1.06 when shrinkage was relaxed.

**`fairness.py`.** `FairnessAuditor` (approval rate, approval among good, approval among bad, disparate impact, four-fifths flag by group), `ReweighingMitigator` (weights that make the label independent of the group in training), and an experiment comparing variants. Group tables mark groups under 30 as unreliable.
- Before the model changes (scorecard at 1/6, test): age disparate impact 0.505 (under-25s approved 25.6% against 50.8% for 35 to 49; bad rates 48.7% against 20.6%), sex 0.672, foreign worker not interpretable (6 applicants in "no").
- Reweighing on one attribute improved that attribute and tended to worsen another.

**`viz.py`.** Seven plots saved as PNG files (class balance, ROC, calibration, cost curve, score distribution, policy comparison, fairness trade-off).

### Stage 4. Packaging

**`bundle.py`.** `ModelBundle` holds the champion, scorecard, cutoffs, segmenter, anomaly detector and monitor in one pickle. `score(raw)` takes the 20 UCI-coded fields and returns PD, points, risk band, decision, manual-review flag and reason, segment and its policy, `final_action`, and the top three risk factors. A manual-review flag overrides both APPROVE and DECLINE. Risk bands are descriptive: the approve cutoff (about 0.16) falls inside band B (0.10 to 0.20), so one band contains approvals and declines. Unknown codes raise a clear `ValueError` (checked: `housing = A999` is rejected with the allowed codes listed). Save and reload gives identical PDs (checked).

**`pipeline.py`.** `CreditRiskProject` runs load, explore, build_models, scorecard_outputs, cost_sensitive, segmentation, explain, fairness, policies, anomalies, package, in that order, and each step saves its tables and figures. The bundle's cutoff is the tuned out-of-fold cutoff of the champion. The monitor reference uses out-of-fold PDs.

**`run_pipeline.py`.** Command-line wrapper. `--synthetic` writes only to `.smoke/`.

### Stage 5. Tests

`pytest.ini` puts the project root on the import path. `tests/test_core.py` holds 8 tests on synthetic data (data preparation, bad-code rejection, WoE and IV, scorecard points sum to score, 1/6 cutoff, cost optimiser, reweighing, PSI). Result: 8 passed in 6.43 s. They prove the code runs, not that the model is good. Models, policy, anomaly, segmentation, explain, bundle, pipeline, viz and the API have no tests.

### Stage 6. Deployment

**`app/api.py`.** FastAPI: `/health`, `/model-info`, `/predict`, `/predict/batch` (1 to 1,000 rows), `/monitor` (PSI of logged traffic, needs at least 100 logged rows). Pydantic checks types and ranges (422 on failure) and the bundle checks the codes. Every prediction is appended to `artifacts/prediction_log.jsonl` with the raw applicant fields. Protect that file. **The API endpoints were not run in this review.**

**`app/dashboard.py`.** Seven tabs showing the saved tables and figures, plus a scoring form. The file names it reads match what the pipeline writes. **The dashboard was not confirmed running in this review.**

**`Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.gitignore`.** See the Docker note in Section 5. `.dockerignore` and `.gitignore` should also list `.venv`, `.git`, `.pytest_cache`, `.ipynb_checkpoints`, `*.ipynb` (for Docker), `.env` and `*.joblib`. Clear notebook outputs before committing.

---

## 7. Changes made after review

| Change | Reason | Evidence |
|---|---|---|
| `EXCLUDE_FROM_MODEL = ["foreign_worker", "personal_status_sex"]` | Remove protected attributes from every model | The old scorecard exposed "Foreign worker: yes" as a reason code |
| `CHAMPION_OVERRIDE = "scorecard"`, read in `pipeline.build_models` | The champion rule was decided by 0.003 AUC, and the scorecard's own score and reasons describe its own decision | GBM advantage was not statistically reliable |

Result on the 200-row test set at a fixed cutoff of 1/6:

| Model | AUC | Approval rate | Cost per applicant |
|---|---|---|---|
| Before: GBM calibrated | 0.796 | 0.360 | 0.580 |
| After: scorecard without the two columns | 0.796 | 0.375 | 0.565 |

Disparate impact, before to after: sex 0.760 to 0.829, age group 0.431 to 0.505, foreign worker not interpretable (6 applicants). Differences of this size on 200 rows are within noise. The new reason codes show checking balance, loan duration and savings, and no foreign-worker reason.

## 8. Known limitations and open decisions

- **Selection bias.** Only approved loans are in the data. "Critical / credits elsewhere" credit history lowers predicted risk (odds ratio 0.50). Reviewers should read this as a feature of who was approved, not as a cause.
- **Proxies.** Removing a column does not remove its signal. Age, marital status and employment can carry it.
- **Sex audit confounded.** No "female single" applicants exist in the file.
- **Tiny groups.** Foreign worker "no" has 6 test applicants.
- **200 test rows.** Report intervals, not single numbers. Reported figures from the earlier model changes carry no confidence intervals.
- **No date column.** The split is random, not out-of-time.
- **Profit model is a toy.** It ignores loan term, interest timing and recoveries.
- **Segments are unstable** (see Section 6, `segmentation.py`).
- **Manual review overrides declines too,** so the review queue (about 8.5% of train and 12% of test flagged) is larger than the number of risky approvals (only 6 of 24 flagged test applicants would have been approved).
- **Environment.** `requirements.txt` has lower bounds only and no `httpx`. Pin versions and record them in the bundle metadata.

## 9. Project layout

```
credit_risk_project/
  requirements.txt  pytest.ini  run_pipeline.py  Dockerfile  docker-compose.yml  README.md
  data/        german.data (downloaded)
  src/         config schema features data synthetic eda scorecard cost models policy
               anomaly monitor segmentation explain fairness viz bundle pipeline
  app/         api.py  dashboard.py
  tests/       test_core.py
  artifacts/   model_bundle.joblib  run_summary.json  prediction_log.jsonl   (generated)
  reports/     tables/*.csv  figures/*.png                                    (generated)
```
