"""The orchestrator: one class, one method per project step, results saved to reports/ and artifacts/."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src import schema as S
from src.anomaly import AnomalyDetector
from src.bundle import ModelBundle
from src.cost import CostOptimizer
from src.data import GermanCreditData
from src.eda import EDA
from src.explain import RiskDriverExplainer
from src.fairness import FairnessAuditor, FairnessMitigationExperiment
from src.models import MetricSuite, ModelFactory, ModelTrainer
from src.monitor import DriftMonitor
from src.policy import PolicySimulator
from src.segmentation import CustomerSegmenter
from src.synthetic import make_synthetic_raw
from src.viz import Plotter


class CreditRiskProject:
    def __init__(self, workdir: Path = C.ROOT, data_path: Path = C.DATA_PATH,
                 synthetic: bool = False, run_shap: bool = True, tune: bool = True):
        self.workdir = Path(workdir)
        self.data = GermanCreditData(data_path)
        self.synthetic, self.run_shap, self.tune = synthetic, run_shap, tune
        self.artifacts = self.workdir / "artifacts"
        self.tables = self.workdir / "reports" / "tables"
        self.plot = Plotter(self.workdir / "reports" / "figures")
        for d in (self.artifacts, self.tables):
            d.mkdir(parents=True, exist_ok=True)
        self.cost_opt = CostOptimizer()
        self.factory = ModelFactory()
        self.summary = {}

    # ---- helpers -----------------------------------------------------------
    def _save(self, df: pd.DataFrame, name: str):
        df.to_csv(self.tables / f"{name}.csv", index=False)

    @staticmethod
    def _log(msg: str):
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    # ---- step 1: data ------------------------------------------------------
    def load(self):
        raw = make_synthetic_raw() if self.synthetic else self.data.load_raw()
        df = GermanCreditData.prepare(raw)
        self.train, self.test = GermanCreditData.split(df)
        self.y_train, self.y_test = self.train[S.TARGET].values, self.test[S.TARGET].values
        self._log(f"Loaded {len(df)} rows -> train {len(self.train)}, test {len(self.test)}")

    # ---- step 2: EDA (training data only) -----------------------------------
    def explore(self):
        eda = EDA(self.train)
        self.summary["eda"] = eda.summary()
        self._save(eda.bad_rate_tables(), "eda_bad_rate_by_variable")
        self._save(eda.numeric_describe().reset_index(), "eda_numeric_describe")
        self.plot.target_balance(self.y_train)

    # ---- steps 3/5: models, scorecard, champion -----------------------------
    def build_models(self):
        self.trainer = ModelTrainer(self.factory).fit_candidates(self.train, self.y_train, tune=self.tune)
        cv = self.trainer.cv_table(self.y_train)
        self._save(cv.reset_index().rename(columns={"index": "model"}), "model_comparison_cv_oof")
        self.champion_name = C.CHAMPION_OVERRIDE or self.trainer.pick_champion(self.y_train)
        self.champion = self.trainer.models_[self.champion_name]
        self.summary["champion"] = self.champion_name
        self.summary["gbm_params"] = self.trainer.gbm_params_
        test_tbl = self.trainer.evaluate(self.test, self.y_test)
        self._save(test_tbl.reset_index().rename(columns={"index": "model"}), "model_comparison_test")
        self.test_pd = self.trainer.test_pd_[self.champion_name]
        self.plot.roc(self.y_test, self.trainer.test_pd_)
        self.plot.calibration({n: MetricSuite.calibration_table(self.y_test, p)
                               for n, p in self.trainer.test_pd_.items() if n in ("scorecard", "gbm", "gbm_calibrated")})
        self._log(f"Champion = {self.champion_name}")

    def scorecard_outputs(self):
        sc = self.trainer.models_["scorecard"]
        self._save(sc.binner_.iv_table(), "scorecard_iv_table")
        self._save(sc.points_table_, "scorecard_points_table")
        self.summary["scorecard"] = {"features": sc.features_, "dropped_negative_coef": sc.dropped_negative_,
                                     "factor": sc.factor, "offset": sc.offset}

    # ---- step 4: cost-sensitive cutoff --------------------------------------
    def cost_sensitive(self):
        oof = self.trainer.oof_[self.champion_name]
        self.threshold = self.cost_opt.optimal_threshold(self.y_train, oof)
        curve = self.cost_opt.curve(self.y_train, oof)
        self._save(curve, "cost_curve_train_oof")
        self.plot.cost_curve(curve, self.threshold, self.cost_opt.theoretical_threshold)
        sc = self.trainer.models_["scorecard"]
        sc_oof_scores = sc.pd_to_score(self.trainer.oof_["scorecard"])
        self.score_cutoff = self.cost_opt.optimal_score_cutoff(self.y_train, sc_oof_scores)
        self.plot.score_distribution(sc_oof_scores, self.y_train, self.score_cutoff)
        rows = []
        for label, t in (("default_0.5", 0.5), ("cost_optimal", self.threshold),
                         ("theoretical_1/6", self.cost_opt.theoretical_threshold)):
            rows.append({"rule": label, "threshold": t,
                         "train_oof_cost_per_applicant": self.cost_opt.total_cost(self.y_train, oof < t) / len(oof),
                         "test_cost_per_applicant": self.cost_opt.total_cost(self.y_test, self.test_pd < t) / len(self.test_pd),
                         **{f"test_{k}": v for k, v in self.cost_opt.breakdown(self.y_test, self.test_pd < t).items()}})
        self._save(pd.DataFrame(rows), "cost_threshold_summary")
        self.summary["thresholds"] = {"cost_optimal_pd": self.threshold, "score_cutoff": self.score_cutoff,
                                      "theoretical_pd": self.cost_opt.theoretical_threshold}
        self._log(f"Cost-optimal PD threshold = {self.threshold:.3f} (theory: {self.cost_opt.theoretical_threshold:.3f})")

    # ---- step 6: segmentation -----------------------------------------------
    def segmentation(self):
        self.segmenter = CustomerSegmenter(method="gmm").fit(self.train)
        self._save(self.segmenter.k_table_, "segmentation_k_selection")
        prof = self.segmenter.profile(self.train, self.y_train)
        prof = CustomerSegmenter.suggest_policy(prof, self.threshold)
        self._save(prof, "segment_profile_train")
        self._save(self.segmenter.profile(self.test, self.y_test), "segment_profile_test")
        self.segment_policies = dict(zip(prof["cluster"], prof["suggested_policy"]))
        self._log(f"Segmentation: k = {self.segmenter.k_}")

    # ---- step 9: explainability ---------------------------------------------
    def explain(self):
        odds = RiskDriverExplainer.odds_ratios(self.trainer.models_["logreg"])
        self._save(odds, "odds_ratios")
        self.summary["plain_language_drivers"] = RiskDriverExplainer.plain_language(odds)
        if not self.run_shap:
            return
        try:
            sv, Xt, names = RiskDriverExplainer.shap_values(self.trainer.models_["gbm"], self.train)
        except ImportError:
            self._log("shap not installed -> skipping SHAP (pip install shap)")
            return
        self._save(RiskDriverExplainer.global_importance(sv, names, self.factory.cat_cols), "shap_importance")
        RiskDriverExplainer.plot_summary(sv, Xt, names, self.plot.dir / "shap_summary.png")

    # ---- step 10: fairness --------------------------------------------------
    def fairness(self):
        auditor = FairnessAuditor(cost_opt=self.cost_opt)
        approve = self.test_pd < self.threshold
        for attr, tbl in auditor.audit(self.test, self.y_test, approve).items():
            self._save(tbl, f"fairness_groups_{attr}")
        self._save(auditor.summary(self.test, self.y_test, approve), "fairness_summary")
        exp = FairnessMitigationExperiment(self.trainer, self.factory, self.cost_opt, auditor)
        res = exp.run(self.train, self.y_train, self.test, self.y_test)
        self._save(res, "fairness_mitigation_tradeoff")
        for attr in auditor.attrs:
            self.plot.fairness_tradeoff(res, attr, name=f"fairness_tradeoff_{attr}")

    # ---- step 7: policy comparison -------------------------------------------
    def policies(self):
        sc = self.trainer.models_["scorecard"]
        policies = {"approve_all": np.ones(len(self.test), bool),
                    "cutoff_0.5": self.test_pd < 0.5,
                    "cost_optimal_cutoff": self.test_pd < self.threshold,
                    "score_based_cutoff": sc.score(self.test) >= self.score_cutoff}
        res = PolicySimulator(self.cost_opt).compare(policies, self.y_test, self.test["credit_amount"].values)
        self._save(res, "policy_comparison")
        self.plot.policy_comparison(res)

    # ---- step 8: anomalies ---------------------------------------------------
    def anomalies(self):
        self.anomaly = AnomalyDetector().fit(self.train)
        flags = self.anomaly.score(self.test)
        self._save(AnomalyDetector.evaluate(flags, self.y_test), "anomaly_flag_vs_bad_rate")
        flagged = pd.concat([self.test[["credit_amount", "duration_months", "monthly_payment", "age"]],
                             flags[["anomaly_score", "review_reason"]]], axis=1)
        self._save(flagged[flags["manual_review"]].sort_values("anomaly_score", ascending=False), "anomalies_test")

    # ---- step 12/13: monitoring reference + deployable bundle -----------------
    def package(self):
        self.monitor = DriftMonitor().fit(self.train, self.trainer.oof_[self.champion_name],
                                          self.factory.num_cols, self.factory.cat_cols)
        self._save(self.monitor.report(self.test, self.test_pd), "psi_train_vs_test")
        bundle = ModelBundle(
            champion_name=self.champion_name, champion=self.champion,
            scorecard=self.trainer.models_["scorecard"], threshold=self.threshold,
            score_cutoff=self.score_cutoff, segmenter=self.segmenter,
            segment_policies=self.segment_policies, anomaly=self.anomaly, monitor=self.monitor,
            metadata={"trained_at": datetime.now(timezone.utc).isoformat(), "train_rows": len(self.train),
                      "synthetic_data": self.synthetic, "cost_matrix": [C.COST_FALSE_APPROVE, C.COST_FALSE_REJECT]})
        bundle.save(self.artifacts / "model_bundle.joblib")
        (self.artifacts / "run_summary.json").write_text(json.dumps(self.summary, indent=2, default=float))
        self._log(f"Saved bundle -> {self.artifacts / 'model_bundle.joblib'}")

    def run_all(self):
        for step in (self.load, self.explore, self.build_models, self.scorecard_outputs, self.cost_sensitive,
                     self.segmentation, self.explain, self.fairness, self.policies, self.anomalies, self.package):
            self._log(f"== {step.__name__}")
            step()
        return self
