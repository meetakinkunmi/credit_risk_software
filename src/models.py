"""Step 5: model zoo, metrics, cross-validation and champion selection.

Every model exposes the same two methods:  fit(X, y_bad[, sample_weight])  and  predict_pd(X).
X is always the FULL frame; each model picks the columns it needs, so audit columns can travel along."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src import schema as S
from src.scorecard import Scorecard


class SklearnPDModel:
    """Adapter: sklearn Pipeline(prep, clf) -> fit / predict_pd."""

    def __init__(self, name: str, pipeline: Pipeline):
        self.name, self.pipeline = name, pipeline

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.pipeline.fit(X, y)
        else:
            self.pipeline.fit(X, y, clf__sample_weight=np.asarray(sample_weight))
        return self

    def predict_pd(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


class PlattCalibrator:
    """Sigmoid calibration on the logit of the raw probability."""

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(p), y)
        return self

    @staticmethod
    def _z(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def transform(self, p):
        return self.lr_.predict_proba(self._z(p))[:, 1]


class CalibratedModel:
    """Wraps a fitted model + calibrator. Calibrated PDs matter because the cost-optimal
    threshold (1/6 for a 5:1 matrix) assumes probabilities are real probabilities."""

    def __init__(self, name, base, calibrator):
        self.name, self.base, self.calibrator = name, base, calibrator

    def predict_pd(self, X):
        return self.calibrator.transform(self.base.predict_pd(X))


class ModelFactory:
    """Builds fresh (unfitted) models, honouring config.EXCLUDE_FROM_MODEL."""

    def __init__(self, exclude=None, seed=C.RANDOM_STATE):
        ex = set(C.EXCLUDE_FROM_MODEL if exclude is None else exclude)
        self.cat_cols = [c for c in S.CATEGORICAL_COLS if c not in ex]
        self.num_cols = [c for c in S.NUMERIC_COLS + S.ENGINEERED_COLS if c not in ex]
        self.seed = seed

    def _prep(self):
        return ColumnTransformer([
            ("num", StandardScaler(), self.num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)])

    def scorecard(self) -> Scorecard:
        cols = [c for c in S.CATEGORICAL_COLS + S.NUMERIC_COLS + ["monthly_payment"]
                if c in self.cat_cols + self.num_cols]
        return Scorecard(columns=cols)

    def logreg(self, C_reg=0.3) -> SklearnPDModel:
        clf = LogisticRegression(C=C_reg, max_iter=3000)
        return SklearnPDModel("logreg", Pipeline([("prep", self._prep()), ("clf", clf)]))

    def gbm(self, **params) -> SklearnPDModel:
        p = dict(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=self.seed)
        p.update(params)
        return SklearnPDModel("gbm", Pipeline([("prep", self._prep()), ("clf", GradientBoostingClassifier(**p))]))


class MetricSuite:
    """Discrimination, ranking and calibration metrics."""

    @staticmethod
    def ks(y, p) -> float:
        fpr, tpr, _ = roc_curve(y, p)
        return float(np.max(tpr - fpr))

    @classmethod
    def summarize(cls, y, p) -> dict:
        auc = roc_auc_score(y, p)
        return {"auc": auc, "gini": 2 * auc - 1, "ks": cls.ks(y, p),
                "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
                "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}

    @staticmethod
    def calibration_table(y, p, n_bins=10) -> pd.DataFrame:
        d = pd.DataFrame({"p": np.asarray(p), "y": np.asarray(y)})
        d["bin"] = pd.qcut(d["p"], n_bins, duplicates="drop")
        t = d.groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                                observed_bad_rate=("y", "mean")).reset_index(drop=True)
        return t


class ModelTrainer:
    """Cross-validation, tuning, calibration and champion selection (training data only)."""

    def __init__(self, factory: ModelFactory, n_folds=C.N_FOLDS, seed=C.RANDOM_STATE):
        self.factory, self.n_folds, self.seed = factory, n_folds, seed
        self.models_, self.oof_ = {}, {}

    def oof_predict(self, make_model, X: pd.DataFrame, y, weight_fn=None) -> np.ndarray:
        """Out-of-fold PDs: every row is predicted by a model that never saw it."""
        y = np.asarray(y)
        oof = np.zeros(len(y))
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(X, y):
            m = make_model()
            Xtr = X.iloc[tr]
            w = weight_fn(Xtr, y[tr]) if weight_fn is not None else None
            m.fit(Xtr, y[tr], sample_weight=w)
            oof[va] = m.predict_pd(X.iloc[va])
        return oof

    def tune_gbm(self, X, y) -> dict:
        grid = {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.08]}
        gs = GridSearchCV(self.factory.gbm().pipeline, grid, scoring="roc_auc",
                          cv=StratifiedKFold(3, shuffle=True, random_state=self.seed), n_jobs=1)
        gs.fit(X, y)
        return {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}

    def fit_candidates(self, X: pd.DataFrame, y, tune=True):
        y = np.asarray(y)
        self.gbm_params_ = self.tune_gbm(X, y) if tune else {}
        makers = {"scorecard": self.factory.scorecard, "logreg": self.factory.logreg,
                  "gbm": lambda: self.factory.gbm(**self.gbm_params_)}
        for name, make in makers.items():
            self.oof_[name] = self.oof_predict(make, X, y)
            self.models_[name] = make().fit(X, y)
        cal = PlattCalibrator().fit(self.oof_["gbm"], y)
        self.models_["gbm_calibrated"] = CalibratedModel("gbm_calibrated", self.models_["gbm"], cal)
        self.oof_["gbm_calibrated"] = cal.transform(self.oof_["gbm"])
        return self

    def cv_table(self, y) -> pd.DataFrame:
        return pd.DataFrame({n: MetricSuite.summarize(y, p) for n, p in self.oof_.items()}).T

    def pick_champion(self, y, tol=C.CHAMPION_AUC_TOLERANCE) -> str:
        """Best CV AUC wins, but the interpretable scorecard is preferred if within `tol`."""
        aucs = {n: roc_auc_score(y, self.oof_[n]) for n in ("scorecard", "logreg", "gbm")}
        best = max(aucs, key=aucs.get)
        if aucs["scorecard"] >= aucs[best] - tol:
            return "scorecard"
        return "gbm_calibrated" if best == "gbm" else best

    def evaluate(self, X_test, y_test) -> pd.DataFrame:
        self.test_pd_ = {n: m.predict_pd(X_test) for n, m in self.models_.items()}
        return pd.DataFrame({n: MetricSuite.summarize(y_test, p) for n, p in self.test_pd_.items()}).T
