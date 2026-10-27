"""Step 9: explain risk drivers with odds ratios (logistic regression) and SHAP (gradient boosting)."""
import numpy as np
import pandas as pd

from src.schema import FRIENDLY


class RiskDriverExplainer:
    @staticmethod
    def odds_ratios(logreg_model) -> pd.DataFrame:
        """exp(coef) on the one-hot / standardised design. Numeric rows: odds multiplier per +1 SD.
        Category rows: odds multiplier relative to the AVERAGE applicant (L2 shrinkage, no reference level)."""
        prep, clf = logreg_model.pipeline.named_steps["prep"], logreg_model.pipeline.named_steps["clf"]
        names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
        coef = clf.coef_[0]
        t = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef),
                          "effect": np.where(coef > 0, "raises default risk", "lowers default risk")})
        return t.reindex(t["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    @staticmethod
    def plain_language(odds: pd.DataFrame, top: int = 10) -> list:
        lines = []
        for _, r in odds.head(top).iterrows():
            lines.append(f"{r['feature']}: odds of default x{r['odds_ratio']:.2f} ({r['effect']}).")
        return lines

    @staticmethod
    def shap_values(gbm_model, X: pd.DataFrame):
        """Returns (shap matrix in log-odds of default, transformed matrix, feature names)."""
        import shap  # imported lazily: heavy, and only needed offline
        prep, clf = gbm_model.pipeline.named_steps["prep"], gbm_model.pipeline.named_steps["clf"]
        Xt = prep.transform(X)
        sv = shap.TreeExplainer(clf).shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv, Xt, [n.split("__", 1)[1] for n in prep.get_feature_names_out()]

    @staticmethod
    def global_importance(sv: np.ndarray, names: list, cat_cols: list) -> pd.DataFrame:
        """Mean |SHAP| summed over one-hot columns, so each ORIGINAL variable gets one number."""
        groups = {}
        for i, n in enumerate(names):
            owner = next((c for c in sorted(cat_cols, key=len, reverse=True) if n.startswith(c + "_")), n)
            groups.setdefault(owner, []).append(i)
        imp = {FRIENDLY.get(k, k): float(np.abs(sv[:, idx]).sum(axis=1).mean()) for k, idx in groups.items()}
        return pd.Series(imp).sort_values(ascending=False).rename("mean_abs_shap").reset_index().rename(
            columns={"index": "variable"})

    @staticmethod
    def plot_summary(sv, Xt, names, path):
        import matplotlib.pyplot as plt
        import shap
        shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
