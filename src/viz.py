"""Plots. Every method saves a PNG and returns its path (kept separate from the logic classes)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


class Plotter:
    def __init__(self, fig_dir: Path):
        self.dir = Path(fig_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        p = self.dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        return p

    def target_balance(self, y, name="eda_target"):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["good", "bad"], [(np.asarray(y) == 0).sum(), (np.asarray(y) == 1).sum()])
        ax.set_title("Class balance (training set)")
        return self._save(fig, name)

    def roc(self, y, preds: dict, name="roc_test"):
        fig, ax = plt.subplots(figsize=(5, 4))
        for n, p in preds.items():
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, label=n)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (hold-out)")
        ax.legend()
        return self._save(fig, name)

    def calibration(self, tables: dict, name="calibration_test"):
        fig, ax = plt.subplots(figsize=(5, 4))
        for n, t in tables.items():
            ax.plot(t["mean_predicted"], t["observed_bad_rate"], marker="o", label=n)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="Mean predicted PD", ylabel="Observed bad rate", title="Calibration (hold-out)")
        ax.legend()
        return self._save(fig, name)

    def cost_curve(self, curve, thr_opt, thr_theory, name="cost_curve"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(curve["threshold"], curve["total_cost"], label="total cost")
        ax.plot(curve["threshold"], curve["cost_false_approvals"], ls=":", label="cost of false approvals")
        ax.plot(curve["threshold"], curve["cost_false_rejections"], ls=":", label="cost of false rejections")
        ax.axvline(0.5, color="grey", ls="--", label="default 0.5")
        ax.axvline(thr_opt, color="green", label=f"empirical optimum {thr_opt:.3f}")
        ax.axvline(thr_theory, color="red", ls="--", label=f"theoretical {thr_theory:.3f}")
        ax.set(xlabel="Approve if PD < threshold", ylabel="Cost (training OOF)", title="Cost-sensitive cutoff")
        ax.legend(fontsize=7)
        return self._save(fig, name)

    def score_distribution(self, scores, y, cutoff=None, name="score_distribution"):
        s, y = np.asarray(scores), np.asarray(y)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s[y == 0], bins=25, alpha=0.6, label="good")
        ax.hist(s[y == 1], bins=25, alpha=0.6, label="bad")
        if cutoff is not None:
            ax.axvline(cutoff, color="k", ls="--", label=f"cost-optimal cutoff {cutoff:.0f}")
        ax.set(xlabel="Scorecard points", title="Score distribution (training OOF)")
        ax.legend()
        return self._save(fig, name)

    def policy_comparison(self, df, name="policy_comparison"):
        fig, ax = plt.subplots(figsize=(6, 4))
        err = [df["matrix_cost_per_applicant"] - df["cost_ci_low"], df["cost_ci_high"] - df["matrix_cost_per_applicant"]]
        ax.bar(df["policy"], df["matrix_cost_per_applicant"], yerr=err, capsize=4)
        ax.set(ylabel="Cost per applicant (5:1 matrix)", title="Policy comparison (hold-out, 95% bootstrap CI)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        return self._save(fig, name)

    def fairness_tradeoff(self, df, attr, name="fairness_tradeoff"):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df[f"DI_{attr}"], df["cost_per_applicant"])
        for _, r in df.iterrows():
            ax.annotate(r["variant"], (r[f"DI_{attr}"], r["cost_per_applicant"]), fontsize=6)
        ax.axvline(0.8, color="red", ls="--", lw=0.8)
        ax.set(xlabel=f"Disparate impact ratio ({attr})", ylabel="Cost per applicant",
               title="Accuracy (cost) vs fairness")
        return self._save(fig, name)
