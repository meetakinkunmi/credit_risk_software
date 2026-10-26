"""Central configuration. Change values here, not inside the classes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "german.data"

RANDOM_STATE = 42
TEST_SIZE = 0.20          # stratified hold-out, touched ONLY for final evaluation
N_FOLDS = 5               # cross-validation folds on the training set

# ---- 5:1 cost matrix from the dataset documentation -------------------------
COST_FALSE_APPROVE = 5.0  # approve a customer who turns out BAD
COST_FALSE_REJECT = 1.0   # reject a customer who would have been GOOD

# ---- Money assumptions for the profit simulation---------------
# Chosen so that LGD / margin = 5, consistent with the 5:1 matrix.
MARGIN_RATE = 0.10        # profit on a good loan, as a fraction of loan amount
LOSS_GIVEN_DEFAULT = 0.50 # loss on a bad loan, as a fraction of loan amount

# ---- Scorecard --------------------------------------------------------------
IV_MIN = 0.02             # drop variables with Information Value below this
N_BINS = 5                # quantile bins for numeric variables
MIN_BIN_FRACTION = 0.05   # rarer categories are merged into "OTHER"
CORR_MAX = 0.70           # drop one of two WoE features correlated above this
BASE_SCORE = 600          # score at the base odds
BASE_ODDS = 50            # good:bad odds at BASE_SCORE
PDO = 20                  # points to double the odds

# ---- Risk bands (upper PD edge of each band) --------------------------------
RISK_BAND_EDGES = [0.10, 0.20, 0.35, 0.55]
RISK_BAND_LABELS = ["A", "B", "C", "D", "E"]

# ---- Fairness ---------------------------------------------------------------
PROTECTED_ATTRS = ["age_group", "sex", "foreign_worker"]
# "Fairness through unawareness" experiment: list columns to REMOVE from the
# models. Note sex is encoded inside personal_status_sex, so the column
# will not be exposed to the model.
EXCLUDE_FROM_MODEL: list = []

# ---- Anomaly detection ------------------------------------------------------
ANOMALY_CONTAMINATION = 0.05

# ---- Champion selection -----------------------------------------------------
# Prefer the interpretable scorecard if its CV AUC is within this margin of the best.
CHAMPION_AUC_TOLERANCE = 0.01
