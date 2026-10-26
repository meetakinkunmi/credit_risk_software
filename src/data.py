"""Step 1: loading, decoding, deriving audit columns and splitting."""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import schema as S
from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE
from src.features import FeatureEngineer

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"


class GermanCreditData:
    """Owns everything between the raw `german.data` file and model-ready frames."""

    def __init__(self, path: Path = DATA_PATH):
        self.path = Path(path)

    # ---- acquisition -------------------------------------------------------
    def download(self) -> None:
        """Fetch german.data from the UCI repository (needs internet access)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        self.path.write_bytes(archive.read("german.data"))

    def load_raw(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run `python run_pipeline.py --download` or place the "
                "UCI file `german.data` there manually.")
        raw = pd.read_csv(self.path, sep=r"\s+", header=None, names=S.RAW_COLUMNS)
        if raw.shape != (1000, 21):
            raise ValueError(f"Unexpected shape {raw.shape}; expected (1000, 21).")
        return raw

    # ---- preparation (also used at inference time) -------------------------
    @staticmethod
    def decode(raw: pd.DataFrame) -> pd.DataFrame:
        """Replace UCI codes (A11, A12, ...) by readable labels. Raises on unknown codes."""
        df = raw.copy()
        missing = [c for c in S.COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df["sex"] = df["personal_status_sex"].map(S.SEX_FROM_STATUS)
        for col, mapping in S.CODE_MAPS.items():
            unknown = set(df[col].astype(str).unique()) - set(mapping)
            if unknown:
                raise ValueError(f"Column '{col}' has invalid codes {sorted(unknown)}; "
                                 f"allowed: {sorted(mapping)}")
            df[col] = df[col].astype(str).map(mapping)
        return df

    @classmethod
    def prepare(cls, raw: pd.DataFrame) -> pd.DataFrame:
        """raw codes -> decoded + derived audit columns + engineered features (+ target)."""
        df = cls.decode(raw)
        df["age_group"] = pd.cut(df["age"], S.AGE_BINS, labels=S.AGE_LABELS, right=False).astype(str)
        df = FeatureEngineer().transform(df)
        if "target" in df.columns:
            df[S.TARGET] = (df["target"] == 2).astype(int)  # 1 = bad
            df = df.drop(columns="target")
        return df.reset_index(drop=True)

    # ---- splitting ---------------------------------------------------------
    @staticmethod
    def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE):
        """Stratified split. The dataset has no date column, so a time-based split is impossible;
        in real projects prefer an out-of-time split."""
        train, test = train_test_split(df, test_size=test_size, stratify=df[S.TARGET], random_state=seed)
        return train.reset_index(drop=True), test.reset_index(drop=True)
