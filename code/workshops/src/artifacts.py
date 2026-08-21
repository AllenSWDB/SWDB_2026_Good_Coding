"""Filesystem paths and small save/load helpers for pipeline artifacts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path('/data/409828_V1DD_Filtered')
EM_DATA_PATH = Path('/data/v1dd_1196')

STEP1_DIR = Path('/data/intermediates/v1dd_step1')   # single-ROI Step 1 artifacts
STEP2_DIR = Path('/data/intermediates/v1dd_step2')   # batch tuning + metrics artifacts


def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_table(df, path):
    """Write a dataframe to parquet, creating parent directories as needed."""
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False)


def load_table(path):
    """Read a parquet file into a dataframe."""
    return pd.read_parquet(path)


def save_json(obj, path):
    """Serialise a dict to JSON, coercing numpy scalars to native types."""
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(obj, default=_json_default, indent=2))


def load_json(path):
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text())


def _json_default(obj):
    """Fallback JSON encoder for numpy scalar and array types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
