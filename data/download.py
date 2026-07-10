"""
Dataset downloader for FraudGraph.
Downloads from Kaggle using the Kaggle API.
Also provides a manual download fallback with clear instructions.

Datasets:
  1. IBM AML (HI-Small): Synthetic transaction data with 8 laundering patterns
     https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
  2. Elliptic: Real Bitcoin transaction graph with licit/illicit labels
     https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IBM_AML_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
ELLIPTIC_DATASET = "ellipticco/elliptic-data-set"

# Target filenames after download
IBM_AML_FILE = "HI-Small_Trans.csv"    # HI = High Illicit rate; Small = manageable size
ELLIPTIC_FEATURES_FILE = "elliptic_txs_features.csv"
ELLIPTIC_CLASSES_FILE = "elliptic_txs_classes.csv"
ELLIPTIC_EDGELIST_FILE = "elliptic_txs_edgelist.csv"


def download_datasets(data_dir: str = "./data/raw") -> None:
    """
    Download both datasets using the Kaggle API.

    Requires KAGGLE_USERNAME and KAGGLE_KEY in environment or ~/.kaggle/kaggle.json.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Load .env variables to set Kaggle credentials dynamically
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    parts = line.split("=", 1)
                    k = parts[0].strip()
                    v = parts[1].strip().strip('"').strip("'")
                    if k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
                        os.environ[k] = v
                        logger.info(f"Loaded environment variable {k}")

    try:
        import kaggle
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    # ── IBM AML ───────────────────────────────────────────────────────────────
    ibm_path = Path(data_dir) / IBM_AML_FILE
    if ibm_path.exists():
        logger.info(f"IBM AML dataset already exists: {ibm_path}")
    else:
        logger.info("Downloading IBM AML dataset (HI-Small variant)...")
        try:
            kaggle.api.dataset_download_files(
                IBM_AML_DATASET,
                path=data_dir,
                unzip=True,
            )
            logger.info(f"IBM AML downloaded to {data_dir}")
        except Exception as e:
            logger.error(f"IBM AML download failed: {e}")
            _print_manual_instructions()

    # ── Elliptic ──────────────────────────────────────────────────────────────
    elliptic_path = Path(data_dir) / ELLIPTIC_FEATURES_FILE
    if elliptic_path.exists():
        logger.info(f"Elliptic dataset already exists: {elliptic_path}")
    else:
        logger.info("Downloading Elliptic dataset...")
        try:
            kaggle.api.dataset_download_files(
                ELLIPTIC_DATASET,
                path=data_dir,
                unzip=True,
            )
            logger.info(f"Elliptic downloaded to {data_dir}")
        except Exception as e:
            logger.error(f"Elliptic download failed: {e}")
            _print_manual_instructions()


def _print_manual_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║               Manual Dataset Download Instructions           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ 1. IBM AML Dataset (HI-Small variant):                       ║
║    https://www.kaggle.com/datasets/ealtman2019/              ║
║    ibm-transactions-for-anti-money-laundering-aml            ║
║    → Download and place HI-Small_Trans.csv in ./data/raw/    ║
║                                                              ║
║ 2. Elliptic Dataset:                                         ║
║    https://www.kaggle.com/datasets/ellipticco/elliptic-data-set ║
║    → Download and place all 3 CSV files in ./data/raw/       ║
║      - elliptic_txs_features.csv                             ║
║      - elliptic_txs_classes.csv                              ║
║      - elliptic_txs_edgelist.csv                             ║
║                                                              ║
║ For Kaggle API credentials:                                  ║
║   https://www.kaggle.com/settings → API → Create New Token  ║
║   Save kaggle.json to ~/.kaggle/kaggle.json                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def verify_datasets(data_dir: str = "./data/raw") -> dict[str, bool]:
    """Check which dataset files are present."""
    status = {}
    data_path = Path(data_dir)

    status["ibm_aml"] = (data_path / IBM_AML_FILE).exists()
    status["elliptic_features"] = (data_path / ELLIPTIC_FEATURES_FILE).exists()
    status["elliptic_classes"] = (data_path / ELLIPTIC_CLASSES_FILE).exists()
    status["elliptic_edgelist"] = (data_path / ELLIPTIC_EDGELIST_FILE).exists()
    status["elliptic_complete"] = all([
        status["elliptic_features"],
        status["elliptic_classes"],
        status["elliptic_edgelist"],
    ])

    for name, present in status.items():
        icon = "✅" if present else "❌"
        logger.info(f"  {icon} {name}")

    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_datasets()
    verify_datasets()
