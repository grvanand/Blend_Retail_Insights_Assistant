# ingestion/loader.py
# Loads all CSVs from data/raw/, cleans them, and generates
# metadata descriptors per CSV for FAISS routing index.

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from config.settings import settings
from utils.logger import logger
from utils.retry import io_retry


# ---------------------------------------------------------------------------
# CSV registry — maps filename to human-readable description + key columns
# Add new CSVs here without changing any other module.
# ---------------------------------------------------------------------------
CSV_REGISTRY = {
    "Amazon Sale Report.csv": {
        "name": "Amazon Sale Report",
        "description": (
            "Amazon sales orders including order ID, date, status, fulfilment type, "
            "sales channel, product category, SKU, size, quantity, amount, "
            "shipping city, state, country, B2B flag, and courier status."
        ),
        "key_columns": ["Order ID", "Date", "Status", "Category", "SKU", "Qty", "Amount",
                        "ship-city", "ship-state", "ship-country", "B2B"],
    },
    "Cloud Warehouse Compersion Chart.csv": {
        "name": "Cloud Warehouse Comparison Chart",
        "description": (
            "Comparison chart of cloud warehouse providers — Shiprocket vs INCREFF, "
            "covering logistics and warehousing metrics."
        ),
        "key_columns": ["Shiprocket", "INCREFF"],
    },
    "Expense IIGF.csv": {
        "name": "Expense IIGF",
        "description": (
            "Expense tracking sheet with received amounts, expenditures, "
            "and net amounts for IIGF financial records."
        ),
        "key_columns": ["Recived Amount", "Amount", "Expance"],
    },
    "International Sale Report.csv": {
        "name": "International Sale Report",
        "description": (
            "International sales transactions including date, month, customer name, "
            "style, SKU, size, quantity (PCS), rate, and gross amount."
        ),
        "key_columns": ["DATE", "Months", "CUSTOMER", "Style", "SKU", "Size", "PCS",
                        "RATE", "GROSS AMT"],
    },
    "May-2022.csv": {
        "name": "May 2022 Product Pricing",
        "description": (
            "Product pricing sheet for May 2022 with SKU, style ID, catalog, category, "
            "weight, transfer price, and MRP across platforms: Ajio, Amazon, Amazon FBA, "
            "Flipkart, Limeroad, Myntra, Paytm, Snapdeal."
        ),
        "key_columns": ["Sku", "Style Id", "Category", "TP", "MRP Old",
                        "Amazon MRP", "Flipkart MRP", "Myntra MRP"],
    },
    "P L March 2021.csv": {
        "name": "P L March 2021",
        "description": (
            "Profit and loss sheet for March 2021 with SKU, style ID, catalog, category, "
            "weight, two transfer price tiers, and MRP across platforms: Ajio, Amazon, "
            "Amazon FBA, Flipkart, Limeroad, Myntra, Paytm, Snapdeal."
        ),
        "key_columns": ["Sku", "Style Id", "Category", "TP 1", "TP 2",
                        "Amazon MRP", "Flipkart MRP", "Myntra MRP"],
    },
    "Sale Report.csv": {
        "name": "Sale Report",
        "description": (
            "Inventory and sale report with SKU code, design number, stock quantity, "
            "category, size, and color."
        ),
        "key_columns": ["SKU Code", "Design No.", "Stock", "Category", "Size", "Color"],
    },
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@io_retry
def load_csv(filepath: str) -> pd.DataFrame:
    """Load a single CSV with basic cleaning."""
    df = pd.read_csv(filepath, low_memory=False)

    # Drop fully empty rows and columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.astype(str).str.strip())

    # Drop unnamed index columns (common in Excel-exported CSVs)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)

    logger.info(f"Loaded '{Path(filepath).name}' — {len(df)} rows, {len(df.columns)} cols")
    return df


def load_all_csvs() -> Dict[str, pd.DataFrame]:
    """
    Load all registered CSVs from DATA_DIR.
    Returns dict: { csv_filename: DataFrame }
    Skips files that are registered but not found on disk.
    """
    data_dir = Path(settings.data_dir)
    loaded = {}

    for filename in CSV_REGISTRY:
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning(f"File not found, skipping: {filepath}")
            continue
        try:
            loaded[filename] = load_csv(str(filepath))
        except Exception as e:
            logger.error(f"Failed to load '{filename}': {e}")

    logger.info(f"Loaded {len(loaded)}/{len(CSV_REGISTRY)} CSV files.")
    return loaded


def get_metadata_descriptors(dataframes: Dict[str, pd.DataFrame]) -> list[dict]:
    """
    Generate metadata descriptor dicts for each loaded CSV.
    Used by embedder.py to build FAISS routing index.

    Each descriptor contains:
    - filename, name, description, key_columns, row_count, column_names
    """
    descriptors = []

    for filename, df in dataframes.items():
        meta = CSV_REGISTRY.get(filename, {})
        descriptor = {
            "filename": filename,
            "name": meta.get("name", filename),
            "description": meta.get("description", ""),
            "key_columns": meta.get("key_columns", list(df.columns)),
            "row_count": len(df),
            "column_names": list(df.columns),
            # Full text used for embedding
            "embedding_text": (
                f"Dataset: {meta.get('name', filename)}. "
                f"{meta.get('description', '')} "
                f"Columns: {', '.join(list(df.columns))}."
            ),
        }
        descriptors.append(descriptor)
        logger.debug(f"Metadata descriptor built for: {filename}")

    return descriptors
