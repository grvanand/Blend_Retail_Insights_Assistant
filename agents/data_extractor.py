# agents/data_extractor.py
# Agent 2 — Data Extractor
# Responsibilities:
#   1. Load the routed CSV(s) identified by query_resolver
#   2. Apply filters extracted from user query
#   3. Compute summary statistics or relevant data slices via Pandas
#   4. Return a concise string context for the LLM in downstream agents

import pandas as pd
from pathlib import Path
from typing import List, Dict

from ingestion.loader import load_csv
from config.settings import settings
from utils.logger import logger
from utils.retry import io_retry


# Max rows to pass as LLM context (avoid token overflow)
MAX_CONTEXT_ROWS = 50


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply extracted filters to a DataFrame.
    Performs case-insensitive substring match on string columns.
    Silently skips filters whose keys don't match any column.
    """
    if not filters:
        return df

    for key, value in filters.items():
        if not value:
            continue

        # Find best-matching column (case-insensitive key match)
        matched_col = next(
            (c for c in df.columns if key.lower() in c.lower()), None
        )

        if matched_col is None:
            logger.debug(f"Filter key '{key}' not found in columns — skipping.")
            continue

        # String filter: case-insensitive contains
        if df[matched_col].dtype == object:
            mask = df[matched_col].str.contains(str(value), case=False, na=False)
            df   = df[mask]
            logger.debug(f"Filter applied: {matched_col} contains '{value}' → {len(df)} rows")

        # Numeric filter: exact match
        else:
            try:
                df = df[df[matched_col] == type(df[matched_col].iloc[0])(value)]
                logger.debug(f"Filter applied: {matched_col} == {value} → {len(df)} rows")
            except (ValueError, IndexError):
                logger.debug(f"Numeric filter failed for '{matched_col}' — skipping.")

    return df


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(df: pd.DataFrame, csv_name: str) -> str:
    """
    Convert a filtered DataFrame into a concise string context for the LLM.
    Includes:
    - Dataset name and shape
    - Numeric column summary statistics
    - Top rows as a readable table (capped at MAX_CONTEXT_ROWS)
    """
    if df.empty:
        return f"[{csv_name}]: No data found after applying filters."

    lines = [f"Dataset: {csv_name} — {len(df)} rows, {len(df.columns)} columns\n"]

    # Numeric summary
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().round(2).to_string()
        lines.append(f"Numeric Summary:\n{stats}\n")

    # Top rows as readable context
    sample      = df.head(MAX_CONTEXT_ROWS)
    sample_text = sample.to_string(index=False, max_cols=15)
    lines.append(f"Sample Data (up to {MAX_CONTEXT_ROWS} rows):\n{sample_text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core extractor logic
# ---------------------------------------------------------------------------

@io_retry
def _load_and_extract(csv_meta: Dict, filters: dict) -> str:
    """
    Load a single CSV, apply filters, build and return context string.
    """
    filepath = Path(settings.data_dir) / csv_meta["filename"]

    if not filepath.exists():
        logger.warning(f"CSV file missing: {filepath}")
        return f"[{csv_meta['name']}]: File not found on disk."

    df      = load_csv(str(filepath))
    df      = _apply_filters(df, filters)
    context = _build_context(df, csv_meta["name"])
    return context


def run_data_extractor(state: dict) -> dict:
    """
    LangGraph node function for Agent 2.

    Reads from state:
        - routed_csvs  (list) : CSV metadata dicts from query_resolver
        - filters      (dict) : column filters to apply
        - error        (str)  : if set, skip processing and pass through

    Writes to state:
        - context  (str)  : combined data context string for LLM
        - error    (str)  : set on failure, else preserved from upstream
    """
    # Pass through if upstream already failed
    if state.get("error"):
        logger.warning("Data extractor skipping — upstream error detected.")
        return {**state, "context": ""}

    routed_csvs: List[Dict] = state.get("routed_csvs", [])
    filters: dict           = state.get("filters", {})

    if not routed_csvs:
        logger.warning("No routed CSVs — returning empty context.")
        return {**state, "context": "No relevant dataset found for this query."}

    contexts = []

    for csv_meta in routed_csvs:
        try:
            context = _load_and_extract(csv_meta, filters)
            contexts.append(context)
        except Exception as e:
            logger.error(f"Failed to extract from '{csv_meta.get('name')}': {e}")
            contexts.append(f"[{csv_meta.get('name', 'Unknown')}]: Extraction failed — {e}")

    combined_context = "\n\n---\n\n".join(contexts)

    logger.info(
        f"Data extractor complete — {len(contexts)} dataset(s) processed, "
        f"context length={len(combined_context)} chars"
    )

    return {
        **state,
        "context": combined_context,
        "error":   None,
    }
