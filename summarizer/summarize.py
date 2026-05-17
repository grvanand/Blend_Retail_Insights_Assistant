# summarizer/summarize.py
# Summarization Mode handler.
# Responsibilities:
#   1. Build a summarization query from loaded CSV metadata
#   2. Extract aggregate stats from all/selected CSVs via Pandas
#   3. Call Groq LLM with SUMMARIZER_PROMPT to generate executive summary
#   4. Return formatted summary string

from pathlib import Path
from typing import Optional
import pandas as pd

from langchain_groq import ChatGroq
from config.settings import settings
from config.prompts import SUMMARIZER_PROMPT
from ingestion.loader import load_all_csvs, CSV_REGISTRY
from utils.logger import logger
from utils.retry import llm_retry, io_retry


# ---------------------------------------------------------------------------
# LLM — singleton
# ---------------------------------------------------------------------------

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """Lazy singleton Groq LLM client."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    return _llm


# ---------------------------------------------------------------------------
# Context builder for summarization
# ---------------------------------------------------------------------------

def _build_summary_context(dataframes: dict) -> str:
    """
    Build a compact summary context string from all loaded DataFrames.
    For each CSV:
      - Row/col count
      - Numeric column stats
      - Top 5 rows
    Keeps total context within LLM token limits.
    """
    sections = []

    for filename, df in dataframes.items():
        meta    = CSV_REGISTRY.get(filename, {})
        name    = meta.get("name", filename)
        section = [f"=== {name} ===", f"Rows: {len(df)} | Columns: {len(df.columns)}"]

        # Numeric summary
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            stats = df[numeric_cols].describe().round(2).to_string()
            section.append(f"Numeric Summary:\n{stats}")

        # Top 5 rows
        sample = df.head(5).to_string(index=False, max_cols=10)
        section.append(f"Sample (5 rows):\n{sample}")

        sections.append("\n".join(section))

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Core summarizer
# ---------------------------------------------------------------------------

@llm_retry
def _call_summarizer_llm(context: str) -> str:
    """Call Groq LLM with SUMMARIZER_PROMPT and data context."""
    chain  = SUMMARIZER_PROMPT | get_llm()
    result = chain.invoke({"context": context})
    return result.content.strip()


def run_summarizer(target_csv: Optional[str] = None) -> str:
    """
    Generate an executive summary from loaded CSV datasets.

    Args:
        target_csv: Optional filename to summarize a single CSV.
                    If None, summarizes all loaded CSVs.

    Returns:
        Formatted summary string for display in UI or CLI.
    """
    logger.info(f"Summarizer started — target_csv={target_csv or 'ALL'}")

    # Load datasets
    all_dataframes = load_all_csvs()

    if not all_dataframes:
        logger.error("No CSV files loaded — check DATA_DIR in .env")
        return "No datasets found. Please ensure CSV files are placed in the data/raw/ directory."

    # Filter to single CSV if requested
    if target_csv:
        if target_csv not in all_dataframes:
            logger.warning(f"Requested CSV '{target_csv}' not found in loaded files.")
            return f"Dataset '{target_csv}' not found. Available: {list(all_dataframes.keys())}"
        dataframes = {target_csv: all_dataframes[target_csv]}
    else:
        dataframes = all_dataframes

    try:
        # Build context
        context = _build_summary_context(dataframes)
        logger.info(f"Summary context built — {len(context)} chars")

        # Call LLM
        summary = _call_summarizer_llm(context)

        if not summary:
            return "Summary could not be generated. Please try again."

        logger.info("Summarizer completed successfully.")
        return summary

    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        return f"An error occurred while generating the summary: {e}"
