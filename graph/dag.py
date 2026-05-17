# graph/dag.py
# LangGraph DAG — Controlled Acyclic Graph
# Wires all 3 agents into a sequential pipeline:
#   START → query_resolver → data_extractor → validator → END
#
# GraphState is the single shared state object passed through all nodes.

from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, START, END

from agents.query_resolver import run_query_resolver
from agents.data_extractor import run_data_extractor
from agents.validator import run_validator
from utils.logger import logger


# ---------------------------------------------------------------------------
# Shared State Schema
# All nodes read from and write to this TypedDict.
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    # Input
    user_query:      str                    # raw user input

    # Agent 1 — query_resolver outputs
    intent:          Optional[str]          # "summarize" | "qa"
    filters:         Optional[Dict]         # extracted filters
    question:        Optional[str]          # cleaned rephrased query
    routed_csvs:     Optional[List[Dict]]   # matched CSV metadata from FAISS

    # Agent 2 — data_extractor outputs
    context:         Optional[str]          # Pandas data context string

    # Agent 3 — validator outputs
    final_response:  Optional[str]          # user-facing final answer

    # Shared error channel
    error:           Optional[str]          # any agent sets this on failure


# ---------------------------------------------------------------------------
# DAG Builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph acyclic DAG.

    Flow:
        START
          └─► query_resolver   (intent extraction + FAISS routing)
                └─► data_extractor  (Pandas query + context building)
                      └─► validator     (LLM answer + validation)
                            └─► END

    Returns:
        Compiled LangGraph app ready for .invoke()
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("query_resolver",  run_query_resolver)
    graph.add_node("data_extractor",  run_data_extractor)
    graph.add_node("validator",       run_validator)

    # Wire edges — strict acyclic flow, no cycles
    graph.add_edge(START,              "query_resolver")
    graph.add_edge("query_resolver",   "data_extractor")
    graph.add_edge("data_extractor",   "validator")
    graph.add_edge("validator",        END)

    compiled = graph.compile()
    logger.info("LangGraph DAG compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Public runner — used by Streamlit UI and summarizer
# ---------------------------------------------------------------------------

# Compiled graph singleton — built once at import time
_graph = None


def get_graph():
    """Lazy singleton for compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(user_query: str) -> dict:
    """
    Execute the full DAG pipeline for a given user query.

    Args:
        user_query: Natural language question or summarization request

    Returns:
        Final GraphState dict containing:
        - final_response (str) : answer to show user
        - intent, filters, routed_csvs, context, error (for debugging)
    """
    if not user_query or not user_query.strip():
        logger.warning("run_pipeline called with empty query.")
        return {
            "user_query":     user_query,
            "final_response": "Please enter a valid question.",
            "error":          "Empty query.",
        }

    initial_state: GraphState = {
        "user_query":     user_query.strip(),
        "intent":         None,
        "filters":        None,
        "question":       None,
        "routed_csvs":    None,
        "context":        None,
        "final_response": None,
        "error":          None,
    }

    logger.info(f"Pipeline started — query: '{user_query[:80]}...'")

    try:
        result = get_graph().invoke(initial_state)
        logger.info("Pipeline completed successfully.")
        return result

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return {
            **initial_state,
            "final_response": (
                "An unexpected error occurred while processing your query. "
                "Please try again."
            ),
            "error": str(e),
        }
