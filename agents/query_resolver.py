# agents/query_resolver.py
# Agent 1 — Query Resolver
# Responsibilities:
#   1. Use FAISS to identify which CSV(s) are relevant to the user query
#   2. Use Groq LLM to extract intent (summarize/qa) and filters from the query
#   3. Populate GraphState with routing info for downstream agents

import json
from langchain_groq import ChatGroq
from config.settings import settings
from config.prompts import QUERY_RESOLVER_PROMPT
from vectorstore.faiss_store import query_index
from utils.logger import logger
from utils.retry import llm_retry


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
        logger.info(f"Groq LLM initialised: {settings.groq_model_name}")
    return _llm


# ---------------------------------------------------------------------------
# Core resolver logic
# ---------------------------------------------------------------------------

@llm_retry
def _extract_intent(user_query: str) -> dict:
    """
    Call Groq LLM to extract intent and filters from user query.
    Returns parsed JSON dict with keys: intent, filters, question.
    Falls back to safe defaults on parse failure.
    """
    chain  = QUERY_RESOLVER_PROMPT | get_llm()
    result = chain.invoke({"user_query": user_query})
    raw    = result.content.strip()

    try:
        parsed = json.loads(raw)
        # Validate required keys
        intent   = parsed.get("intent", "qa")
        filters  = parsed.get("filters", {})
        question = parsed.get("question", user_query)

        if intent not in ("summarize", "qa"):
            intent = "qa"

        logger.debug(f"Intent extracted — intent={intent}, filters={filters}")
        return {"intent": intent, "filters": filters, "question": question}

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Intent parse failed ({e}), using safe defaults.")
        return {"intent": "qa", "filters": {}, "question": user_query}


def run_query_resolver(state: dict) -> dict:
    """
    LangGraph node function for Agent 1.

    Reads from state:
        - user_query (str)

    Writes to state:
        - intent        (str)  : "summarize" | "qa"
        - filters       (dict) : extracted metadata filters
        - question      (str)  : cleaned query
        - routed_csvs   (list) : list of matched CSV metadata dicts from FAISS
        - error         (str)  : set if this node fails, else None
    """
    user_query = state.get("user_query", "").strip()

    if not user_query:
        logger.error("Query resolver received empty user_query.")
        return {**state, "error": "Empty query received.", "intent": "qa",
                "filters": {}, "question": "", "routed_csvs": []}

    try:
        # Step 1 — FAISS routing: find relevant CSV(s)
        routed_csvs = query_index(user_query, top_k=settings.faiss_top_k)

        if not routed_csvs:
            logger.warning("FAISS returned no matches — proceeding with empty routing.")

        # Step 2 — LLM intent extraction
        intent_data = _extract_intent(user_query)

        logger.info(
            f"Query resolved — intent='{intent_data['intent']}', "
            f"routed={[c['name'] for c in routed_csvs]}"
        )

        return {
            **state,
            "intent":      intent_data["intent"],
            "filters":     intent_data["filters"],
            "question":    intent_data["question"],
            "routed_csvs": routed_csvs,
            "error":       None,
        }

    except Exception as e:
        logger.error(f"Query resolver failed: {e}")
        return {
            **state,
            "intent":      "qa",
            "filters":     {},
            "question":    user_query,
            "routed_csvs": [],
            "error":       str(e),
        }
