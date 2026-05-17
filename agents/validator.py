# agents/validator.py
# Agent 3 — Validator
# Responsibilities:
#   1. Call Groq LLM with data context to generate a draft answer
#   2. Validate and clean the draft via a second LLM pass
#   3. Handle fallback gracefully if context is empty or LLM fails
#   4. Write final response to GraphState

from langchain_groq import ChatGroq
from config.settings import settings
from config.prompts import DATA_EXTRACTOR_PROMPT, VALIDATOR_PROMPT
from utils.logger import logger
from utils.retry import llm_retry


# ---------------------------------------------------------------------------
# LLM — reuse same singleton pattern as query_resolver
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
        logger.info(f"Groq LLM initialised (validator): {settings.groq_model_name}")
    return _llm


# ---------------------------------------------------------------------------
# Fallback response
# ---------------------------------------------------------------------------

FALLBACK_RESPONSE = (
    "I was unable to find sufficient data to answer your question. "
    "Please try rephrasing or ensure the relevant dataset has been loaded."
)


# ---------------------------------------------------------------------------
# Step 1 — Generate draft answer using data context
# ---------------------------------------------------------------------------

@llm_retry
def _generate_draft(question: str, context: str) -> str:
    """
    First LLM pass: answer the question using retrieved data context.
    """
    chain  = DATA_EXTRACTOR_PROMPT | get_llm()
    result = chain.invoke({"question": question, "context": context})
    return result.content.strip()


# ---------------------------------------------------------------------------
# Step 2 — Validate and clean the draft response
# ---------------------------------------------------------------------------

@llm_retry
def _validate_draft(question: str, draft: str) -> str:
    """
    Second LLM pass: clean, validate, and finalise the draft response.
    """
    chain  = VALIDATOR_PROMPT | get_llm()
    result = chain.invoke({"question": question, "draft_response": draft})
    return result.content.strip()


# ---------------------------------------------------------------------------
# Core validator logic
# ---------------------------------------------------------------------------

def run_validator(state: dict) -> dict:
    """
    LangGraph node function for Agent 3.

    Reads from state:
        - question  (str) : cleaned user question
        - context   (str) : data context from data_extractor
        - error     (str) : upstream error if any

    Writes to state:
        - final_response  (str) : validated, user-facing answer
        - error           (str) : set on failure, else None
    """
    # Upstream failure — return polite fallback immediately
    if state.get("error"):
        logger.warning(f"Validator received upstream error: {state['error']}")
        return {**state, "final_response": FALLBACK_RESPONSE}

    question: str = state.get("question", "").strip()
    context:  str = state.get("context", "").strip()

    # Empty context guard
    if not context or context == "No relevant dataset found for this query.":
        logger.warning("Validator: empty context — returning fallback.")
        return {**state, "final_response": FALLBACK_RESPONSE, "error": None}

    # Empty question guard
    if not question:
        logger.warning("Validator: empty question — returning fallback.")
        return {**state, "final_response": FALLBACK_RESPONSE, "error": None}

    try:
        # Step 1 — Generate draft from data context
        logger.info("Validator: generating draft response...")
        draft = _generate_draft(question, context)
        logger.debug(f"Draft response: {draft[:200]}...")

        # Step 2 — Validate and clean draft
        logger.info("Validator: validating draft response...")
        final = _validate_draft(question, draft)

        # Final safety check — if LLM returns empty string
        if not final:
            logger.warning("Validator: LLM returned empty final response — using fallback.")
            final = FALLBACK_RESPONSE

        logger.info("Validator: final response ready.")
        return {
            **state,
            "final_response": final,
            "error":          None,
        }

    except Exception as e:
        logger.error(f"Validator failed: {e}")
        return {
            **state,
            "final_response": FALLBACK_RESPONSE,
            "error":          str(e),
        }
