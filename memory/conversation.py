# memory/conversation.py
# Lightweight in-memory conversation history manager.
# Maintains chat turns for multi-turn Q&A context.
# Injected into user queries before passing to the DAG pipeline.

from typing import List, Dict
from utils.logger import logger


# Max turns to retain in context window (older turns are dropped)
MAX_HISTORY_TURNS = 10


class ConversationMemory:
    """
    Manages chat history for a single user session.
    Stores alternating user/assistant turns as a list of dicts.
    Injected into user queries to provide conversation context.
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns  = max_turns
        self._history:  List[Dict[str, str]] = []   # [{role, content}, ...]

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def add_user_message(self, message: str) -> None:
        """Append a user turn to history."""
        self._history.append({"role": "user", "content": message.strip()})
        self._trim()
        logger.debug(f"Memory: user message added. Total turns: {len(self._history)}")

    def add_assistant_message(self, message: str) -> None:
        """Append an assistant turn to history."""
        self._history.append({"role": "assistant", "content": message.strip()})
        self._trim()
        logger.debug(f"Memory: assistant message added. Total turns: {len(self._history)}")

    # -----------------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------------

    def get_history(self) -> List[Dict[str, str]]:
        """Return full chat history as list of {role, content} dicts."""
        return list(self._history)

    def get_context_string(self) -> str:
        """
        Format chat history as a plain string for injection into LLM prompts.
        Format:
            User: ...
            Assistant: ...
        """
        if not self._history:
            return ""

        lines = []
        for turn in self._history:
            role    = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")

        return "\n".join(lines)

    def build_contextual_query(self, new_query: str) -> str:
        """
        Prepend recent conversation context to the new query.
        Helps query_resolver understand follow-up questions.

        Example output:
            Previous conversation:
            User: Show me Amazon sales
            Assistant: Here are the Amazon sales stats...

            New question: Which category had highest sales?
        """
        history_str = self.get_context_string()

        if not history_str:
            return new_query

        return (
            f"Previous conversation:\n{history_str}\n\n"
            f"New question: {new_query}"
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all history — called on session reset."""
        self._history.clear()
        logger.info("Conversation memory cleared.")

    def is_empty(self) -> bool:
        return len(self._history) == 0

    def __len__(self) -> int:
        return len(self._history)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _trim(self) -> None:
        """
        Keep only the last max_turns messages.
        Each turn = 1 message, so max_turns=10 retains 5 exchanges.
        """
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]
            logger.debug("Memory trimmed to max_turns limit.")
