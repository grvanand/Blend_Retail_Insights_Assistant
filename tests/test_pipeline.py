# tests/test_pipeline.py
# Smoke tests — verify all major components initialise and run correctly.
# Run: pytest tests/test_pipeline.py -v

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# 1. CSV Loading
# ---------------------------------------------------------------------------

class TestCSVLoader:

    def test_load_all_csvs_returns_dict(self):
        """load_all_csvs() should return a dict (even if empty when no files present)."""
        from ingestion.loader import load_all_csvs
        result = load_all_csvs()
        assert isinstance(result, dict)

    def test_metadata_descriptors_structure(self):
        """get_metadata_descriptors() should return list of dicts with required keys."""
        from ingestion.loader import load_all_csvs, get_metadata_descriptors
        dataframes  = load_all_csvs()
        if not dataframes:
            pytest.skip("No CSV files in data/raw/ — skipping descriptor test.")
        descriptors = get_metadata_descriptors(dataframes)
        assert isinstance(descriptors, list)
        assert len(descriptors) > 0
        for d in descriptors:
            assert "filename"       in d
            assert "name"           in d
            assert "embedding_text" in d
            assert "row_count"      in d


# ---------------------------------------------------------------------------
# 2. Embedder
# ---------------------------------------------------------------------------

class TestEmbedder:

    def test_embed_texts_returns_matrix(self):
        """embed_texts() should return a 2D numpy array."""
        from ingestion.embedder import embed_texts
        import numpy as np
        texts  = ["Amazon sales data with order ID and category."]
        result = embed_texts(texts)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_embed_query_returns_vector(self):
        """embed_query() should return a 1D numpy array."""
        from ingestion.embedder import embed_query
        import numpy as np
        result = embed_query("Which category had highest sales?")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_embed_query_empty_raises(self):
        """embed_query() should raise ValueError for empty input."""
        from ingestion.embedder import embed_query
        with pytest.raises(ValueError):
            embed_query("")


# ---------------------------------------------------------------------------
# 3. FAISS Index
# ---------------------------------------------------------------------------

class TestFAISSStore:

    def test_ensure_index_builds_without_error(self):
        """ensure_index() should build or skip without raising."""
        from ingestion.loader import load_all_csvs, get_metadata_descriptors
        from vectorstore.faiss_store import ensure_index
        dataframes  = load_all_csvs()
        if not dataframes:
            pytest.skip("No CSV files — skipping FAISS build test.")
        descriptors = get_metadata_descriptors(dataframes)
        ensure_index(descriptors)   # should not raise

    def test_query_index_returns_list(self):
        """query_index() should return a list of dicts."""
        from vectorstore.faiss_store import query_index
        try:
            results = query_index("Show me Amazon sales by category")
            assert isinstance(results, list)
            if results:
                assert "filename" in results[0]
                assert "name"     in results[0]
                assert "score"    in results[0]
        except FileNotFoundError:
            pytest.skip("FAISS index not built yet — skipping query test.")


# ---------------------------------------------------------------------------
# 4. Conversation Memory
# ---------------------------------------------------------------------------

class TestConversationMemory:

    def test_add_and_retrieve_messages(self):
        """Memory should store and return messages correctly."""
        from memory.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_user_message("Hello")
        mem.add_assistant_message("Hi there!")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["role"]    == "user"
        assert history[1]["role"]    == "assistant"

    def test_context_string_format(self):
        """get_context_string() should format turns correctly."""
        from memory.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_user_message("What are total sales?")
        mem.add_assistant_message("Total sales are 500k.")
        ctx = mem.get_context_string()
        assert "User:" in ctx
        assert "Assistant:" in ctx

    def test_build_contextual_query(self):
        """build_contextual_query() should prepend history to new query."""
        from memory.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_user_message("Show Amazon data")
        mem.add_assistant_message("Here it is.")
        result = mem.build_contextual_query("Which category is top?")
        assert "Previous conversation:" in result
        assert "Which category is top?" in result

    def test_clear_resets_history(self):
        """clear() should empty all history."""
        from memory.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_user_message("Test")
        mem.clear()
        assert mem.is_empty()

    def test_trim_respects_max_turns(self):
        """Memory should not exceed max_turns * 2 messages."""
        from memory.conversation import ConversationMemory
        mem = ConversationMemory(max_turns=2)
        for i in range(6):
            mem.add_user_message(f"Message {i}")
        assert len(mem) <= 4   # max_turns=2 → 2*2=4 messages max


# ---------------------------------------------------------------------------
# 5. DAG Pipeline (mocked LLM)
# ---------------------------------------------------------------------------

class TestDAGPipeline:

    def test_empty_query_returns_gracefully(self):
        """Pipeline should handle empty query without crashing."""
        from graph.dag import run_pipeline
        result = run_pipeline("")
        assert "final_response" in result
        assert result["final_response"] != ""

    @patch("agents.query_resolver._extract_intent")
    @patch("vectorstore.faiss_store.query_index")
    def test_pipeline_runs_end_to_end(self, mock_index, mock_intent):
        """Pipeline should complete with mocked LLM and FAISS."""
        mock_index.return_value = [{
            "filename":    "Amazon Sale Report.csv",
            "name":        "Amazon Sale Report",
            "description": "Amazon sales data",
            "key_columns": ["Category", "Amount"],
            "row_count":   100,
            "score":       0.95,
        }]
        mock_intent.return_value = {
            "intent":   "qa",
            "filters":  {"category": "T-shirt"},
            "question": "Which T-shirt category had highest sales?",
        }

        with patch("agents.validator._generate_draft", return_value="T-shirts led sales."), \
             patch("agents.validator._validate_draft", return_value="T-shirts led sales in Q3."):
            from graph.dag import run_pipeline
            result = run_pipeline("Which T-shirt category had highest sales?")

        assert result.get("final_response") is not None
        assert result.get("error") is None


# ---------------------------------------------------------------------------
# 6. Summarizer (mocked LLM)
# ---------------------------------------------------------------------------

class TestSummarizer:

    @patch("summarizer.summarize._call_summarizer_llm", return_value="Sales grew 12% YoY.")
    def test_summarizer_returns_string(self, mock_llm):
        """run_summarizer() should return a non-empty string."""
        from summarizer.summarize import run_summarizer
        result = run_summarizer()
        assert isinstance(result, str)
        assert len(result) > 0
