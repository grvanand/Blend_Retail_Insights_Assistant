# ui/app.py
# Streamlit UI — Retail Insights Assistant
# Modes: Summarize | Chat Q&A
# Run: streamlit run ui/app.py

import sys
import os

# Ensure project root is in path when running from ui/ directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ingestion.loader import load_all_csvs, get_metadata_descriptors, CSV_REGISTRY
from vectorstore.faiss_store import ensure_index
from graph.dag import run_pipeline
from summarizer.summarize import run_summarizer
from memory.conversation import ConversationMemory
from utils.logger import logger


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Retail Insights Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# App startup — index build (once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading datasets and building index...")
def initialise_app():
    """
    Load all CSVs, build FAISS index (if not exists).
    Cached — runs only once per Streamlit session.
    """
    dataframes  = load_all_csvs()
    descriptors = get_metadata_descriptors(dataframes)
    ensure_index(descriptors)
    logger.info("App initialised — datasets loaded, FAISS index ready.")
    return dataframes, descriptors


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []     # [{role, content}]
    if "summary_output" not in st.session_state:
        st.session_state.summary_output = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(dataframes: dict) -> tuple[str, str]:
    """
    Render sidebar controls.
    Returns: (selected_mode, selected_csv)
    """
    st.sidebar.title("🛍️ Retail Insights")
    st.sidebar.markdown("---")

    # Mode selector
    mode = st.sidebar.radio(
        "Select Mode",
        options=["💬 Chat Q&A", "📊 Summarize"],
        index=0,
    )

    st.sidebar.markdown("---")

    # Dataset selector (used in summarize mode)
    csv_options = ["All Datasets"] + [
        CSV_REGISTRY[f]["name"] for f in dataframes.keys()
    ]
    selected_name = st.sidebar.selectbox("Dataset (Summarize Mode)", csv_options)

    # Map name back to filename
    selected_csv = None
    if selected_name != "All Datasets":
        selected_csv = next(
            (f for f, meta in CSV_REGISTRY.items() if meta["name"] == selected_name),
            None,
        )

    st.sidebar.markdown("---")

    # Loaded datasets info
    st.sidebar.markdown("**Loaded Datasets**")
    for filename in dataframes:
        name = CSV_REGISTRY.get(filename, {}).get("name", filename)
        rows = len(dataframes[filename])
        st.sidebar.caption(f"✅ {name} ({rows:,} rows)")

    st.sidebar.markdown("---")

    # Reset chat button
    if st.sidebar.button("🗑️ Reset Chat"):
        st.session_state.memory.clear()
        st.session_state.chat_history = []
        st.session_state.summary_output = ""
        st.rerun()

    return mode, selected_csv


# ---------------------------------------------------------------------------
# Chat Q&A Mode
# ---------------------------------------------------------------------------

def render_chat_mode():
    """Render multi-turn chat interface."""
    st.title("💬 Retail Chat Q&A")
    st.caption("Ask any question about your retail data in natural language.")

    # Display chat history
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about your sales data...")

    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Append to UI history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Build context-aware query using memory
        contextual_query = st.session_state.memory.build_contextual_query(user_input)

        # Run pipeline
        with st.chat_message("assistant"):
            with st.spinner("Analysing data..."):
                result   = run_pipeline(contextual_query)
                response = result.get("final_response", "Unable to process query.")

            st.markdown(response)

            # Debug expander — visible in development only
            if os.getenv("APP_ENV", "development") == "development":
                with st.expander("🔍 Debug Info", expanded=False):
                    st.json({
                        "intent":      result.get("intent"),
                        "filters":     result.get("filters"),
                        "routed_csvs": [c["name"] for c in (result.get("routed_csvs") or [])],
                        "error":       result.get("error"),
                    })

        # Update memory
        st.session_state.memory.add_user_message(user_input)
        st.session_state.memory.add_assistant_message(response)

        # Append assistant response to UI history
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Summarize Mode
# ---------------------------------------------------------------------------

def render_summarize_mode(selected_csv: str):
    """Render summarization panel."""
    st.title("📊 Retail Data Summarizer")
    st.caption("Generate an executive summary of your retail datasets.")

    col1, col2 = st.columns([3, 1])

    with col2:
        generate = st.button("▶ Generate Summary", use_container_width=True)
        st.caption(
            f"Target: **{'All Datasets' if not selected_csv else CSV_REGISTRY.get(selected_csv, {}).get('name', selected_csv)}**"
        )

    if generate:
        with st.spinner("Generating summary..."):
            summary = run_summarizer(target_csv=selected_csv)
            st.session_state.summary_output = summary

    if st.session_state.summary_output:
        st.markdown("---")
        st.subheader("📋 Executive Summary")
        st.markdown(st.session_state.summary_output)

        # Download button
        st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state.summary_output,
            file_name="retail_summary.txt",
            mime="text/plain",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_session_state()

    # Initialise app resources
    try:
        dataframes, _ = initialise_app()
    except Exception as e:
        st.error(f"❌ Startup failed: {e}")
        st.stop()

    if not dataframes:
        st.error("❌ No CSV files found in `data/raw/`. Please add your datasets and restart.")
        st.stop()

    # Sidebar
    mode, selected_csv = render_sidebar(dataframes)

    # Route to selected mode
    if mode == "💬 Chat Q&A":
        render_chat_mode()
    else:
        render_summarize_mode(selected_csv)


if __name__ == "__main__":
    main()
