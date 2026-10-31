import chromadb
import streamlit as st

from ingest import process_uploaded_files

from vector_store import (
    add_chunks,
    get_collection_count
)

from rag_pipeline import generate_answer


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="College Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)


# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp {
    background-color: #10141D;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #181F2B;
}

.block-container {
    padding-top: 2rem;
}

div[data-testid="stMetric"] {
    background-color: #1B2534;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #303B4D;
}

.stButton > button {
    background-color: #536DFE;
    color: white;
    border-radius: 8px;
    min-height: 42px;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# SESSION STATE
# =====================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================
# DATABASE STATUS
# =====================================

try:

    database_count = get_collection_count()

except Exception:

    database_count = 0


# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🎓 CKA")

    st.caption("COLLEGE KNOWLEDGE ASSISTANT")

    st.divider()

    if st.button("🏠 Dashboard", use_container_width=True):

        st.session_state.page = "Dashboard"
        st.rerun()

    if st.button("📁 Documents", use_container_width=True):

        st.session_state.page = "Documents"
        st.rerun()

    if st.button("💬 AI Assistant", use_container_width=True):

        st.session_state.page = "Assistant"
        st.rerun()

    st.divider()

    st.caption("SYSTEM STATUS")

    st.success("Streamlit Online")

    st.success("ChromaDB Connected")

    st.success("Ollama Local")

    st.caption("EMBEDDING MODEL")

    st.write("Nomic Embed Text")

    st.caption("LANGUAGE MODEL")

    st.write("Gemma 4 12B")


# =====================================
# DASHBOARD
# =====================================

if st.session_state.page == "Dashboard":

    st.title("College Knowledge Assistant")

    st.write(
        "Your intelligent assistant for college policies, "
        "academic rules and institutional documents."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Knowledge Base Chunks",
            database_count
        )

    with col2:

        st.metric(
            "AI Model",
            "Gemma 4 12B"
        )

    with col3:

        st.metric(
            "Embedding Model",
            "Nomic"
        )

    st.divider()

    st.subheader("Welcome!")

    st.write(
        "Upload official college documents and allow students "
        "to ask questions about institutional policies."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("📁 Manage Documents", use_container_width=True):

            st.session_state.page = "Documents"
            st.rerun()

    with col2:

        if st.button("💬 Open AI Assistant", use_container_width=True):

            st.session_state.page = "Assistant"
            st.rerun()


# =====================================
# DOCUMENT MANAGEMENT
# =====================================

elif st.session_state.page == "Documents":

    st.title("Document Management")

    st.write(
        "Upload official college PDF or TXT documents."
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "Choose your college documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.subheader("Selected Documents")

        for file in uploaded_files:

            st.write(
                f"📄 {file.name} "
                f"({file.size / 1024:.2f} KB)"
            )

        if st.button(
            "⚡ Process and Index Documents",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Extracting text and generating embeddings..."
                ):

                    documents, chunks = process_uploaded_files(
                        uploaded_files
                    )

                    if not chunks:

                        st.warning(
                            "No readable text found in uploaded documents."
                        )

                    else:

                        stored_count = add_chunks(chunks)

                        st.success(
                            "Documents successfully indexed!"
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Pages Extracted",
                            len(documents)
                        )

                        col2.metric(
                            "Chunks Generated",
                            len(chunks)
                        )

                        col3.metric(
                            "Chunks Stored",
                            stored_count
                        )

            except Exception as error:

                st.error(
                    f"Processing error: {error}"
                )

    st.divider()

    st.subheader("Knowledge Base")

    st.metric(
        "Total Stored Chunks",
        get_collection_count()
    )

    st.success(
        "Your indexed documents are stored persistently in ChromaDB."
    )


# =====================================
# AI ASSISTANT
# =====================================

elif st.session_state.page == "Assistant":

    st.title("AI Knowledge Assistant")

    st.caption(
        "Ask questions about college policies and documents."
    )

    st.divider()

    # CHAT HISTORY

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander("View Sources"):

                    for source in message["sources"]:

                        st.write(
                            f"📄 {source['name']} "
                            f"| Page {source['page']}"
                        )

    # USER INPUT

    question = st.chat_input(
        "Ask your college-related question..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner("Searching documents and generating answer..."):

                    result = generate_answer(question)

                st.markdown(result["answer"])

                if result["sources"]:

                    with st.expander("📚 Supporting Sources"):

                        for source in result["sources"]:

                            st.write(
                                f"📄 {source['name']} "
                                f"| Page {source['page']}"
                            )

                else:

                    st.warning(
                        "No supporting document sources were found."
                    )

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": result["answer"],

                    "sources": result["sources"]

                })

            except Exception as error:

                st.error(
                    f"AI processing failed: {error}"
                )


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "College Knowledge Assistant | "
    "Python • Streamlit • Ollama • Gemma • ChromaDB • RAG"
)