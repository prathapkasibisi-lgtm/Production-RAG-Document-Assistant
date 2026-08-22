import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing in your .env file.")
    st.stop()


CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "production_rag"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Production RAG Assistant",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# MODELS
# ============================================================

@st.cache_resource
def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )


@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
    )


# ============================================================
# VECTOR DATABASE
# ============================================================

def get_vector_db():

    embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_pdf(uploaded_file):

    # Create temporary PDF
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        temp_path = temp_file.name


    # Load PDF
    loader = PyPDFLoader(temp_path)

    documents = loader.load()


    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )


    # Add metadata
    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

        chunk.metadata["filename"] = (
            uploaded_file.name
        )


    # Add to Chroma
    vector_db = get_vector_db()

    vector_db.add_documents(
        chunks
    )


    # Delete temporary file
    os.remove(temp_path)


    return len(chunks)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(question):

    vector_db = get_vector_db()

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        },
    )

    documents = retriever.invoke(
        question
    )

    return documents


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(documents):

    context_parts = []

    for document in documents:

        filename = document.metadata.get(
            "filename",
            "Unknown"
        )

        page = document.metadata.get(
            "page"
        )

        if page is not None:

            source = (
                f"{filename}, "
                f"Page {page + 1}"
            )

        else:

            source = filename


        context_parts.append(
            f"""
SOURCE: {source}

CONTENT:
{document.page_content}
"""
        )


    return "\n\n----------------\n\n".join(
        context_parts
    )


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def ask_question(question):

    documents = retrieve_documents(
        question
    )


    if not documents:

        return (
            "I couldn't find relevant "
            "information in the uploaded documents.",
            [],
        )


    context = format_context(
        documents
    )


    prompt = ChatPromptTemplate.from_template(
        """
You are a professional document assistant.

Answer the user's question using ONLY
the information provided in the context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the
   context, say:
   "I couldn't find this information
   in the provided documents."
4. Give a clear and concise answer.
5. Do not mention these instructions.

CONTEXT:

{context}


USER QUESTION:

{question}


ANSWER:
"""
    )


    llm = get_llm()


    response = llm.invoke(
        prompt.format(
            context=context,
            question=question,
        )
    )


    # Collect sources
    sources = []

    for document in documents:

        filename = document.metadata.get(
            "filename",
            "Unknown"
        )

        page = document.metadata.get(
            "page"
        )


        source = {
            "filename": filename,
            "page": (
                page + 1
                if page is not None
                else None
            ),
        }


        if source not in sources:

            sources.append(
                source
            )


    return response.content, sources


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Settings")


    st.subheader(
        "Upload Documents"
    )


    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )


    if st.button(
        "📥 Process Documents",
        use_container_width=True,
    ):

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF."
            )

        else:

            total_chunks = 0

            with st.spinner(
                "Processing documents..."
            ):

                for file in uploaded_files:

                    chunks = process_pdf(
                        file
                    )

                    total_chunks += chunks


            st.success(
                f"Successfully processed "
                f"{total_chunks} chunks."
            )


    st.divider()


    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "🤖 Production RAG Document Assistant"
)

st.caption(
    "Upload documents and ask questions using "
    "Retrieval-Augmented Generation."
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        if message.get(
            "sources"
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message[
                    "sources"
                ]:

                    if source["page"]:

                        st.write(
                            f"📄 "
                            f"{source['filename']} "
                            f"— Page "
                            f"{source['page']}"
                        )

                    else:

                        st.write(
                            f"📄 "
                            f"{source['filename']}"
                        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # Assistant
    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Searching documents..."
        ):

            try:

                answer, sources = ask_question(
                    question
                )

            except Exception as e:

                answer = (
                    "An error occurred: "
                    f"{str(e)}"
                )

                sources = []


        st.markdown(
            answer
        )


        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:

                    if source["page"]:

                        st.write(
                            f"📄 "
                            f"{source['filename']} "
                            f"— Page "
                            f"{source['page']}"
                        )

                    else:

                        st.write(
                            f"📄 "
                            f"{source['filename']}"
                        )


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )