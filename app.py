import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Hotel-related Question Answering System")

# ----------------------------------
# LOAD EMBEDDINGS
# ----------------------------------
import os

print("Current Directory:", os.getcwd())
print("Files:", os.listdir("."))

if os.path.exists("faiss_db"):
    print("FAISS folder exists")
    print(os.listdir("faiss_db"))
else:
    print("FAISS folder NOT found")

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    VECTOR_DB_PATH = "faiss_db"

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

vector_db = load_vectorstore()

retriever = vector_db.as_retriever(
    search_kwargs={"k":3}
)

# ----------------------------------
# LOAD GROQ MODEL
# ----------------------------------

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key = st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# ----------------------------------
# PROMPT
# ----------------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Answer only from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
)

output_parser = StrOutputParser()

# ----------------------------------
# RAG FUNCTION
# ----------------------------------

def get_answer(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    chain = prompt | llm | output_parser

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, docs


# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
}

.stChatMessage {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# HEADER
# ======================================

st.markdown("""
# 🤖 HotelGPT: AI Document Assistant

Ask questions about your uploaded documents using RAG and Groq.

---
""")

# ======================================
# SIDEBAR
# ======================================

with st.sidebar:

    st.title("⚙ Settings")

    top_k = st.slider(
        "Documents to Retrieve",
        1,
        10,
        3
    )

    st.markdown("---")

    st.write("### Model")

    st.write("🧠 Llama 3.3 70B")
    st.write("📚 MiniLM-L6-v2")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Update retriever dynamically
retriever = vector_db.as_retriever(
    search_kwargs={"k": top_k}
)

# ======================================
# CHAT HISTORY
# ======================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ======================================
# CHAT INPUT
# ======================================

question = st.chat_input(
    "Ask a question about your documents..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            answer, docs = get_answer(question)

        st.markdown(answer)

        with st.expander("📄 Retrieved Sources"):

            for i, doc in enumerate(docs):

                st.markdown(f"### Source {i+1}")

                st.write(
                    doc.page_content[:1000]
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
