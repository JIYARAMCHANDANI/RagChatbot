# app.py
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import tokenizers

st.set_page_config(page_title="Loan Approval RAG Chatbot")

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None}, allow_output_mutation=True)
def load_resources():
    index = faiss.read_index("faiss_index.bin")
    docs = pickle.load(open("docs.pkl", "rb"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        device=-1
    )
    return index, docs, embedder, generator

index, docs, embedder, generator = load_resources()

def generate_answer(query: str, k: int = 5) -> str:
    q_vec = embedder.encode([query], convert_to_numpy=True)
    _, I = index.search(q_vec, k)
    context = "\n".join(docs[i] for i in I[0])
    prompt = f"Context:\n{context}\n\nQ: {query}\nA:"
    out = generator(prompt)
    return out[0]["generated_text"], context

st.title("💰 Loan Approval RAG Chatbot")
user_query = st.text_input("Ask your question about loans:")

if st.button("🔄 Clear history"):
    st.session_state.history = []

if user_query:
    with st.spinner("Thinking..."):
        answer, context = generate_answer(user_query)
    st.session_state.history.append((user_query, answer, context))

# Display chat history
for q, a, ctx in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    with st.expander("🔍 Retrieved context"):
        st.write(ctx.replace("\n", "\n\n"))
