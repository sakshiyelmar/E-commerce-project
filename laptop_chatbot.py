import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from groq import Groq

os.environ['GROQ_API_KEY'] = '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'

# Cache models
@st.cache_resource
def load_models():
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query = SentenceTransformer('all-MiniLM-L6-v2')
    return embed, query

# Cache data and FAISS index
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_data.csv")
    df['full_text'] = df['Product_name'].astype(str) + ". " + df['Description'].astype(str)
    embed, _ = load_models()
    docs = [Document(page_content=row['full_text'], metadata={"Product_name": row['Product_name']}) for _, row in df.iterrows()]
    index = FAISS.from_documents(docs, embed)
    return index

# Main app
def main():
    st.title("💻 Laptop Finder")
    st.write("Search, chat, and pick your laptop!")

    # Load models and index
    _, query_model = load_models()
    index = load_data()

    # Groq client
    client = Groq(api_key=os.environ['GROQ_API_KEY'])

    # Session state
    if 'chat' not in st.session_state:
        st.session_state.chat = []
        st.session_state.matches = []
        st.session_state.selected = None
        st.session_state.last_search = ""

    # Search bar
    st.subheader("Search")
    search = st.text_input("Query:", key="search")

    if search and search != st.session_state.last_search:
        st.session_state.last_search = search
        st.session_state.chat = []
        st.session_state.matches = []
        st.session_state.selected = None

        # Get 5 matches
        results = index.similarity_search(search, k=5)
        st.session_state.matches = [doc.page_content for doc in results]

        # Show matches
        st.subheader("Matches")
        for i, match in enumerate(st.session_state.matches, 1):
            st.markdown(f"**{i}.** {match}")

        # Groq response
        context = "\n".join(st.session_state.matches)
        prompt = f"User query: '{search}'\nDescriptions:\n{context}\nSuggest the best laptops."
        response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=300).choices[0].message.content

        st.subheader("Recommendation")
        st.markdown(response)

        st.session_state.chat.append({"role": "user", "content": search})
        st.session_state.chat.append({"role": "assistant", "content": response})

    # Chat to refine
    st.subheader("Chat")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    chat_query = st.chat_input("Refine (e.g., '16GB RAM'):")

    if chat_query:
        with st.chat_message("user"):
            st.markdown(chat_query)
        st.session_state.chat.append({"role": "user", "content": chat_query})

        # Combine queries for search
        all_queries = " ".join([msg["content"] for msg in st.session_state.chat if msg["role"] == "user"])
        results = index.similarity_search(all_queries, k=5)
        st.session_state.matches = [doc.page_content for doc in results]

        # Show updated matches
        st.subheader("Updated Matches")
        for i, match in enumerate(st.session_state.matches, 1):
            st.markdown(f"**{i}.** {match}")

        # Groq response
        context = "\n".join(st.session_state.matches)
        chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat])
        prompt = f"Query: '{chat_query}'\nHistory:\n{chat_history}\nDescriptions:\n{context}\nSuggest the best laptops. Confirm if user chooses (e.g., 'I choose MacBook Air')."
        response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=300).choices[0].message.content

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat.append({"role": "assistant", "content": response})

        # Check for chat selection
        if any(phrase in chat_query.lower() for phrase in ["i choose", "i want", "select", "pick"]):
            for match in st.session_state.matches:
                name = match.split(". ")[0]
                if name.lower() in chat_query.lower():
                    st.session_state.selected = match
                    break

    # Selection
    if st.session_state.matches:
        st.subheader("Pick a Laptop")
        pick = st.radio("Choose:", options=st.session_state.matches, format_func=lambda x: x.split(". ")[0], key="pick")
        if st.button("Confirm"):
            st.session_state.selected = pick
            st.success(f"Selected: **{pick.split('. ')[0]}**")

    # Final choice
    if st.session_state.selected:
        st.subheader("Your Choice")
        st.markdown(f"**{st.session_state.selected.split('. ')[0]}**")
        st.markdown(st.session_state.selected)

if __name__ == "__main__":
    main()