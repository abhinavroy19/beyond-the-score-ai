import streamlit as st
from openai import OpenAI
import chromadb

# OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get collection
collection = chroma_client.get_collection(name="sports_insights")

# Page config
st.set_page_config(
    page_title="BeyondTheScore AI",
    page_icon="🏆",
    layout="centered"
)

# Title
st.title("🏆 BeyondTheScore AI")

st.markdown("""
### AI-Powered Sports Storytelling & Insights Assistant

Explore resilience, leadership, pressure handling, mental toughness, and comeback stories in sports using semantic AI.
""")

st.divider()

# Sample prompts
st.subheader("🎯 Sports Insight Explorer")

sample_prompts = [
    "Show athletes known for resilience after setbacks.",
    "Which athletes demonstrated calm leadership under pressure?",
    "Show sports stories about mental toughness.",
    "Athletes who transformed criticism into success."
]

selected_prompt = st.selectbox(
    "Try a sample question:",
    sample_prompts
)

# User input
query = st.text_input(
    "Or ask your own sports insight question:",
    selected_prompt
)

# Generate button
if st.button("🚀 Generate Insights"):

    with st.spinner("Analyzing sports stories and generating insights..."):

        # Create embedding
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )

        query_embedding = embedding_response.data[0].embedding

        # Semantic search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        retrieved_stories = "\n\n".join(results["documents"][0])

        # Prompt
        prompt = f"""
        You are BeyondTheScore AI, a sports storytelling assistant.

        User question:
        {query}

        Relevant sports stories:
        {retrieved_stories}

        Create a short, engaging response with storytelling and insights.
        Keep it human, inspiring, and easy to read.
        """

        # LLM response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        final_response = response.choices[0].message.content

        st.divider()

        st.subheader("🧠 AI Insights")

        st.write(final_response)

        st.divider()

        st.markdown(
    "⚙️ Powered by Semantic Search, Vector Retrieval, and AI Storytelling"
)

        