from openai import OpenAI
import chromadb

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="sports_insights")

query = "Show athletes known for resilience after setbacks."

# Create embedding for user query
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)

query_embedding = embedding_response.data[0].embedding

# Retrieve top 3 relevant stories
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

retrieved_stories = "\n\n".join(results["documents"][0])

# LLM response layer
prompt = f"""
You are BeyondTheScore AI, a sports storytelling assistant.

User question:
{query}

Relevant sports stories retrieved:
{retrieved_stories}

Create a short, engaging response that explains which athletes show resilience after setbacks.
Keep it insightful, human, and easy to read.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)