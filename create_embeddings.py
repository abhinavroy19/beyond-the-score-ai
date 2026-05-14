import pandas as pd
from openai import OpenAI
import chromadb

# Load CSV
df = pd.read_csv("sports_insights.csv")
print(df.columns)

# OpenAI client
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = chroma_client.create_collection(name="sports_insights")

# Loop through stories
for index, row in df.iterrows():

    text = f"""
    Athlete: {row['athlete']}
    Sport: {row['sport']}
    Theme: {row['theme']}
    Traits: {row['traits']}
    Story: {row['story_summary']}
    Insight: {row['insight']}
    """

    # Generate embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    # Store in ChromaDB
    collection.add(
        ids=[str(index)],
        embeddings=[embedding],
        documents=[text]
    )

print("Embeddings created successfully!")