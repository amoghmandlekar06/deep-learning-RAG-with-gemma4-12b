import chromadb
import ollama
import hashlib
import os


# DATABASE CONFIGURATION

DATABASE_PATH = "database/chroma_db"

COLLECTION_NAME = "college_documents"

EMBEDDING_MODEL = "nomic-embed-text"


# CREATE DATABASE

os.makedirs(DATABASE_PATH, exist_ok=True)

client = chromadb.PersistentClient(
    path=DATABASE_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# GENERATE EMBEDDINGS

def generate_embeddings(texts):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return response.embeddings


# STORE DOCUMENTS

def add_chunks(chunks):

    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)

    ids = []
    metadatas = []

    for chunk in chunks:

        unique_text = (
            f"{chunk['source']}_"
            f"{chunk['page']}_"
            f"{chunk['chunk_index']}_"
            f"{chunk['text']}"
        )

        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

        metadatas.append({
            "source": chunk["source"],
            "page": int(chunk["page"]),
            "chunk_index": int(chunk["chunk_index"])
        })

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


# SEARCH DOCUMENTS

def search_chunks(question, n_results=5):

    if collection.count() == 0:
        return []

    question_embedding = generate_embeddings(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(n_results, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    for index, text in enumerate(results["documents"][0]):

        retrieved.append({

            "text": text,

            "source": results["metadatas"][0][index]["source"],

            "page": results["metadatas"][0][index]["page"],

            "distance": results["distances"][0][index]

        })

    return retrieved


# DATABASE STATISTICS

def get_collection_count():

    return collection.count()