import ollama

from vector_store import search_chunks


# MODEL CONFIGURATION

LLM_MODEL = "gemma4:12b"


# RAG PIPELINE

def generate_answer(question):

    # STEP 1: SEARCH RELEVANT DOCUMENTS

    results = search_chunks(
        question,
        n_results=5
    )

    if not results:

        return {
            "answer": "I could not find relevant information in the uploaded college documents. Please contact the college administration.",
            "sources": []
        }

    # STEP 2: CHECK RETRIEVAL RELEVANCE

    best_distance = results[0]["distance"]

    if best_distance > 0.75:

        return {
            "answer": "I could not find sufficient information in the uploaded documents to answer this question reliably.",
            "sources": []
        }

    # STEP 3: PREPARE CONTEXT

    context = ""

    sources = []

    for result in results:

        context += (
            f"\nSource: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"Content: {result['text']}\n"
        )

        source = {
            "name": result["source"],
            "page": result["page"]
        }

        if source not in sources:
            sources.append(source)

    # STEP 4: CREATE PROMPT

    prompt = f"""
You are a College Knowledge Assistant.

Answer the student's question using ONLY the
provided college documents.

RULES:

1. Never invent information.
2. Do not use outside knowledge.
3. If information is unavailable, clearly say so.
4. If documents contain conflicting information,
   explain the conflict and cite both sources.
5. Keep answers clear and student-friendly.
6. Do not claim something is official unless
   the documents support it.

COLLEGE DOCUMENTS:

{context}

STUDENT QUESTION:

{question}

Provide your answer based only on the documents.
"""

    # STEP 5: GENERATE ANSWER USING GEMMA

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.message.content

    return {
        "answer": answer,
        "sources": sources
    }