# ============================================================
# generator.py
# PURPOSE: Generate the final answer using an LLM.
#
# This is the "Generation" step in RAG (Retrieval-Augmented Generation).
# We take the user's question + the retrieved chunks, build a prompt,
# and ask the LLM to answer based ONLY on that context.
#
# WHY GROQ?
# Groq has a free API tier with fast inference.
# Get your free key at: https://console.groq.com
# The code looks almost identical to OpenAI — easy to swap.
# ============================================================

from groq import Groq


def generate_answer(question: str, context_chunks: list, api_key: str) -> str:
    """
    Builds a prompt from the question + context chunks and calls the LLM.

    question       : the user's question string
    context_chunks : list of dicts like [{"text": "...", "score": 0.9}, ...]
    api_key        : your Groq API key (entered in the UI sidebar)

    Returns the LLM's answer as a plain string.
    """

    # ── 1. Build the context string ──────────────────────────────────────────
    # Combine the top retrieved chunks into one block of text.
    # This is the "Augmentation" step: we inject document knowledge into the prompt.
    context = "\n\n---\n\n".join(chunk["text"] for chunk in context_chunks)

    # ── 2. Build the prompt ───────────────────────────────────────────────────
    # We explicitly tell the model to ONLY use the provided context.
    # This prevents the model from hallucinating answers from its training data.
    prompt = f"""You are a helpful assistant. Answer the question below using ONLY the context provided.
If the context does not contain enough information to answer, say: "I don't know based on the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    # ── 3. Call the LLM ───────────────────────────────────────────────────────
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",     # free, fast model on Groq
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,             # keep answers concise
        temperature=0.2,            # low = more factual, less creative
    )

    # ── 4. Extract and return the answer text ─────────────────────────────────
    answer = response.choices[0].message.content
    return answer.strip()