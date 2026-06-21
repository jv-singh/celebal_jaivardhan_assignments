# ============================================================
# embedder.py
# PURPOSE: Convert text (chunks or queries) into embedding vectors.
#
# WHAT IS AN EMBEDDING?
# An embedding is a list of numbers (a vector) that represents
# the *meaning* of a piece of text.
# Texts with similar meaning → vectors that are close together.
# Texts with different meaning → vectors that are far apart.
#
# We use "all-MiniLM-L6-v2" — a small, fast, free model.
# It outputs a 384-dimensional vector for any piece of text.
# Downloaded automatically (~90 MB) on first use.
# ============================================================

from sentence_transformers import SentenceTransformer

# Load the model once at module level.
# Loading it here (not inside a function) means it's only loaded once,
# even if get_embeddings() is called many times.
print("[Embedder] Loading embedding model (downloads once on first run)...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[Embedder] Model ready.")


def get_embeddings(texts: list):
    """
    Embeds a list of text strings.
    Returns a numpy array of shape (len(texts), 384).

    Used for: embedding all document chunks during indexing.
    """
    # show_progress_bar=False keeps the console clean;
    # set to True if you want to see a progress bar for large docs
    embeddings = _model.encode(texts, show_progress_bar=False)
    print(f"[Embedder] Created {len(embeddings)} embeddings.")
    return embeddings


def get_single_embedding(text: str):
    """
    Embeds a single string.
    Returns a numpy array of shape (384,).

    Used for: embedding the user's question before searching.
    """
    # encode() always returns a 2D array; [0] gives us the first (only) row
    return _model.encode([text])[0]