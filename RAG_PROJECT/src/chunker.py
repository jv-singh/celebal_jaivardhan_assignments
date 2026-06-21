# ============================================================
# chunker.py
# PURPOSE: Split the large document text into small, overlapping chunks.
#
# WHY CHUNK?
# - Embedding models have a max token limit (can't embed a 50-page PDF at once)
# - Smaller chunks = more precise retrieval (we find exactly the right paragraph)
# - Overlap between chunks prevents losing context at chunk boundaries
# ============================================================


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits text into overlapping character-level chunks.

    chunk_size : max characters per chunk  (500 chars ≈ ~100 words)
    overlap    : characters shared between consecutive chunks
                 e.g. the last 50 chars of chunk 1 are also the first 50 of chunk 2

    Visual example (chunk_size=10, overlap=3):
      Text : "Hello World Python Rocks"
      C1   : "Hello Worl"
      C2   : "orld Pytho"   <- 3-char overlap with C1
      C3   : "thon Rocks"   <- 3-char overlap with C2
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Skip chunks that are only whitespace (blank pages, etc.)
        if chunk.strip():
            chunks.append(chunk.strip())

        # Advance window by (chunk_size - overlap)
        # This is what creates the overlap: we step back by `overlap` chars
        start += chunk_size - overlap

    print(f"[Chunker] Split text into {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, overlap={overlap})")
    return chunks