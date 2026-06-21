# ============================================================
# loader.py
# PURPOSE: Read your document (PDF or .txt) and return raw text.
# This is the very first step in the RAG pipeline.
# No AI happens here — just plain file reading.
# ============================================================

import fitz  # PyMuPDF — used only for PDFs


def load_pdf(file_path: str) -> str:
    """
    Opens a PDF and reads every page, returning all text as one big string.
    fitz (PyMuPDF) handles the low-level PDF parsing for us.
    """
    text = ""

    # fitz.open() reads the PDF file from disk
    with fitz.open(file_path) as pdf:
        for page in pdf:
            # get_text() extracts plain text from a single page
            text += page.get_text()

    return text


def load_txt(file_path: str) -> str:
    """
    Opens a plain .txt file and returns its contents.
    Straightforward — no special library needed.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(file_path: str) -> str:
    """
    Entry point: auto-detects file type and calls the right loader.
    Supported types: .pdf and .txt
    """
    if file_path.lower().endswith(".pdf"):
        print(f"[Loader] Reading PDF: {file_path}")
        return load_pdf(file_path)

    elif file_path.lower().endswith(".txt"):
        print(f"[Loader] Reading TXT: {file_path}")
        return load_txt(file_path)

    else:
        raise ValueError("Unsupported file type. Please use a .pdf or .txt file.")