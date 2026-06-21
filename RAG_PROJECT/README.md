# 📄 RAG — Document Question Answering

A simple, educational **Retrieval-Augmented Generation (RAG)** application built with Python and Streamlit. Upload a PDF or TXT document, ask a question, and get an answer grounded in your document's content.

This project is intentionally written in plain Python (no LangChain, no FAISS) so you can see every step of the RAG pipeline clearly.

---

## ✨ Features

- 📖 Loads `.pdf` and `.txt` files
- ✂️ Splits documents into overlapping chunks
- 🔢 Embeds chunks with a local Sentence-Transformers model (`all-MiniLM-L6-v2`)
- 💾 In-memory vector store with cosine similarity search (pure NumPy)
- 🤖 Generates answers via the **Groq API** (free tier, using `llama-3.1-8b-instant`)
- 🖥️ Simple Streamlit web UI

---

## 📂 Project Structure

```
RAG_PROJECT/
├── documents/
│   └── sample.txt              # Example document you can load
├── src/
│   ├── app.py                 # Streamlit UI (entry point)
│   ├── loader.py              # Reads PDF / TXT files
│   ├── chunker.py             # Splits text into overlapping chunks
│   ├── embedder.py            # Converts text → embedding vectors
│   ├── vector_store.py        # In-memory vector store + cosine similarity
│   └── generator.py           # Calls Groq LLM to generate the answer
├── requirements.txt
└── README.md
```

---

## 🧠 How the RAG Pipeline Works

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Load doc    │ →  │ Split into   │ →  │   Embed      │
│  (PDF/TXT)   │    │  chunks      │    │   chunks     │
└──────────────┘    └──────────────┘    └──────────────┘
                                               ↓
                                       ┌──────────────┐
                                       │ Vector Store │
                                       └──────────────┘
                                               ↑
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Display    │ ←  │   LLM call   │ ←  │  Retrieve    │
│   answer     │    │   (Groq)     │    │  top-K       │
└──────────────┘    └──────────────┘    └──────────────┘
                                               ↑
                                        ┌──────────────┐
                                        │  Embed user  │
                                        │  question    │
                                        └──────────────┘
```

1. **Load** — Read the raw text from a PDF or TXT file (`loader.py`)
2. **Chunk** — Split the text into overlapping 500-character windows (`chunker.py`)
3. **Embed** — Convert each chunk to a 384-dim vector using `all-MiniLM-L6-v2` (`embedder.py`)
4. **Store** — Keep chunks + vectors in an in-memory list (`vector_store.py`)
5. **Query** — Embed the user's question and find the top-3 most similar chunks via cosine similarity
6. **Generate** — Send the question + retrieved chunks to the LLM and display the answer (`generator.py`)

---

## 🛠️ Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** (Python package manager)
- A **Groq API key** (free) — get one at [https://console.groq.com](https://console.groq.com)

> 💡 The first run will download the embedding model (~90 MB) from Hugging Face.

---

## 🚀 Installation & Setup

### 1. Clone / navigate to the project

```powershell
cd C:\Users\jaiva\OneDrive\Desktop\CELEBAL_ASSIGNMENTS\RAG_PROJECT
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you're on macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs:

| Package              | Purpose                                |
| -------------------- | -------------------------------------- |
| `streamlit`          | Web UI framework                       |
| `sentence-transformers` | Local embedding model                |
| `PyMuPDF`            | PDF parsing (imported as `fitz`)       |
| `numpy`              | Vector math (cosine similarity)         |
| `groq`               | LLM API client                         |

---

## ▶️ How to Run

### Start the Streamlit app

From the `RAG_PROJECT` directory (with your virtual environment activated):

```powershell
streamlit run src/app.py
```

A browser tab will open at `http://localhost:8501`.

### Using the app

1. Open the **sidebar** (left panel) and paste your **Groq API key**
2. In the main area, **upload a PDF or TXT file** (or use the included `documents/sample.txt`)
3. Wait for the spinner — the document is being chunked, embedded, and indexed
4. **Type a question** about the document and click **"Get Answer"**
5. The answer appears, and you can expand the **"Context the LLM used"** section to see which chunks were retrieved

### Example questions for `documents/sample.txt`

- *"What is Retrieval-Augmented Generation?"*
- *"What are the advantages of RAG systems?"*
- *"Name some popular vector databases."*

---

## 🔑 Getting a Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key and paste it into the app's sidebar

> 🔒 The key is only used in-memory for the current session. It is **not** stored anywhere.

---

## ⚙️ Configuration

You can tweak the chunking parameters in `src/app.py`:

```python
chunks = split_into_chunks(raw_text, chunk_size=500, overlap=50)
```

| Parameter    | Default | Effect                                              |
| ------------ | ------- | --------------------------------------------------- |
| `chunk_size` | `500`   | Max characters per chunk. Larger = more context per chunk but less precise retrieval. |
| `overlap`    | `50`    | Characters shared between consecutive chunks. Prevents losing context at boundaries. |

To change the LLM model, edit `src/generator.py`:

```python
model="llama-3.1-8b-instant"   # swap with any Groq-supported model
```

---

## 🐛 Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'fitz'` | Run `pip install PyMuPDF` |
| `ModuleNotFoundError: No module named 'groq'` | Run `pip install groq` |
| First run hangs / is slow | The embedding model is downloading (~90 MB). Be patient. |
| `401 Unauthorized` from Groq | Your API key is missing or wrong. Re-paste it in the sidebar. |
| `Please upload a document first.` | You clicked **Get Answer** before uploading a file. |
| Streamlit can't find modules | Make sure you launched the app from the `RAG_PROJECT` directory, not from inside `src/`. |

---

## 📚 Learn More

- [RAG explained (AWS)](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
- [Sentence-Transformers docs](https://www.sbert.net/)
- [Streamlit docs](https://docs.streamlit.io/)
- [Groq API docs](https://console.groq.com/docs/overview)

---

## 📝 License

This is a learning project — feel free to use, modify, and extend it.
