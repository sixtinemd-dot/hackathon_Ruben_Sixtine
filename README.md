# AI-Powered Document Search and Summarization

This project ingests documents, splits them into searchable chunks, retrieves the most relevant sections for a user query, and generates a concise summary from the retrieved content.

## Supported Documents

- Plain text: `.txt`
- Markdown: `.md`
- PDF: `.pdf` with `pypdf`
- Word: `.docx` with `python-docx` or the built-in XML fallback for simple files

## Architecture

```text
Upload / Folder Input
  -> Text Extraction
  -> Cleaning and Chunking
  -> Embedding
  -> Vector Index
  -> Query Search
  -> Summarization
  -> Results + Evaluation
```

The app automatically uses the best available local backend:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` when installed, otherwise TF-IDF from scikit-learn.
- Vector search: FAISS `IndexFlatIP` when installed with sentence-transformer embeddings, otherwise cosine similarity.
- Summarization: `transformers` with `t5-small` when installed, otherwise a lightweight extractive summarizer.

## Quick Start

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the Streamlit interface:

```bash
streamlit run hackathon.py
```

Then upload one or more documents, click **Ingest documents**, enter a search query, and review the summary plus top matching sections.

## CLI Mode

You can also run the engine without Streamlit:

```bash
mkdir -p documents
python3 hackathon.py --cli --folder documents --query "What are the key decisions?"
```

The CLI indexes every supported file in the folder, prints a summary, and lists the top matching chunks.

## Automated Processing

The bonus file-watcher mode reprocesses a folder whenever supported documents are added or changed:

```bash
python3 hackathon.py --watch --folder documents --query "What changed?"
```

## Evaluation

The Streamlit app includes a JSON evaluation area. Each test case should look like this:

```json
[
  {
    "query": "What are the key decisions?",
    "relevant_chunks": [
      {"doc_name": "meeting-notes.txt", "chunk_index": 0}
    ],
    "reference_summary": "The team decided to build a Streamlit prototype with local semantic search."
  }
]
```

Metrics included:

- Search: `precision@k`, `recall@k`
- Summarization: lightweight BLEU approximation, ROUGE-L

Perplexity is not computed in the local fallback because it requires a language model probability distribution. If you install a causal language model, it can be added as a separate evaluation function.

## CPU Choices and Tradeoffs

- Chunking defaults to 260 words with 45 words of overlap to keep context manageable on CPU.
- TF-IDF fallback is fast and works offline, but it is lexical rather than semantic.
- `all-MiniLM-L6-v2` gives better semantic search while staying CPU-friendly.
- FAISS uses a simple flat index by default. This is slower than IVF on very large corpora but avoids training overhead and is reliable for hackathon-scale datasets.
- The extractive summarizer is deterministic and cheap. Transformer summarization is more fluent but slower and requires model downloads.

## Project Files

- `hackathon.py`: ingestion, search engine, summarization, Streamlit UI, CLI mode, watcher, and evaluation helpers.
- `requirements.txt`: optional packages for the full app experience.
