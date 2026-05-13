from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@dataclass
class DocumentChunk:
    doc_id: str
    doc_name: str
    chunk_index: int
    text: str


def clean_text(text: str) -> str:
    """Normalize whitespace and remove repeated blank space from extracted text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text_file(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to extract PDF files: pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(data: bytes) -> str:
    try:
        import docx

        document = docx.Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except ImportError:
        # Lightweight stdlib fallback for simple .docx files.
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for paragraph in root.findall(".//w:p", namespace):
            texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
            if texts:
                paragraphs.append("".join(texts))
        return "\n".join(paragraphs)


def extract_text(filename: str, data: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension in {".txt", ".md"}:
        return clean_text(read_text_file(data))
    if extension == ".pdf":
        return clean_text(extract_pdf(data))
    if extension == ".docx":
        return clean_text(extract_docx(data))
    raise ValueError(f"Unsupported file type: {extension}")


def chunk_text(text: str, doc_id: str, doc_name: str, chunk_size: int = 260, overlap: int = 45) -> list[DocumentChunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[DocumentChunk] = []
    step = max(1, chunk_size - overlap)
    for chunk_index, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(
            DocumentChunk(
                doc_id=doc_id,
                doc_name=doc_name,
                chunk_index=chunk_index,
                text=" ".join(window),
            )
        )
        if start + chunk_size >= len(words):
            break
    return chunks


class EmbeddingBackend:
    def __init__(self) -> None:
        self.kind = "tfidf"
        self.model = None
        self.vectorizer: TfidfVectorizer | None = None
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.kind = "sentence-transformer"
        except Exception:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=12000,
            )

    def fit_transform(self, texts: list[str]):
        if self.kind == "sentence-transformer":
            return self.model.encode(texts, normalize_embeddings=True, batch_size=4)
        assert self.vectorizer is not None
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        if self.kind == "sentence-transformer":
            return self.model.encode(texts, normalize_embeddings=True, batch_size=4)
        assert self.vectorizer is not None
        return self.vectorizer.transform(texts)


class DocumentSearchEngine:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("No text chunks were created. Upload documents with extractable text.")
        self.chunks = chunks
        self.backend = EmbeddingBackend()
        self.embeddings = self.backend.fit_transform([chunk.text for chunk in chunks])
        self.faiss_index = self._build_faiss_index()

    def _build_faiss_index(self):
        if self.backend.kind != "sentence-transformer":
            return None
        try:
            import faiss

            vectors = np.asarray(self.embeddings, dtype="float32")
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            return index
        except Exception:
            return None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.backend.transform([query])

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(np.asarray(query_vector, dtype="float32"), top_k)
            pairs = zip(indices[0].tolist(), scores[0].tolist(), strict=False)
        else:
            similarities = cosine_similarity(query_vector, self.embeddings)[0]
            ranked = np.argsort(similarities)[::-1][:top_k]
            pairs = ((int(index), float(similarities[index])) for index in ranked)

        results = []
        for index, score in pairs:
            chunk = self.chunks[index]
            results.append(
                {
                    "score": score,
                    "doc_id": chunk.doc_id,
                    "doc_name": chunk.doc_name,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                }
            )
        return results


def sentence_split(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [piece.strip() for piece in pieces if len(piece.split()) >= 5]


class Summarizer:
    def __init__(self) -> None:
        self.pipeline = None
        try:
            from transformers import pipeline

            self.pipeline = pipeline("summarization", model="t5-small")
        except Exception:
            self.pipeline = None

    def summarize(self, chunks: Iterable[str], query: str, max_sentences: int = 4) -> str:
        combined = clean_text("\n".join(chunks))
        if not combined:
            return "No relevant content found."

        if self.pipeline is not None:
            trimmed = " ".join(combined.split()[:700])
            summary = self.pipeline(trimmed, max_length=130, min_length=35, do_sample=False)
            return summary[0]["summary_text"].strip()

        return self._extractive_summary(combined, query, max_sentences=max_sentences)

    def _extractive_summary(self, text: str, query: str, max_sentences: int) -> str:
        sentences = sentence_split(text)
        if not sentences:
            return " ".join(text.split()[:90])

        corpus = [query] + sentences
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus)
        query_scores = cosine_similarity(matrix[0], matrix[1:])[0]
        centrality = cosine_similarity(matrix[1:], matrix[1:]).mean(axis=1)
        length_penalty = np.array([1.0 / (1.0 + abs(len(sentence.split()) - 24) / 40) for sentence in sentences])
        scores = (0.68 * query_scores + 0.32 * centrality) * length_penalty
        selected = sorted(np.argsort(scores)[::-1][:max_sentences])
        return " ".join(sentences[index] for index in selected)


def ingest_uploaded_files(files: list[tuple[str, bytes]], chunk_size: int, overlap: int) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for doc_number, (filename, data) in enumerate(files, start=1):
        text = extract_text(filename, data)
        chunks.extend(chunk_text(text, doc_id=f"doc-{doc_number}", doc_name=filename, chunk_size=chunk_size, overlap=overlap))
    return chunks


def precision_recall_at_k(results: list[dict], relevant: set[tuple[str, int]], k: int) -> tuple[float, float]:
    retrieved = {(item["doc_name"], item["chunk_index"]) for item in results[:k]}
    hits = len(retrieved & relevant)
    precision = hits / max(1, k)
    recall = hits / max(1, len(relevant))
    return precision, recall


def rouge_l(candidate: str, reference: str) -> float:
    a = candidate.lower().split()
    b = reference.lower().split()
    if not a or not b:
        return 0.0
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, word_a in enumerate(a, start=1):
        for j, word_b in enumerate(b, start=1):
            table[i][j] = table[i - 1][j - 1] + 1 if word_a == word_b else max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    precision = lcs / len(a)
    recall = lcs / len(b)
    return (2 * precision * recall) / max(precision + recall, 1e-9)


def simple_bleu(candidate: str, reference: str) -> float:
    candidate_words = candidate.lower().split()
    reference_words = set(reference.lower().split())
    if not candidate_words:
        return 0.0
    overlap = sum(1 for word in candidate_words if word in reference_words)
    brevity = min(1.0, math.exp(1 - len(reference.split()) / max(1, len(candidate_words))))
    return brevity * overlap / len(candidate_words)


def run_evaluation(engine: DocumentSearchEngine, summarizer: Summarizer, test_cases: list[dict], top_k: int) -> list[dict]:
    rows = []
    for case in test_cases:
        results = engine.search(case["query"], top_k=top_k)
        relevant = {(item["doc_name"], int(item["chunk_index"])) for item in case.get("relevant_chunks", [])}
        precision, recall = precision_recall_at_k(results, relevant, top_k)
        summary = summarizer.summarize([item["text"] for item in results], case["query"])
        reference = case.get("reference_summary", "")
        rows.append(
            {
                "query": case["query"],
                f"precision@{top_k}": round(precision, 3),
                f"recall@{top_k}": round(recall, 3),
                "rouge_l": round(rouge_l(summary, reference), 3) if reference else None,
                "bleu": round(simple_bleu(summary, reference), 3) if reference else None,
                "summary": summary,
            }
        )
    return rows


def streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="AI Document Search", layout="wide")
    st.title("AI-Powered Document Search and Summarization")

    with st.sidebar:
        st.header("Ingestion")
        chunk_size = st.slider("Words per chunk", 120, 500, 260, 20)
        overlap = st.slider("Chunk overlap", 0, 100, 45, 5)
        top_k = st.slider("Top-k results", 1, 10, 5)
        st.caption("Supports TXT/MD by default. Install optional packages for PDF, Word, FAISS, and transformer models.")

    uploads = st.file_uploader(
        "Upload documents",
        type=[extension.strip(".") for extension in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

    if uploads and st.button("Ingest documents", type="primary"):
        files = [(upload.name, upload.getvalue()) for upload in uploads]
        with st.spinner("Extracting text, chunking, embedding, and indexing..."):
            chunks = ingest_uploaded_files(files, chunk_size=chunk_size, overlap=overlap)
            st.session_state.engine = DocumentSearchEngine(chunks)
            st.session_state.summarizer = Summarizer()
            st.session_state.chunks = chunks
        st.success(f"Indexed {len(chunks)} chunks from {len(files)} document(s).")

    engine: DocumentSearchEngine | None = st.session_state.get("engine")
    if engine is None:
        st.info("Upload documents and click ingest to build the searchable index.")
        return

    st.caption(f"Embedding backend: {engine.backend.kind}. Vector search: {'FAISS IndexFlatIP' if engine.faiss_index else 'cosine similarity'}.")
    query = st.text_input("Search query", placeholder="Ask a question about your uploaded documents")
    if not query:
        return

    results = engine.search(query, top_k=top_k)
    summarizer: Summarizer = st.session_state.summarizer
    summary = summarizer.summarize([item["text"] for item in results], query)

    st.subheader("Summary")
    st.write(summary)

    st.subheader("Relevant sections")
    for rank, result in enumerate(results, start=1):
        with st.expander(f"{rank}. {result['doc_name']} - chunk {result['chunk_index']} - score {result['score']:.3f}", expanded=rank == 1):
            st.write(result["text"])

    st.subheader("Evaluate")
    st.caption("Paste a JSON list with query, relevant_chunks, and reference_summary fields.")
    default_case = json.dumps(
        [
            {
                "query": query,
                "relevant_chunks": [{"doc_name": results[0]["doc_name"], "chunk_index": results[0]["chunk_index"]}],
                "reference_summary": summary,
            }
        ],
        indent=2,
    )
    evaluation_json = st.text_area("Evaluation set", value=default_case, height=180)
    if st.button("Run evaluation"):
        try:
            rows = run_evaluation(engine, summarizer, json.loads(evaluation_json), top_k=top_k)
            st.dataframe(rows, use_container_width=True)
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")


def load_folder(folder: Path) -> list[tuple[str, bytes]]:
    files = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append((path.name, path.read_bytes()))
    return files


def run_cli(args: argparse.Namespace) -> None:
    files = load_folder(Path(args.folder))
    if not files:
        raise SystemExit(f"No supported documents found in {args.folder}")

    chunks = ingest_uploaded_files(files, chunk_size=args.chunk_size, overlap=args.overlap)
    engine = DocumentSearchEngine(chunks)
    summarizer = Summarizer()
    results = engine.search(args.query, top_k=args.top_k)
    print(f"Indexed {len(chunks)} chunks with {engine.backend.kind}.")
    print("\nSUMMARY\n" + summarizer.summarize([item["text"] for item in results], args.query))
    print("\nTOP RESULTS")
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result["text"].split()[:45])
        print(f"{rank}. {result['doc_name']} chunk {result['chunk_index']} score={result['score']:.3f}: {preview}...")


def watch_folder(args: argparse.Namespace) -> None:
    folder = Path(args.folder)
    seen: dict[str, float] = {}
    print(f"Watching {folder} for new or changed documents. Press Ctrl+C to stop.")
    while True:
        current = {
            path.name: path.stat().st_mtime
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        }
        changed = [name for name, modified in current.items() if seen.get(name) != modified]
        if changed:
            print(f"Detected updates: {', '.join(changed)}")
            try:
                run_cli(args)
            except Exception as exc:
                print(f"Processing failed: {exc}")
            seen = current
        time.sleep(args.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-powered document search and summarization.")
    parser.add_argument("--cli", action="store_true", help="Run a terminal query instead of the Streamlit UI.")
    parser.add_argument("--watch", action="store_true", help="Continuously reprocess documents when the folder changes.")
    parser.add_argument("--folder", default="documents", help="Folder used by --cli and --watch.")
    parser.add_argument("--query", default="What are the most important points?", help="Search query for --cli and --watch.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=260)
    parser.add_argument("--overlap", type=int, default=45)
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.watch:
        watch_folder(arguments)
    elif arguments.cli:
        run_cli(arguments)
    else:
        try:
            streamlit_app()
        except ModuleNotFoundError as exc:
            if exc.name != "streamlit":
                raise
            print("Streamlit is not installed. Run `pip install -r requirements.txt`, then `streamlit run hackathon.py`.")
            print("You can still use the CLI: `python3 hackathon.py --cli --folder documents --query \"your question\"`.")
