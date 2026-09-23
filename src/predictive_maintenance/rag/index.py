"""Simple FAISS RAG over synthetic maintenance docs (no LangChain)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np

from predictive_maintenance.config import DOCS_DIR, RAG_INDEX_DIR


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        # Prefer breaking on paragraph / sentence
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)
            if break_at <= start:
                break_at = text.rfind(". ", start, end)
            if break_at > start:
                end = break_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed_texts(texts: list[str], dim: int = 384) -> np.ndarray:
    """Deterministic bag-of-words hashing embedder (no external API required).

    Good enough for a small synthetic corpus demo; OpenAI embeddings optional later.
    """
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            matrix[i, idx] += sign
        norm = np.linalg.norm(matrix[i])
        if norm > 0:
            matrix[i] /= norm
    return matrix


def load_documents(docs_dir: Path | None = None) -> list[dict]:
    directory = docs_dir or DOCS_DIR
    if not directory.exists() or not list(directory.glob("*.md")):
        from predictive_maintenance.data.generate import write_synthetic_docs

        write_synthetic_docs(directory)
    docs = []
    for path in sorted(directory.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(body)):
            docs.append({"source": path.name, "chunk_id": i, "text": chunk})
    return docs


def build_index(
    docs_dir: Path | None = None,
    index_dir: Path | None = None,
) -> dict:
    out = index_dir or RAG_INDEX_DIR
    out.mkdir(parents=True, exist_ok=True)
    docs = load_documents(docs_dir)
    vectors = embed_texts([d["text"] for d in docs])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out / "index.faiss"))
    meta_path = out / "chunks.json"
    meta_path.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    return {"n_chunks": len(docs), "index_dir": str(out)}


class MaintenanceRAG:
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or RAG_INDEX_DIR
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "chunks.json"
        if not index_path.exists() or not meta_path.exists():
            build_index(index_dir=self.index_dir)
        self.index = faiss.read_index(str(index_path))
        self.chunks: list[dict] = json.loads(meta_path.read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = embed_texts([query])
        scores, idxs = self.index.search(q, min(k, len(self.chunks)))
        hits = []
        for score, idx in zip(scores[0], idxs[0], strict=True):
            if idx < 0:
                continue
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(score)
            hits.append(chunk)
        return hits

    def answer(self, query: str, k: int = 4) -> dict:
        hits = self.search(query, k=k)
        context = "\n\n---\n\n".join(
            f"[{h['source']}#{h['chunk_id']}] {h['text']}" for h in hits
        )
        return {
            "query": query,
            "context": context,
            "sources": [
                {"source": h["source"], "chunk_id": h["chunk_id"], "score": h["score"]}
                for h in hits
            ],
            "answer": None,  # filled by advisor LLM / mock
        }


def main() -> None:
    info = build_index()
    print(f"Built RAG index with {info['n_chunks']} chunks → {info['index_dir']}")


if __name__ == "__main__":
    main()
