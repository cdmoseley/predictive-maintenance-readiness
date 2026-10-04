"""Simple RAG over synthetic maintenance docs.

FAISS is preferred when available; falls back to pure-numpy cosine search so
Streamlit Cloud can still serve engineer search if faiss-cpu fails to import.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from predictive_maintenance.config import DOCS_DIR, RAG_INDEX_DIR

_FAISS = None
_FAISS_TRIED = False


def _get_faiss():
    """Lazy-import faiss; return None if unavailable."""
    global _FAISS, _FAISS_TRIED
    if _FAISS_TRIED:
        return _FAISS
    _FAISS_TRIED = True
    try:
        import faiss as _faiss_mod

        _FAISS = _faiss_mod
    except Exception:
        _FAISS = None
    return _FAISS


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
    """Deterministic bag-of-words hashing embedder (no external API required)."""
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


def _numpy_search(vectors: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Inner-product search without FAISS (vectors assumed L2-normalized)."""
    scores = vectors @ query.T  # (n, 1)
    flat = scores[:, 0]
    k = min(k, len(flat))
    if k <= 0:
        return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)
    idxs = np.argpartition(-flat, k - 1)[:k]
    idxs = idxs[np.argsort(-flat[idxs])]
    return flat[idxs][None, :].astype(np.float32), idxs[None, :].astype(np.int64)


def build_index(
    docs_dir: Path | None = None,
    index_dir: Path | None = None,
) -> dict:
    out = index_dir or RAG_INDEX_DIR
    out.mkdir(parents=True, exist_ok=True)
    docs = load_documents(docs_dir)
    vectors = embed_texts([d["text"] for d in docs])
    # Always persist numpy vectors for Cloud fallback
    np.save(out / "vectors.npy", vectors)
    faiss = _get_faiss()
    if faiss is not None:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(out / "index.faiss"))
    meta_path = out / "chunks.json"
    meta_path.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    return {"n_chunks": len(docs), "index_dir": str(out), "faiss": faiss is not None}


class MaintenanceRAG:
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or RAG_INDEX_DIR
        index_path = self.index_dir / "index.faiss"
        vectors_path = self.index_dir / "vectors.npy"
        meta_path = self.index_dir / "chunks.json"
        if not meta_path.exists() or (not index_path.exists() and not vectors_path.exists()):
            build_index(index_dir=self.index_dir)

        self.chunks: list[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
        self._faiss_index: Any = None
        self._vectors: np.ndarray | None = None

        faiss = _get_faiss()
        if faiss is not None and index_path.exists():
            try:
                self._faiss_index = faiss.read_index(str(index_path))
            except Exception:
                self._faiss_index = None

        if self._faiss_index is None:
            if vectors_path.exists():
                self._vectors = np.load(vectors_path)
            else:
                # Rebuild vectors from chunk texts
                self._vectors = embed_texts([c["text"] for c in self.chunks])
                np.save(vectors_path, self._vectors)

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = embed_texts([query])
        k_eff = min(k, len(self.chunks))
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q, k_eff)
        else:
            assert self._vectors is not None
            scores, idxs = _numpy_search(self._vectors, q, k_eff)

        hits = []
        # Py3.9-safe: do not pass strict to zip()
        for score, idx in zip(scores[0], idxs[0]):
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
