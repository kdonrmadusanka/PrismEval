from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

MIN_BUDGET = 256
MAX_BUDGET = 4096
CHUNK_TOKENS = 160          # target size of one knowledge chunk
EMBED_DIM = 512             # size of the hashed dense vectors
MAX_QUERY_TOKENS = 512      # cap on query length used for scoring

# Words, with each CJK character as its own token.
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[^\W_\u4e00-\u9fff]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Placeholder knowledge base, used only when RAG_DOCS_DIR has no .txt/.md files.
# Replace with your real policy and brand documents.
DEFAULT_KNOWLEDGE_BASE: List[Tuple[str, str]] = [
    ("policy/claims",
     "Claims must be verifiable. Avoid absolute or guaranteed statements such as "
     "always, never, guaranteed or absolutely unless supported by evidence."),
    ("policy/privacy",
     "Never disclose personal data such as names, contact details, grades or "
     "identifiers of other people without consent."),
    ("policy/respect",
     "Communication must be respectful and free of harassment, discrimination, "
     "insults or demeaning language toward any person or group."),
    ("brand/tone",
     "The brand voice is warm, professional and empathetic. Use plain language, "
     "acknowledge the reader's perspective and avoid sarcasm or condescension."),
    ("brand/clarity",
     "Messages should state their purpose in the first sentence, use short "
     "sentences and end with one clear call to action."),
    ("brand/engagement",
     "Engaging messages open with a relevant hook, address the audience "
     "directly, use concrete examples and show a benefit to the reader."),
]


# ----------------------------------------------------------------------
# Text utilities and token scaling
# ----------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 chars per token for Latin text, 1 per CJK char."""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    return math.ceil(cjk + (len(text) - cjk) / 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Longest prefix of `text` within `max_tokens`, cut at a word boundary if possible."""
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1

    cut = text[: max(lo, 1)]
    if lo < len(text):
        space = cut.rfind(" ")
        if space > lo * 0.6:
            cut = cut[:space]
    return cut.rstrip()


def _chunk_text(text: str, max_tokens: int) -> List[str]:
    """Split text into paragraph-based chunks of at most ~max_tokens."""
    chunks: List[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if estimate_tokens(paragraph) > max_tokens:
            flush()
            remaining = paragraph
            while remaining:
                piece = truncate_to_tokens(remaining, max_tokens)
                if not piece:  # safety: always make progress
                    piece = remaining[:1]
                chunks.append(piece)
                remaining = remaining[len(piece):].strip()
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if estimate_tokens(candidate) > max_tokens:
            flush()
            current = paragraph
        else:
            current = candidate

    flush()
    return chunks


# ----------------------------------------------------------------------
# Sparse retrieval: BM25
# ----------------------------------------------------------------------
class _BM25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.tf = [Counter(d) for d in docs]

        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        self.idf: Dict[str, float] = {
            term: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def score(self, query_tokens: Sequence[str]) -> List[float]:
        scores = [0.0] * self.n_docs
        if not self.n_docs or self.avg_len <= 0:
            return scores

        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                freq = counts.get(term, 0)
                if not freq:
                    continue
                norm = 1.0 - self.b + self.b * self.doc_len[i] / self.avg_len
                scores[i] += idf * freq * (self.k1 + 1.0) / (freq + self.k1 * norm)
        return scores


# ----------------------------------------------------------------------
# Dense retrieval: hashed embeddings (drop-in replaceable by a real model)
# ----------------------------------------------------------------------
def _embed(tokens: Sequence[str]) -> Dict[int, float]:
    """Signed feature-hashing vector over words, char trigrams and bigrams, L2-normalised."""
    feats: Counter = Counter()
    for i, tok in enumerate(tokens):
        feats[tok] += 1.0
        if len(tok) > 4:
            padded = f"#{tok}#"
            for j in range(len(padded) - 2):
                feats["g:" + padded[j:j + 3]] += 0.5
        if i + 1 < len(tokens):
            feats[f"b:{tok} {tokens[i + 1]}"] += 0.5

    vec: Dict[int, float] = {}
    for feat, count in feats.items():
        h = int.from_bytes(
            hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest(), "big"
        )
        idx = h % EMBED_DIM
        sign = 1.0 if (h >> 32) & 1 else -1.0
        vec[idx] = vec.get(idx, 0.0) + sign * (1.0 + math.log(count))

    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0.0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 if hi > 0 else 0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


# ----------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------
@dataclass
class _Chunk:
    source: str
    text: str
    tokens: List[str]
    token_set: Set[str]
    vector: Dict[int, float]


class _KnowledgeIndex:
    def __init__(self, docs: Sequence[Tuple[str, str]]):
        self.chunks: List[_Chunk] = []
        for source, text in docs:
            for piece in _chunk_text(text, CHUNK_TOKENS):
                tokens = tokenize(piece)
                if not tokens:
                    continue
                self.chunks.append(
                    _Chunk(source, piece, tokens, set(tokens), _embed(tokens))
                )
        self.bm25 = _BM25([c.tokens for c in self.chunks])

    def search(
        self, query: str, alpha: float, max_candidates: int
    ) -> List[Tuple[_Chunk, float]]:
        q_tokens = tokenize(query)[:MAX_QUERY_TOKENS]
        if not q_tokens or not self.chunks:
            return []

        # Hybrid score: alpha * dense + (1 - alpha) * BM25, both min-max normalised.
        sparse = _minmax(self.bm25.score(q_tokens))
        q_vec = _embed(q_tokens)
        dense = _minmax([_cosine(q_vec, c.vector) for c in self.chunks])
        hybrid = [alpha * d + (1.0 - alpha) * s for d, s in zip(dense, sparse)]

        order = sorted(range(len(self.chunks)), key=lambda i: hybrid[i], reverse=True)
        pool = order[: max(max_candidates * 2, 10)]

        # Lightweight rerank: blend in query-term coverage (stand-in for a
        # cross-encoder such as bge-reranker).
        q_terms = {t for t in q_tokens if t in self.bm25.idf}
        reranked: List[Tuple[_Chunk, float]] = []
        for i in pool:
            chunk = self.chunks[i]
            coverage = len(q_terms & chunk.token_set) / len(q_terms) if q_terms else 0.0
            reranked.append((chunk, 0.8 * hybrid[i] + 0.2 * coverage))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:max_candidates]


# ----------------------------------------------------------------------
# Public service
# ----------------------------------------------------------------------
class RAGService:
    _index: Optional[_KnowledgeIndex] = None
    _lock = threading.Lock()

    @staticmethod
    def _load_documents() -> List[Tuple[str, str]]:
        docs_dir = Path(os.getenv("RAG_DOCS_DIR", "knowledge_base"))
        if not docs_dir.is_dir():
            return []

        docs: List[Tuple[str, str]] = []
        for path in sorted(docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Skipping unreadable knowledge file %s: %s", path, exc)
                continue
            if text.strip():
                docs.append((path.relative_to(docs_dir).as_posix(), text))
        return docs

    @classmethod
    def _get_index(cls) -> _KnowledgeIndex:
        if cls._index is None:
            with cls._lock:
                if cls._index is None:
                    docs = cls._load_documents()
                    if not docs:
                        logger.warning(
                            "No documents found in RAG_DOCS_DIR; using built-in defaults."
                        )
                        docs = DEFAULT_KNOWLEDGE_BASE
                    cls._index = _KnowledgeIndex(docs)
        return cls._index

    @classmethod
    def reload_index(cls) -> None:
        """Call after changing the knowledge files; the index rebuilds on next use."""
        with cls._lock:
            cls._index = None

    @classmethod
    def _retrieve_sync(cls, text: str, budget: int, alpha: float) -> str:
        index = cls._get_index()

        # Token scaling: a bigger budget considers more candidate chunks.
        max_candidates = max(3, min(40, budget // 100))
        ranked = [
            (chunk, score)
            for chunk, score in index.search(text, alpha, max_candidates)
            if score > 0
        ]
        if not ranked:  # nothing matched: fall back to the general guidelines
            ranked = [(chunk, 0.0) for chunk in index.chunks]

        header = f"[Hybrid Search | alpha={alpha:.2f} | budget={budget} tokens]"
        remaining = budget - estimate_tokens(header) - 1
        parts: List[str] = []

        # Pack chunks in rank order; truncate the last one to fit the budget.
        for chunk, _score in ranked:
            if remaining < 20:
                break
            line = f"- [{chunk.source}] {chunk.text}"
            cost = estimate_tokens(line) + 1
            if cost <= remaining:
                parts.append(line)
                remaining -= cost
            else:
                clipped = truncate_to_tokens(line, remaining - 2)
                if clipped:
                    parts.append(clipped + "…")
                break

        return header + "\n" + "\n".join(parts)

    @classmethod
    async def retrieve_context(
        cls, submission_text: str, token_budget: int, hybrid_alpha: float
    ) -> str:
        """Hybrid (dense + BM25) retrieval packed into the given token budget."""
        if not submission_text or not submission_text.strip():
            raise ValueError("Submission cannot be empty.")

        budget = max(MIN_BUDGET, min(MAX_BUDGET, int(token_budget)))
        alpha = max(0.0, min(1.0, float(hybrid_alpha)))

        # CPU-bound work runs off the event loop.
        return await asyncio.to_thread(cls._retrieve_sync, submission_text, budget, alpha)