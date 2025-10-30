import os
from pathlib import Path
from typing import List, Tuple
import numpy as np

SentenceTransformer = None
_HAS_SENTENCE_TRANSFORMERS = False



KB_FILE = Path("knowledge_ru.txt")
if not KB_FILE.exists():
    fallback = "Git — распределённая система контроля версий... "
    KB_FILE.write_text(fallback, encoding="utf-8")


def chunk_text(txt: str, maxlen: int = 1500) -> List[str]:
    """Split text into chunks roughly by paragraph, each chunk <= maxlen."""
    parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 2 <= maxlen:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks



_texts: List[str] = []
_doc_emb: np.ndarray = None
_model_name = "intfloat/multilingual-e5-base"
_embed_model = None


def _load_model():
    global _embed_model
    global SentenceTransformer, _HAS_SENTENCE_TRANSFORMERS
    if _embed_model is not None:
        return _embed_model
    if not _HAS_SENTENCE_TRANSFORMERS:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST
            _HAS_SENTENCE_TRANSFORMERS = True
        except Exception:
            _HAS_SENTENCE_TRANSFORMERS = False
            return None
    _embed_model = SentenceTransformer(_model_name)
    return _embed_model


def load_knowledge(rebuild_embeddings: bool = True):
    """Load `knowledge_ru.txt` into memory and optionally build embeddings."""
    global _texts, _doc_emb
    raw = KB_FILE.read_text(encoding="utf-8")
    norm = raw.replace("//", "\n\n")
    _texts = chunk_text(norm, maxlen=1500)
    if rebuild_embeddings:
        build_embeddings()


def build_embeddings():
    """Build embeddings for currently loaded `_texts` using SentenceTransformer."""
    global _doc_emb
    model = _load_model()
    if not _HAS_SENTENCE_TRANSFORMERS or model is None:
        _doc_emb = None
        return
    if not _texts:
        _doc_emb = np.zeros((0, model.get_sentence_embedding_dimension()))
        return
    doc_emb = model.encode([f"passage: {t}" for t in _texts], normalize_embeddings=True, convert_to_numpy=True)
    _doc_emb = np.asarray(doc_emb)


def retrieve(query: str, k: int = 3) -> List[Tuple[str, float]]:
    if _HAS_SENTENCE_TRANSFORMERS and _doc_emb is not None and _doc_emb.size != 0:
        model = _load_model()
        qv = model.encode([f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = _doc_emb @ qv
        idx = np.argsort(-scores)[:k]
        return [( _texts[i], float(scores[i]) ) for i in idx]

    import difflib
    q = query.lower()
    scores = []
    for t in _texts:
        s = difflib.SequenceMatcher(a=q, b=t.lower()).ratio()
        scores.append(s)
    idx = np.argsort(-np.array(scores))[:k]
    return [( _texts[i], float(scores[i]) ) for i in idx]


def add_fact(fact_text: str) -> None:
    with KB_FILE.open("a", encoding="utf-8") as f:
        f.write("\n//\n")
        f.write(fact_text.strip() + "\n")
    
    load_knowledge(rebuild_embeddings=True)

try:
    load_knowledge(rebuild_embeddings=False)
except Exception:
    pass

