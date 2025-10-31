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
    """
    Add a fact to knowledge_ru.txt with simple topic detection.

    Rules used:
    - If fact starts with `[topic]` -> create a new topic section with that name.
    - Else, if any existing topic name appears in the fact (case-insensitive substring) -> append the fact under that topic.
    - Else, if fact contains a short "Topic: description" (left of first ':' is short) -> create a new topic with that name.
    - Otherwise append the fact to the `[Разное]` section (create it if missing).

    After update reload knowledge and (optionally) rebuild embeddings.
    """
    s = fact_text.strip()
    raw = KB_FILE.read_text(encoding="utf-8")

    # Parse existing sections: split on lines with only // as separator
    parts = [p.strip() for p in raw.split("//") if p.strip()]
    sections = []  # list of (header, list_of_lines)
    for p in parts:
        lines = [l.rstrip() for l in p.splitlines() if l.strip()]
        if not lines:
            continue
        # header if first line like [topic]
        first = lines[0]
        if first.startswith("[") and "]" in first:
            header = first
            body = lines[1:]
        else:
            header = None
            body = lines
        sections.append((header, body))

    # collect topic names
    topics = []
    for hdr, _ in sections:
        if hdr and hdr.startswith("[") and hdr.endswith("]"):
            topics.append(hdr[1:-1])

    lower_s = s.lower()

    # 1) explicit [topic] at start
    import re
    m = re.match(r"^\s*\[(.+?)\]\s*(.*)$", s, flags=re.DOTALL)
    if m:
        topic = m.group(1).strip()
        content = m.group(2).strip()
        # create new section with header [topic]
        new_lines = []
        if content:
            for line in content.splitlines():
                if line.strip():
                    new_lines.append("-" + line.strip())
        sections.append((f"[{topic}]", new_lines))
        # write back
        out = []
        for hdr, body in sections:
            if hdr:
                out.append(hdr)
            out.extend(body)
            out.append("//")
        KB_FILE.write_text("\n".join(out).strip()+"\n", encoding="utf-8")
        load_knowledge(rebuild_embeddings=True)
        return

    # 2) match existing topic names inside text
    matched_topic = None
    for t in topics:
        if t.lower() in lower_s:
            matched_topic = t
            break

    if matched_topic:
        # append as a '-fact' to that topic
        new_sections = []
        for hdr, body in sections:
            if hdr and hdr.startswith("[") and hdr.endswith("]") and hdr[1:-1] == matched_topic:
                body = body + ["-" + s]
            new_sections.append((hdr, body))
        # write
        out = []
        for hdr, body in new_sections:
            if hdr:
                out.append(hdr)
            out.extend(body)
            out.append("//")
        KB_FILE.write_text("\n".join(out).strip()+"\n", encoding="utf-8")
        load_knowledge(rebuild_embeddings=True)
        return

    # 3) check for 'Topic: description' pattern
    if ":" in s:
        left, right = s.split(":", 1)
        if 1 <= len(left.strip()) <= 60 and len(right.strip()) > 0 and " " not in left.strip()[:2]:
            # treat left as topic name
            topic = left.strip()
            content = right.strip()
            new_lines = ["-" + content]
            sections.append((f"[{topic}]", new_lines))
            out = []
            for hdr, body in sections:
                if hdr:
                    out.append(hdr)
                out.extend(body)
                out.append("//")
            KB_FILE.write_text("\n".join(out).strip()+"\n", encoding="utf-8")
            load_knowledge(rebuild_embeddings=True)
            return

    # 4) otherwise add to Разное
    found = False
    new_sections = []
    for hdr, body in sections:
        if hdr and hdr.startswith("[") and hdr.endswith("]") and hdr[1:-1].lower() == "разное":
            body = body + ["-" + s]
            found = True
        new_sections.append((hdr, body))
    if not found:
        new_sections.append(("[Разное]", ["-" + s]))

    out = []
    for hdr, body in new_sections:
        if hdr:
            out.append(hdr)
        out.extend(body)
        out.append("//")
    KB_FILE.write_text("\n".join(out).strip()+"\n", encoding="utf-8")
    load_knowledge(rebuild_embeddings=True)

try:
    load_knowledge(rebuild_embeddings=False)
except Exception:
    pass

