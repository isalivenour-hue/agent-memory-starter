"""Append-only memory store with keyword + recency retrieval.

The store is a JSONL file: one JSON object per line, never rewritten in
place. Entries are immutable once written; "forgetting" is a tombstone and
"pinning" is a flag, both appended as new lines. That keeps the file safe to
append to from a long-running agent without locks or rewrites.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional

WORD_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)

RECENCY_HALFLIFE_DAYS = 30.0
PINNED_BOOST = 2.0


@dataclass
class Entry:
    id: str
    ts: float
    kind: str  # "episode" (what happened) or "fact" (what the agent learned)
    text: str
    pinned: bool = False
    forgotten: bool = False

    def tokens(self) -> set:
        return set(WORD_RE.findall(self.text.lower()))


def _tokens(text: str) -> set:
    return set(WORD_RE.findall(text.lower()))


class MemoryStore:
    """A file-backed memory store for an agent.

    Parameters
    ----------
    path:
        JSONL file to use. Parent directories are created on first write.
    now:
        Optional callable returning the current unix timestamp (testing).
    """

    def __init__(self, path: str, now=None):
        self.path = path
        self._now = now or time.time

    # ---- writes ---------------------------------------------------------

    def remember(self, text: str, kind: str = "episode",
                 pinned: bool = False) -> Entry:
        """Append a new memory and return it."""
        if kind not in ("episode", "fact"):
            raise ValueError("kind must be 'episode' or 'fact'")
        entry = Entry(id=uuid.uuid4().hex[:12], ts=self._now(),
                      kind=kind, text=text, pinned=pinned)
        self._append(entry)
        return entry

    def pin(self, entry_id: str) -> bool:
        entry = self.get(entry_id)
        if entry is None or entry.forgotten:
            return False
        entry.pinned = True
        self._append(entry)
        return True

    def forget(self, entry_id: str) -> bool:
        entry = self.get(entry_id)
        if entry is None or entry.forgotten:
            return False
        entry.forgotten = True
        self._append(entry)
        return True

    def _append(self, entry: Entry) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")

    # ---- reads ----------------------------------------------------------

    def entries(self) -> List[Entry]:
        """Current state: latest version of each entry, forgotten excluded."""
        if not os.path.exists(self.path):
            return []
        latest = {}
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                latest[data["id"]] = Entry(**data)
        return [e for e in latest.values() if not e.forgotten]

    def get(self, entry_id: str) -> Optional[Entry]:
        for e in self.entries():
            if e.id == entry_id:
                return e
        return None

    def recall(self, query: str, k: int = 5,
               kinds: Optional[Iterable[str]] = None) -> List[Entry]:
        """Return the k most relevant entries for a query.

        Score = word-overlap similarity x recency decay x pinned boost.
        A pinned entry skips recency decay: pinning means "always relevant".
        Entries sharing no words with the query are not returned.
        """
        q_tokens = _tokens(query)
        if not q_tokens:
            return []
        allowed = set(kinds) if kinds else None
        now = self._now()
        scored = []
        for e in self.entries():
            if allowed and e.kind not in allowed:
                continue
            overlap = len(q_tokens & e.tokens())
            if overlap == 0:
                continue
            similarity = overlap / math.sqrt(len(q_tokens) * max(1, len(e.tokens())))
            age_days = max(0.0, (now - e.ts) / 86400.0)
            recency = 1.0 if e.pinned else 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
            boost = PINNED_BOOST if e.pinned else 1.0
            scored.append((similarity * recency * boost, e))
        scored.sort(key=lambda item: (item[0], item[1].ts), reverse=True)
        return [e for _, e in scored[:k]]
