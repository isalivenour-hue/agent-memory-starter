"""Reflection: consolidate raw episodes into durable facts.

An agent that only appends episodes ends up with a noisy log. Reflection
periodically reads recent episodes and writes a smaller number of "fact"
entries that future recall can find faster.

`reflect` takes an optional `summarize` callable so you can plug in a real
language model::

    reflect(store, summarize=lambda episodes: my_llm.summarize(episodes))

Without one, it falls back to a deterministic extractive summary (most
frequent meaningful terms), which keeps the whole package dependency-free
and testable offline.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import Callable, List, Optional

from .store import MemoryStore, Entry

WORD_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)

STOPWORDS = frozenset("""
a an and are as at be by for from has have i in is it its of on or that the
this to was were will with we you he she they them their our my your me him
her us not but so if then than too very can just about into over after
""".split())


def default_summarize(episodes: List[Entry]) -> str:
    """Extractive fallback: the most frequent meaningful terms."""
    counter = Counter()
    for e in episodes:
        for token in WORD_RE.findall(e.text.lower()):
            if len(token) >= 4 and token not in STOPWORDS:
                counter[token] += 1
    top = [w for w, _ in counter.most_common(8)]
    if not top:
        return "Nothing notable in recent episodes."
    return "Recurring themes: " + ", ".join(top) + "."


def reflect(store: MemoryStore,
            summarize: Optional[Callable[[List[Entry]], str]] = None,
            since_hours: float = 24.0,
            min_episodes: int = 3,
            now: Optional[float] = None) -> Optional[Entry]:
    """Summarize recent episodes into one durable fact.

    Returns the new fact Entry, or None when there is too little new
    material to be worth consolidating.
    """
    now = time.time() if now is None else now
    cutoff = now - since_hours * 3600.0
    recent = [e for e in store.entries()
              if e.kind == "episode" and e.ts >= cutoff]
    if len(recent) < min_episodes:
        return None
    summarize = summarize or default_summarize
    text = summarize(sorted(recent, key=lambda e: e.ts))
    if not text or not text.strip():
        return None
    return store.remember(text.strip(), kind="fact")
