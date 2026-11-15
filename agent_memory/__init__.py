"""agent-memory-starter: a small, honest reference for agent memory.

Three pieces:
  MemoryStore  - append-only JSONL memory with keyword + recency recall
  reflect      - consolidate episodes into durable facts (LLM optional)
  Proposal     - propose-test-apply guardrail for self-modifying agents
"""
from .store import MemoryStore, Entry
from .reflect import reflect, default_summarize
from .propose import Proposal, Result, ProposalError, apply_if_tests_pass

__all__ = [
    "MemoryStore",
    "Entry",
    "reflect",
    "default_summarize",
    "Proposal",
    "Result",
    "ProposalError",
    "apply_if_tests_pass",
]

__version__ = "0.1.0"
