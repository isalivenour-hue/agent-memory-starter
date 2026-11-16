# agent-memory-starter

A small, dependency-free reference implementation of the memory pattern most
LLM agent projects end up needing:

1. **Store** raw episodes as they happen.
2. **Recall** the relevant ones for the current context.
3. **Reflect** periodically to consolidate episodes into durable facts.
4. **Propose-test-apply** when the agent wants to change its own project:
   stage the change, run the tests, only then apply.

Pure Python 3.10+, standard library only. Read it in one sitting, copy the
pieces you need, plug in your own model where it matters.

## Install

```bash
git clone https://github.com/noah-hayes-dev/agent-memory-starter.git
cd agent-memory-starter
python3 examples/demo_agent.py
```

## The pieces

### MemoryStore - append-only memory with useful recall

```python
from agent_memory import MemoryStore

store = MemoryStore("memory.jsonl")
store.remember("User prefers metric units.")
store.remember("User is training for a 10k run on Saturdays.")

for entry in store.recall("running training", k=3):
    print(entry.kind, entry.text)
```

- One JSON object per line (JSONL). Appends only, never rewrites, so a
  long-running agent cannot corrupt its history mid-write.
- Recall scores by **word overlap x recency decay x pinned boost**.
  `store.pin(id)` marks a memory as always relevant (skips recency decay);
  `store.forget(id)` tombstones it.

### reflect - episodes become facts

A log of episodes gets noisy. `reflect` summarizes recent episodes into one
durable `fact` entry. Pass your own model:

```python
from agent_memory import reflect

fact = reflect(store, summarize=lambda eps: my_llm.summarize(eps))
```

Without a `summarize` callable it uses a deterministic extractive fallback,
so the whole package works offline and in tests.

### Proposal - a guardrail for self-modifying agents

Never let an agent write straight into its live project:

```python
from agent_memory import Proposal, apply_if_tests_pass

change = Proposal(
    description="add util.double",
    files={"util.py": "def double(x):\n    return x * 2\n"},
)
result = apply_if_tests_pass("/path/to/project", change)
result.applied  # True only if the staged copy passed the project's tests
```

Changes are written to a throwaway copy of the project, the test command
runs there, and only a green run is copied back. Paths are confined to the
project (no `..`, no absolute paths), and file count and size are capped.

## Why these choices

- **JSONL over a database**: inspectable with `tail`, diffable in git, and
  recoverable line by line. Good enough until you have ~100k entries.
- **Keyword recall over embeddings**: zero dependencies and no API keys.
  Swap `recall` for an embedding index later; the store format does not
  change.
- **Facts as entries**: reflections live in the same store, so recall finds
  both what happened and what the agent concluded from it.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT - see [LICENSE](LICENSE).
