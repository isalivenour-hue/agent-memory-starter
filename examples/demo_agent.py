"""A tiny agent with memory, runnable offline.

    python3 examples/demo_agent.py

It records a few episodes, recalls relevant memories for a query, reflects
them into a durable fact, and shows the propose-test-apply guardrail on a
scratch project.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_memory import MemoryStore, reflect, Proposal, apply_if_tests_pass


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(os.path.join(tmp, "memory.jsonl"))

        print("== recording episodes ==")
        store.remember("User asked for a daily weather summary at 8am.")
        store.remember("User prefers metric units.")
        store.remember("User is learning Spanish and likes short quizzes.")
        store.remember("User asked about weekend train times twice this week.")

        print("\n== recall: 'weather summary preferences' ==")
        for e in store.recall("weather summary preferences", k=3):
            print(f"  [{e.kind}] {e.text}")

        print("\n== reflect (min_episodes=3) ==")
        fact = reflect(store, min_episodes=3)
        print(f"  new fact: {fact.text if fact else 'none'}")

        print("\n== propose-test-apply on a scratch project ==")
        project = os.path.join(tmp, "project")
        os.makedirs(os.path.join(project, "tests"))
        with open(os.path.join(project, "tests", "test_ok.py"), "w") as fh:
            fh.write("import unittest\n\n"
                     "from util import double\n\n"
                     "class T(unittest.TestCase):\n"
                     "    def test_double(self):\n"
                     "        self.assertEqual(double(2), 4)\n\n"
                     "if __name__ == '__main__':\n"
                     "    unittest.main()\n")

        good = Proposal(
            description="add util.double",
            files={"util.py": "def double(x):\n    return x * 2\n"})
        result = apply_if_tests_pass(project, good)
        print(f"  good proposal applied: {result.applied} ({result.reason})")

        bad = Proposal(
            description="break util.double",
            files={"util.py": "def double(x):\n    return x * 3\n"})
        result = apply_if_tests_pass(project, bad)
        print(f"  bad proposal applied: {result.applied} ({result.reason})")

        with open(os.path.join(project, "util.py")) as fh:
            print(f"  live project still has: {fh.read().strip()}")


if __name__ == "__main__":
    main()
