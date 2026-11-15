import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_memory import (
    MemoryStore,
    Proposal,
    ProposalError,
    apply_if_tests_pass,
    default_summarize,
    reflect,
)


class Clock:
    def __init__(self, start=1_700_000_000.0):
        self.t = start

    def __call__(self):
        return self.t


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.store = MemoryStore(
            os.path.join(self.tmp.name, "mem.jsonl"), now=self.clock)

    def tearDown(self):
        self.tmp.cleanup()

    def test_remember_and_entries(self):
        self.store.remember("hello world")
        self.store.remember("a learned fact", kind="fact")
        entries = self.store.entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual({e.kind for e in entries}, {"episode", "fact"})

    def test_kind_validation(self):
        with self.assertRaises(ValueError):
            self.store.remember("x", kind="weird")

    def test_recall_ranks_by_overlap(self):
        self.store.remember("the user likes python examples")
        self.store.remember("weather was rainy today")
        hits = self.store.recall("python examples")
        self.assertEqual(len(hits), 1)
        self.assertIn("python", hits[0].text)

    def test_recall_recency_and_pin(self):
        old = self.store.remember("user likes tea")
        self.clock.t += 200 * 86400  # 200 days later
        self.store.remember("user likes tea and coffee")
        hits = self.store.recall("tea")
        self.assertIn("coffee", hits[0].text)
        # pinning the old entry pulls it back to the top
        self.store.pin(old.id)
        hits = self.store.recall("tea")
        self.assertEqual(hits[0].text, "user likes tea")

    def test_forget_excludes(self):
        e = self.store.remember("temporary thing about boats")
        self.assertTrue(self.store.forget(e.id))
        self.assertEqual(self.store.recall("boats"), [])
        self.assertFalse(self.store.forget(e.id))
        self.assertFalse(self.store.pin(e.id))

    def test_recall_empty_query_and_no_overlap(self):
        self.store.remember("something")
        self.assertEqual(self.store.recall(""), [])
        self.assertEqual(self.store.recall("zzzzqqqq"), [])

    def test_missing_file_is_empty(self):
        fresh = MemoryStore(os.path.join(self.tmp.name, "nope.jsonl"))
        self.assertEqual(fresh.entries(), [])


class TestReflect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.store = MemoryStore(
            os.path.join(self.tmp.name, "mem.jsonl"), now=self.clock)

    def tearDown(self):
        self.tmp.cleanup()

    def test_too_few_episodes(self):
        self.store.remember("only one episode")
        self.assertIsNone(reflect(self.store, now=self.clock()))

    def test_default_summarize(self):
        for i in range(4):
            self.store.remember(f"user talked about robotics and python again {i}")
        fact = reflect(self.store, now=self.clock())
        self.assertIsNotNone(fact)
        self.assertEqual(fact.kind, "fact")
        self.assertIn("robotics", fact.text)

    def test_old_episodes_ignored(self):
        for i in range(3):
            self.store.remember(f"ancient history {i}")
        self.clock.t += 10 * 86400
        self.assertIsNone(reflect(self.store, now=self.clock()))

    def test_custom_summarize(self):
        for i in range(3):
            self.store.remember(f"episode {i}")
        fact = reflect(self.store, summarize=lambda eps: f"saw {len(eps)} episodes",
                       now=self.clock())
        self.assertEqual(fact.text, "saw 3 episodes")

    def test_default_summarize_empty(self):
        self.assertEqual(default_summarize([]), "Nothing notable in recent episodes.")


class TestPropose(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.project = os.path.join(self.tmp.name, "project")
        os.makedirs(os.path.join(self.project, "tests"))
        with open(os.path.join(self.project, "tests", "test_util.py"), "w") as fh:
            fh.write("import unittest\n\n"
                     "from util import double\n\n"
                     "class T(unittest.TestCase):\n"
                     "    def test_double(self):\n"
                     "        self.assertEqual(double(3), 6)\n\n"
                     "if __name__ == '__main__':\n"
                     "    unittest.main()\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_good_proposal_applies(self):
        p = Proposal("add double", {"util.py": "def double(x):\n    return x*2\n"})
        r = apply_if_tests_pass(self.project, p)
        self.assertTrue(r.applied, r.reason)
        with open(os.path.join(self.project, "util.py")) as fh:
            self.assertIn("x*2", fh.read())

    def test_failing_tests_block_apply(self):
        p = Proposal("break double", {"util.py": "def double(x):\n    return x*5\n"})
        r = apply_if_tests_pass(self.project, p)
        self.assertFalse(r.applied)
        self.assertIn("tests failed", r.reason)
        self.assertFalse(os.path.exists(os.path.join(self.project, "util.py")))

    def test_unsafe_paths_rejected(self):
        for bad in ("../escape.py", "/abs/path.py", "a/../../b.py"):
            p = Proposal("escape", {bad: "x = 1\n"})
            with self.assertRaises(ProposalError):
                p.validate()

    def test_empty_proposal_rejected(self):
        r = apply_if_tests_pass(self.project, Proposal("nothing", {}))
        self.assertFalse(r.applied)
        self.assertIn("no files", r.reason)


if __name__ == "__main__":
    unittest.main()
