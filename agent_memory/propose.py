"""Propose-test-apply: a guardrail for agents that edit their own project.

A self-improving agent should never write changes straight into its live
project. This module stages proposed file changes in a throwaway copy, runs
the project's test command there, and only copies the changes into the real
project when the tests pass.

Guardrails:
  * paths must stay inside the project directory (no absolute paths, no ..)
  * the number and size of files per proposal are capped
  * the test command runs with a timeout in the staged copy
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict

MAX_FILES_PER_PROPOSAL = 20
MAX_FILE_BYTES = 512 * 1024


class ProposalError(ValueError):
    pass


@dataclass
class Proposal:
    description: str
    # relative path -> full new file content
    files: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.files:
            raise ProposalError("proposal has no files")
        if len(self.files) > MAX_FILES_PER_PROPOSAL:
            raise ProposalError(
                f"too many files ({len(self.files)} > {MAX_FILES_PER_PROPOSAL})")
        for path, content in self.files.items():
            if os.path.isabs(path) or ".." in path.split(os.sep) or ".." in path.split("/"):
                raise ProposalError(f"unsafe path: {path!r}")
            if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                raise ProposalError(f"file too large: {path!r}")


@dataclass
class Result:
    applied: bool
    description: str
    test_output: str = ""
    reason: str = ""


def _stage(project_dir: str, proposal: Proposal, staging: str) -> str:
    """Copy the project into staging and write the proposed files there."""
    staged = os.path.join(staging, "project")
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".venv", "node_modules")
    shutil.copytree(project_dir, staged, ignore=ignore)
    for rel_path, content in proposal.files.items():
        dest = os.path.join(staged, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(content)
    return staged


def apply_if_tests_pass(project_dir: str, proposal: Proposal,
                        test_command: str = "python3 -m unittest discover -s tests",
                        timeout_seconds: int = 300) -> Result:
    """Stage, test, and only then apply a proposal. Never raises for a
    failing test run; the Result explains what happened."""
    try:
        proposal.validate()
    except ProposalError as exc:
        return Result(applied=False, description=proposal.description,
                      reason=f"rejected: {exc}")

    with tempfile.TemporaryDirectory(prefix="agent-proposal-") as staging:
        staged = _stage(project_dir, proposal, staging)
        try:
            proc = subprocess.run(
                test_command, shell=True, cwd=staged,
                capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return Result(applied=False, description=proposal.description,
                          reason=f"tests timed out after {timeout_seconds}s")
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            return Result(applied=False, description=proposal.description,
                          test_output=output,
                          reason=f"tests failed (exit {proc.returncode})")
        for rel_path in proposal.files:
            src = os.path.join(staged, rel_path)
            dest = os.path.join(project_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
    return Result(applied=True, description=proposal.description,
                  test_output=output, reason="tests passed")
