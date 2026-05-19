"""
HuggingFace dataset loaders for coding problems.

All loaders return problems in the standard format:
    {
        'id':          str,
        'description': str,
        'tests':       [{'input': str, 'expected': str}],
        'difficulty':  float,
        'solution':    str | None,
    }

Install: pip install coding-env[problems]   (pulls datasets library)

Datasets used:
  HumanEval     164 problems  openai/openai_humaneval
  MBPP          500 problems  google-research-datasets/mbpp
  CodeContests  13K problems  deepmind/code_contests   (stdin/stdout format)
  APPS          10K problems  codeparrot/apps           (mixed difficulty)
"""
from __future__ import annotations

import re
import textwrap
from typing import Iterator


def _require_datasets():
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "datasets required: pip install coding-env[problems]\n"
            "or: pip install datasets"
        )


class HumanEvalSource:
    """
    OpenAI HumanEval: 164 Python function completion problems.
    Difficulty: easy-medium. Problems have typed function signatures + docstrings.
    Test format: assert statements in 'test' field.

    Average difficulty mapping: ~0.3–0.5 (roughly 60% of models solve each).
    """

    DATASET = "openai/openai_humaneval"

    def __init__(self, split: str = 'test', cache_dir: str | None = None):
        self.split     = split
        self.cache_dir = cache_dir

    def load(self, limit: int | None = None) -> list[dict]:
        """Download and convert HumanEval to standard problem format."""
        ds_lib = _require_datasets()
        ds     = ds_lib.load_dataset(self.DATASET, split=self.split,
                                      cache_dir=self.cache_dir)
        problems = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            prob = self._convert(row)
            if prob is not None:
                problems.append(prob)
        return problems

    def _convert(self, row: dict) -> dict | None:
        """Convert HumanEval row to standard format."""
        try:
            task_id   = row['task_id']
            prompt    = row['prompt']         # function signature + docstring
            canonical = row['canonical_solution']
            test_code = row['test']           # assert-based test function
            entry     = row['entry_point']

            # Build test cases by extracting assert statements
            tests = _parse_humaneval_tests(test_code, entry, prompt + canonical)
            if not tests:
                return None

            description = (
                f"Complete the following Python function:\n\n```python\n{prompt}\n```"
            )
            return {
                'id':          task_id.replace('/', '_'),
                'description': description,
                'tests':       tests,
                'difficulty':  0.4,  # HumanEval is roughly medium difficulty
                'solution':    prompt + canonical,
            }
        except Exception:
            return None

    def stream(self) -> Iterator[dict]:
        """Iterate without loading all into memory."""
        ds_lib = _require_datasets()
        ds     = ds_lib.load_dataset(self.DATASET, split=self.split,
                                      streaming=True)
        for row in ds:
            prob = self._convert(row)
            if prob is not None:
                yield prob


class MBPPSource:
    """
    Google MBPP: ~500 Python programming problems, beginner-friendly.
    Difficulty: 0.2–0.4. Problems have short descriptions + test assertions.
    Good for the 20-60% difficulty zone.
    """

    DATASET = "google-research-datasets/mbpp"

    def __init__(self, split: str = 'train', sanitized: bool = True,
                 cache_dir: str | None = None):
        self.split      = split
        self.sanitized  = sanitized  # use sanitized version (cleaner problems)
        self.cache_dir  = cache_dir

    def load(self, limit: int | None = None) -> list[dict]:
        ds_lib = _require_datasets()
        config = 'sanitized' if self.sanitized else 'full'
        ds     = ds_lib.load_dataset(self.DATASET, config, split=self.split,
                                      cache_dir=self.cache_dir)
        problems = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            prob = self._convert(row)
            if prob is not None:
                problems.append(prob)
        return problems

    def _convert(self, row: dict) -> dict | None:
        try:
            task_id     = str(row.get('task_id', ''))
            description = row['text']
            code        = row['code']
            test_list   = row.get('test_list', []) or row.get('test_imports', [])

            tests = _parse_mbpp_tests(test_list, code)
            if not tests:
                return None

            return {
                'id':          f'mbpp_{task_id}',
                'description': description,
                'tests':       tests,
                'difficulty':  0.3,
                'solution':    code,
            }
        except Exception:
            return None

    def stream(self) -> Iterator[dict]:
        ds_lib  = _require_datasets()
        config  = 'sanitized' if self.sanitized else 'full'
        ds      = ds_lib.load_dataset(self.DATASET, config, split=self.split,
                                       streaming=True)
        for row in ds:
            prob = self._convert(row)
            if prob is not None:
                yield prob


class CodeContestsSource:
    """
    DeepMind CodeContests: 13,000+ competitive programming problems.
    Difficulty: 0.4–0.9. Problems have stdin/stdout format — perfect for our executor.
    Source: Codeforces, CodeChef, AtCoder.
    """

    DATASET = "deepmind/code_contests"

    def __init__(self, split: str = 'train', max_difficulty: float = 0.7,
                 cache_dir: str | None = None):
        self.split          = split
        self.max_difficulty = max_difficulty
        self.cache_dir      = cache_dir

    def load(self, limit: int | None = None) -> list[dict]:
        ds_lib = _require_datasets()
        ds     = ds_lib.load_dataset(self.DATASET, split=self.split,
                                      cache_dir=self.cache_dir)
        problems = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            prob = self._convert(row)
            if prob is not None and prob['difficulty'] <= self.max_difficulty:
                problems.append(prob)
        return problems

    def _convert(self, row: dict) -> dict | None:
        try:
            name   = row.get('name', '')
            desc   = row.get('description', '')
            # Difficulty: CF ratings roughly 800-3500 → normalize to 0-1
            rating = row.get('cf_rating', 1200) or 1200
            diff   = min(1.0, (rating - 800) / 2500)

            # Use public test inputs/outputs
            pub_inputs  = row.get('public_tests', {}).get('input', [])
            pub_outputs = row.get('public_tests', {}).get('output', [])

            if not pub_inputs:
                return None

            tests = [
                {'input': inp.strip(), 'expected': out.strip()}
                for inp, out in zip(pub_inputs[:5], pub_outputs[:5])
                if inp.strip() and out.strip()
            ]
            if not tests:
                return None

            return {
                'id':          f'cc_{name}',
                'description': desc,
                'tests':       tests,
                'difficulty':  diff,
                'solution':    None,
            }
        except Exception:
            return None

    def stream(self) -> Iterator[dict]:
        ds_lib = _require_datasets()
        ds     = ds_lib.load_dataset(self.DATASET, split=self.split,
                                      streaming=True)
        for row in ds:
            prob = self._convert(row)
            if prob is not None and prob['difficulty'] <= self.max_difficulty:
                yield prob


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_humaneval_tests(test_code: str, entry_point: str,
                            full_code: str) -> list[dict]:
    """
    Convert HumanEval assert-based tests to stdin/stdout format.
    Wraps each assert in a runnable script that prints the result.
    """
    # Extract assert lines
    assert_lines = [l.strip() for l in test_code.splitlines()
                    if l.strip().startswith('assert')]
    if not assert_lines:
        return []

    tests = []
    for i, assert_line in enumerate(assert_lines[:5]):
        # Convert: assert func(input) == expected
        # To: script that runs func(input) and prints result
        m = re.match(
            rf'assert\s+{re.escape(entry_point)}\s*\((.+)\)\s*==\s*(.+)',
            assert_line)
        if not m:
            # Try: assert func(input) (boolean result)
            m2 = re.match(rf'assert\s+{re.escape(entry_point)}\s*\((.+)\)', assert_line)
            if m2:
                args    = m2.group(1).strip()
                script  = full_code + f"\nprint({entry_point}({args}))"
                tests.append({'input': '', 'expected': 'True', '_script': script})
            continue

        args     = m.group(1).strip()
        expected = m.group(2).strip()
        # Build runnable script
        script = full_code + f"\nresult = {entry_point}({args})\nprint(result)"
        tests.append({
            'input':    '',
            'expected': expected,
            '_script':  script,   # executor uses this directly
        })

    return tests[:5]


def _parse_mbpp_tests(test_list: list[str], solution: str) -> list[dict]:
    """Convert MBPP assertion tests to stdin/stdout format."""
    tests = []
    for assert_str in test_list[:5]:
        if not assert_str.strip().startswith('assert'):
            continue
        # Wrap solution + assert → run and capture result
        script = (solution + "\n" +
                  assert_str.replace('assert', 'print(bool(') + '))')
        tests.append({'input': '', 'expected': 'True', '_script': script})
    return tests
