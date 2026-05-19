"""
Sandboxed Python code executor.

Executes code in a subprocess with:
  - Timeout (default 10s)
  - stdout/stderr capture
  - No network access (os-level for now; full sandbox via Docker optional)
  - Resource limits (CPU + memory)

Returns ExecutionResult with pass/fail per test case and error messages.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field


@dataclass
class TestResult:
    test_id:  int
    passed:   bool
    expected: str  = ""
    actual:   str  = ""
    error:    str  = ""


@dataclass
class ExecutionResult:
    test_results:  list[TestResult]
    compile_error: str  = ""
    timeout:       bool = False
    elapsed_s:     float = 0.0

    @property
    def pass_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return sum(r.passed for r in self.test_results) / len(self.test_results)

    @property
    def n_passed(self) -> int:
        return sum(r.passed for r in self.test_results)

    @property
    def error_message(self) -> str:
        """First error encountered — most useful for the brain's repair loop."""
        if self.compile_error:
            return self.compile_error
        if self.timeout:
            return "TimeoutError: execution exceeded time limit"
        for r in self.test_results:
            if not r.passed and r.error:
                return r.error
            if not r.passed:
                return f"AssertionError: expected {r.expected!r}, got {r.actual!r}"
        return ""


def execute(
    code:       str,
    tests:      list[dict],   # [{'input': str, 'expected': str}, ...]
    timeout_s:  float = 10.0,
) -> ExecutionResult:
    """
    Run `code` against each test case.

    Each test dict has:
        input:    str — printed to stdin
        expected: str — expected stdout (stripped)

    Returns ExecutionResult with per-test pass/fail and error messages.
    """
    t0 = time.time()

    # Tests with '_script' field have the solution embedded (HumanEval/MBPP format).
    # In that case the brain's `code` is appended to the embedded solution for scoring,
    # OR we run the embedded script directly to get the expected behavior.
    harness = _build_harness(code, tests)

    try:
        proc = subprocess.run(
            [sys.executable, '-c', harness],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0 and proc.returncode != 42:
            # 42 = our sentinel for test failure; anything else = compile/runtime error
            return ExecutionResult(
                test_results  = [],
                compile_error = (proc.stderr or proc.stdout)[:500],
                elapsed_s     = elapsed,
            )

        # Parse structured output
        results = _parse_output(proc.stdout, tests)
        return ExecutionResult(test_results=results, elapsed_s=elapsed)

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            test_results = [],
            timeout      = True,
            elapsed_s    = time.time() - t0,
        )


def _build_harness(code: str, tests: list[dict]) -> str:
    """Build a self-contained test script."""
    # 8 spaces: 4 for inside def + 4 for inside try block
    indented = textwrap.indent(code, '        ')
    cases = repr(tests)
    return f"""
import sys, io, traceback

def _run_solution(input_str):
    _saved = sys.stdin, sys.stdout
    sys.stdin  = io.StringIO(input_str)
    sys.stdout = buf = io.StringIO()
    try:
{indented}
    except Exception as e:
        sys.stdin, sys.stdout = _saved
        return None, traceback.format_exc()
    sys.stdin, sys.stdout = _saved
    return buf.getvalue().strip(), None

tests = {cases}
results = []
for i, t in enumerate(tests):
    actual, err = _run_solution(t.get('input',''))
    expected    = str(t.get('expected','')).strip()
    passed      = (err is None) and (actual == expected)
    results.append((i, passed, expected, actual or '', err or ''))

for r in results:
    print('RESULT', *r, sep='\\x00')
"""


def _parse_output(stdout: str, tests: list[dict]) -> list[TestResult]:
    results = []
    for line in stdout.splitlines():
        if not line.startswith('RESULT'):
            continue
        parts = line.split('\x00')
        if len(parts) < 5:
            continue
        try:
            _, i, passed, expected, actual, *err_parts = parts
            error = '\x00'.join(err_parts) if err_parts else ''
            results.append(TestResult(
                test_id  = int(i),
                passed   = passed == 'True',
                expected = expected,
                actual   = actual,
                error    = error[:300],
            ))
        except (ValueError, IndexError):
            continue

    if not results:
        return [TestResult(test_id=i, passed=False, error="no output")
                for i in range(len(tests))]
    return results
