"""
CodingEnvironment: multi-turn code execution as an ecoframe EnvironmentProtocol.

RLEF-style (RL from Execution Feedback, ICML 2025):
  Episode structure (max K=3 turns per problem):
    Turn 1: brain sees problem description as text tokens
    Brain generates code (ActionBundle.text_tokens)
    Execute → SensorBundle with pass_rate as reward + error as next context
    Turn 2: brain sees problem + code + error → revises
    ...
    Final: reward = pass_rate on held-out tests

Key insight from RLEF paper: the multi-turn structure is what enables learning
to USE execution feedback. Our persistent SSM carries context across turns
without reset — exactly the architecture this requires.

Optimal task distribution (RLEF + TAROT findings):
  Train on problems where current pass rate is 20-60%.
  Too easy → no gradient (entropy collapse).
  Too hard → no signal. The 'zone of proximal development' maximizes RL gradient.

SensorManifest:
  text_tokens:  (MAX_TOKENS,) — problem + prev code + error, tokenized
                action_affected=True: different code → different execution result
                world_external=True:  error message comes from the world (executor)
  pass_rate:    (1,) — fraction of tests passed (0.0–1.0)
                action_affected=True, world_external=True — prediction target
  turn:         (1,) — current turn number / max_turns
                action_affected=False, world_external=False — episode progress

ActionBundle:
  text_tokens: (MAX_CODE_TOKENS,) int32 — brain's generated code tokens
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ecoframe.protocol import (
    ActionBundle, CapacityError, HardwareSpec,
    SensorBundle, SensorManifest, SensorSpec, Session,
)
from ecoframe.signal import CertSignal, EnvironmentSignal


# ── Manifest ──────────────────────────────────────────────────────────────────

MAX_TOKENS = 2048  # context window for problem + code + error

CODING_MANIFEST = SensorManifest(
    env_id="coding",
    sensors=(
        SensorSpec(
            name            = "text_tokens",
            shape           = (MAX_TOKENS,),
            dtype           = "int32",
            action_affected = True,   # different code → different execution result
            world_external  = True,   # executor is external to brain
            temporal_res    = 1.0,
        ),
        SensorSpec(
            name            = "pass_rate",
            shape           = (1,),
            dtype           = "float32",
            action_affected = True,
            world_external  = True,
            temporal_res    = 1.0,
        ),
        SensorSpec(
            name            = "turn",
            shape           = (1,),
            dtype           = "float32",
            action_affected = False,
            world_external  = False,
            temporal_res    = 1.0,
        ),
    ),
)

HARDWARE = HardwareSpec.cpu(n_workers=1, accelerator="subprocess_executor")


# ── CodingEnvironment ─────────────────────────────────────────────────────────

class CodingEnvironment:
    """
    Multi-turn code generation environment.

    Each episode: one problem, up to max_turns attempts.
    The brain's SSM accumulates context across turns naturally —
    no reset between turns within one problem.

    Certification: CertSignal(cert_name="coding_basics") published when
    brain achieves ≥80% pass rate on 10 consecutive eval problems.
    """

    env_id        = "coding"
    manifest      = CODING_MANIFEST
    hardware_spec = HARDWARE
    capacity      = 1   # one active session (coding is single-brain)

    def __init__(
        self,
        problems:      list[dict]    | None = None,
        eval_problems: list[dict]    | None = None,
        max_turns:     int           = 3,
        tokenizer                    = None,  # optional; uses simple char-level fallback
        field                        = None,
        cert_threshold: float        = 0.8,
        cert_window:    int          = 10,    # eval problems for cert
        verbose:        bool         = False,
    ):
        from coding_env.problems import training_problems, eval_problems as _eval_p
        self._problems      = problems      or training_problems()
        self._eval_problems = eval_problems or _eval_p()
        self._max_turns     = max_turns
        self._tokenizer     = tokenizer
        self._field         = field
        self._cert_threshold = cert_threshold
        self._cert_window    = cert_window
        self._verbose        = verbose

        self._sessions:    dict[str, Session] = {}
        self._step_count   = 0

        # Episode state (reset per problem)
        self._current_problem: dict | None = None
        self._current_turn:    int  = 0
        self._current_code:    str  = ""
        self._last_error:      str  = ""
        self._episode_pass_rates: list[float] = []

        # Pending actions from step_async
        self._pending_actions: dict[str, ActionBundle] = {}

        # Eval tracking for certification
        self._eval_results: list[float] = []

        # Problem stream
        from coding_env.problems import stream_problems
        self._problem_iter: Iterator[dict] = stream_problems(self._problems)

    # ── EnvironmentProtocol ────────────────────────────────────────────────────

    def start(self) -> None:
        self._next_problem()
        if self._verbose:
            print(f"CodingEnvironment: loaded problem '{self._current_problem['id']}'",
                  flush=True)

    def close(self) -> None:
        pass

    def enter(self, brain_id: str, ssm_state: dict | None = None) -> Session:
        if len(self._sessions) >= self.capacity:
            raise CapacityError(f"{self.env_id}: capacity {self.capacity} reached")
        session = Session(
            brain_id   = brain_id,
            env_id     = self.env_id,
            agent_id   = brain_id,
            ssm_state  = ssm_state or {},
            entered_at = self._step_count,
        )
        self._sessions[brain_id] = session
        return session

    def exit(self, session: Session) -> dict:
        self._sessions.pop(session.brain_id, None)
        return session.ssm_state

    def reset(self, session: Session) -> dict[str, SensorBundle]:
        if self._current_problem is None:
            self.start()
        self._current_turn = 0
        self._current_code = ""
        self._last_error   = ""
        return {session.brain_id: self._make_bundle(session.brain_id, pass_rate=0.0)}

    def step_async(self, actions: dict[str, ActionBundle]) -> None:
        """Receive code from brain (non-blocking). Execute in step_wait."""
        self._pending_actions = actions

    def step_wait(self) -> dict[str, SensorBundle]:
        """Execute the pending code, advance episode state."""
        self._step_count += 1
        bundles: dict[str, SensorBundle] = {}

        for brain_id, session in self._sessions.items():
            action = self._pending_actions.get(brain_id)
            if action is None:
                bundles[brain_id] = self._make_bundle(brain_id, pass_rate=0.0)
                continue

            # Decode code from action
            code = self._decode_code(action)
            self._current_code = code
            self._current_turn += 1

            # Execute against test cases
            from coding_env.executor import execute
            exec_result = execute(
                code    = code,
                tests   = self._current_problem['tests'],
                timeout_s = 10.0,
            )
            pass_rate = exec_result.pass_rate
            self._last_error = exec_result.error_message

            if self._verbose:
                print(f"CodingEnv turn={self._current_turn} "
                      f"pass={exec_result.n_passed}/{len(exec_result.test_results)} "
                      f"({pass_rate:.2f})", flush=True)

            # Check episode end: all tests pass OR max turns reached
            done = (pass_rate >= 1.0) or (self._current_turn >= self._max_turns)

            # Build bundle
            bundle = self._make_bundle(
                brain_id, pass_rate=pass_rate, done=done)
            bundles[brain_id] = bundle

            if done:
                self._episode_pass_rates.append(pass_rate)
                self._evaluate_cert(brain_id, session)
                self._next_problem()
                self._current_turn = 0
                self._current_code = ""
                self._last_error   = ""

        self._pending_actions = {}
        return bundles

    # ── Bundle construction ────────────────────────────────────────────────────

    def _make_bundle(
        self,
        brain_id: str,
        pass_rate: float = 0.0,
        done:      bool  = False,
    ) -> SensorBundle:
        context = self._build_context()
        tokens  = self._tokenize(context)

        return SensorBundle(
            extra          = {'text_tokens': tokens,
                              'pass_rate':   np.array([pass_rate], dtype=np.float32)},
            proprioceptive = np.array([self._current_turn / self._max_turns],
                                       dtype=np.float32),
            reward         = pass_rate,
            done           = done,
            env_id         = self.env_id,
            agent_id       = brain_id,
            step           = self._step_count,
        )

    def _build_context(self) -> str:
        """Build the text context the brain sees: problem + previous attempt + error."""
        if self._current_problem is None:
            return ""
        ctx = f"### Problem: {self._current_problem['id']}\n"
        ctx += self._current_problem['description'] + "\n"
        if self._current_code:
            ctx += f"\n### Your previous code (turn {self._current_turn}):\n"
            ctx += self._current_code + "\n"
        if self._last_error:
            ctx += f"\n### Execution error:\n{self._last_error}\n"
        ctx += "\n### Write your solution:\n"
        return ctx

    def _tokenize(self, text: str) -> np.ndarray:
        """Simple character-level tokenization (brain uses real tokenizer in practice)."""
        if self._tokenizer is not None:
            ids = self._tokenizer(text)[:MAX_TOKENS]
        else:
            ids = [ord(c) % 256 for c in text[:MAX_TOKENS]]
        padded = ids + [0] * (MAX_TOKENS - len(ids))
        return np.array(padded[:MAX_TOKENS], dtype=np.int32)

    def _decode_code(self, action: ActionBundle) -> str:
        """Decode code from ActionBundle.text_tokens."""
        if action.text_tokens is not None:
            tokens = action.text_tokens
            if self._tokenizer is not None:
                return self._tokenizer.decode(tokens)
            return ''.join(chr(min(t, 127)) for t in tokens if t > 0)
        return ""

    # ── Problem management ────────────────────────────────────────────────────

    def _next_problem(self) -> None:
        try:
            self._current_problem = next(self._problem_iter)
        except StopIteration:
            from coding_env.problems import stream_problems
            self._problem_iter = stream_problems(self._problems)
            self._current_problem = next(self._problem_iter)

    # ── Certification ─────────────────────────────────────────────────────────

    def _evaluate_cert(self, brain_id: str, session: Session) -> None:
        """
        Run a mini eval after each episode to track cert progress.
        Issues CertSignal when brain achieves cert_threshold on cert_window problems.
        """
        if self._field is None:
            return
        if len(self._eval_results) < self._cert_window:
            return

        recent = self._eval_results[-self._cert_window:]
        avg_pass = sum(recent) / len(recent)
        passed   = avg_pass >= self._cert_threshold

        cert_env_id = "cert_coding_basics"
        self._field.register_agent(cert_env_id, pos=(3.0, 0.0))
        sig = CertSignal(
            position           = (3.0, 0.0),
            timestamp          = self._step_count,
            publisher          = cert_env_id,
            brain_id           = brain_id,
            cert_name          = "coding_basics",
            passed             = 1.0 if passed else 0.0,
            score              = float(avg_pass),
            retry_after_steps  = 5000,
        )
        self._field.publish(cert_env_id, sig)

        if self._verbose:
            print(f"CodingEnv cert: avg_pass={avg_pass:.2f} "
                  f"({'PASS' if passed else 'FAIL'})", flush=True)

    # ── Metrics ───────────────────────────────────────────────────────────────

    @property
    def avg_pass_rate(self) -> float:
        if not self._episode_pass_rates:
            return 0.0
        return sum(self._episode_pass_rates[-100:]) / min(100, len(self._episode_pass_rates))
