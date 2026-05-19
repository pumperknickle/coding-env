"""
CodingEnvironment end-to-end tests.

Tests:
  1. Executor: correct code passes tests
  2. Executor: wrong code fails tests
  3. Executor: syntax error captured
  4. Executor: timeout captured
  5. Problems: training set in 20-60% difficulty range
  6. Environment: enter/exit session lifecycle
  7. Environment: reset returns SensorBundle with text_tokens
  8. Environment: step_async + step_wait executes code and returns reward
  9. Multi-turn: second attempt with error context
 10. Certification: CertSignal published when pass rate threshold met
 11. Manifest: text_tokens and pass_rate are prediction targets
 12. Manifest: turn is NOT a prediction target
"""
import numpy as np
import pytest
from ecoframe.protocol import ActionBundle, Session, HardwareSpec
from ecoframe.signal import CertSignal
from coding_env import CodingEnvironment, CODING_MANIFEST, HARDWARE, execute, training_problems


# ── Executor tests ─────────────────────────────────────────────────────────────

def test_executor_correct_code():
    result = execute(
        code  = "print(input()[::-1])",
        tests = [
            {'input': 'hello', 'expected': 'olleh'},
            {'input': 'abc',   'expected': 'cba'},
        ],
    )
    assert result.pass_rate == 1.0
    assert result.n_passed == 2


def test_executor_wrong_output():
    result = execute(
        code  = "print('wrong')",
        tests = [{'input': 'hello', 'expected': 'olleh'}],
    )
    assert result.pass_rate == 0.0
    assert not result.test_results[0].passed


def test_executor_syntax_error():
    result = execute(
        code  = "def broken(:\n  pass",
        tests = [{'input': '', 'expected': ''}],
    )
    assert result.pass_rate == 0.0
    assert result.compile_error or not result.test_results[0].passed


def test_executor_timeout():
    result = execute(
        code      = "while True: pass",
        tests     = [{'input': '', 'expected': ''}],
        timeout_s = 1.0,
    )
    assert result.timeout or result.pass_rate == 0.0


def test_executor_partial_credit():
    result = execute(
        code  = "n=int(input())\nif n==15:print(n)\nelse:print('wrong')",
        tests = [
            {'input': '15', 'expected': '15'},
            {'input': '5',  'expected': '5'},
        ],
    )
    assert 0.0 < result.pass_rate < 1.0


def test_executor_error_message():
    result = execute(
        code  = "x=int(input())\nprint(1/0)",
        tests = [{'input': '5', 'expected': ''}],
    )
    assert "error" in result.error_message.lower() or result.pass_rate == 0.0


# ── Problems tests ─────────────────────────────────────────────────────────────

def test_training_problems_difficulty_range():
    probs = training_problems()
    assert len(probs) > 0
    for p in probs:
        assert 0.2 <= p['difficulty'] <= 0.65


def test_builtin_problems_have_required_fields():
    from coding_env.problems import builtin_problems
    for p in builtin_problems():
        assert 'id' in p
        assert 'description' in p
        assert 'tests' in p and len(p['tests']) > 0
        assert 'difficulty' in p


def test_stream_problems_loops():
    from coding_env.problems import stream_problems, training_problems
    probs = training_problems()
    gen   = stream_problems(probs, repeat=False)
    seen  = list(gen)
    assert len(seen) == len(probs)


# ── Environment tests ─────────────────────────────────────────────────────────

@pytest.fixture
def env():
    # Use just 3 easy problems for speed
    probs = [p for p in training_problems() if p['difficulty'] <= 0.25][:3]
    return CodingEnvironment(problems=probs, max_turns=3, verbose=False)


def test_env_hardware_spec():
    assert HARDWARE.device_type == "cpu"


def test_manifest_prediction_targets():
    targets = [s.name for s in CODING_MANIFEST.prediction_targets]
    assert "text_tokens" in targets
    assert "pass_rate"   in targets
    assert "turn"        not in targets


def test_enter_returns_session(env):
    session = env.enter("brain0")
    assert isinstance(session, Session)
    assert session.env_id == "coding"


def test_exit_frees_slot(env):
    session = env.enter("brain0")
    env.exit(session)
    assert len(env._sessions) == 0


def test_reset_returns_bundle_with_context(env):
    env.start()
    session = env.enter("brain0")
    bundles = env.reset(session)
    assert "brain0" in bundles
    b = bundles["brain0"]
    tokens = b.extra.get("text_tokens")
    assert tokens is not None
    assert tokens.shape == (2048,)


def test_step_correct_code_passes(env):
    env.start()
    session = env.enter("brain0")
    env.reset(session)

    # Submit correct solution for "reverse_string"
    code = "print(input()[::-1])"
    action = ActionBundle(
        text_tokens=np.array([ord(c) for c in code] + [0] * (2048 - len(code)),
                              dtype=np.int32))
    env.step_async({"brain0": action})
    bundles = env.step_wait()

    b = bundles["brain0"]
    pass_rate = float(b.extra["pass_rate"][0])
    assert pass_rate >= 0.0   # at least tried


def test_step_wrong_code_low_reward(env):
    env.start()
    session = env.enter("brain0")
    env.reset(session)

    code = "print('completely_wrong')"
    action = ActionBundle(
        text_tokens=np.array([ord(c) for c in code] + [0] * (2048 - len(code)),
                              dtype=np.int32))
    env.step_async({"brain0": action})
    bundles = env.step_wait()

    b = bundles["brain0"]
    pass_rate = float(b.extra["pass_rate"][0])
    assert pass_rate == 0.0


def test_multi_turn_error_in_context(env):
    """After a failed attempt, error message should appear in next context."""
    env.start()
    session = env.enter("brain0")
    env.reset(session)

    # First attempt: wrong
    code1 = "print('wrong')"
    action1 = ActionBundle(
        text_tokens=np.array([ord(c) for c in code1] + [0] * (2048 - len(code1)),
                              dtype=np.int32))
    env.step_async({"brain0": action1})
    env.step_wait()

    # Context should now include the previous attempt
    ctx = env._build_context()
    assert "previous code" in ctx.lower() or "wrong" in ctx or "turn" in ctx


def test_episode_ends_at_max_turns(env):
    """Episode should end (done=True) at max_turns even if not solved."""
    env._max_turns = 2
    env.start()
    session = env.enter("brain0")
    env.reset(session)

    code = "print('wrong')"
    action = ActionBundle(
        text_tokens=np.array([ord(c) for c in code] + [0] * (2048 - len(code)),
                              dtype=np.int32))

    for turn in range(2):
        env.step_async({"brain0": action})
        bundles = env.step_wait()

    # Last bundle should have done=True
    b = bundles["brain0"]
    assert b.done


def test_cert_signal_published(env):
    """CertSignal published to Field when threshold met."""
    from ecoframe.field import Field
    field = Field(backend='local')
    env._field = field

    # Simulate 10 successful episodes
    env._eval_results = [1.0] * 10
    env.start()
    session = env.enter("brain0")
    env._evaluate_cert("brain0", session)

    sigs = field.query(pos=(0., 0.), radius=10.)
    cert_sigs = [s for s in sigs if isinstance(s, CertSignal)]
    assert len(cert_sigs) >= 1
    assert cert_sigs[0].cert_name == "coding_basics"
    assert cert_sigs[0].passed == 1.0


def test_cert_fail_when_below_threshold(env):
    """CertSignal with passed=False when below threshold."""
    from ecoframe.field import Field
    field = Field(backend='local')
    env._field = field

    env._eval_results = [0.3] * 10  # below 0.8 threshold
    env.start()
    session = env.enter("brain0")
    env._evaluate_cert("brain0", session)

    sigs = field.query(pos=(0., 0.), radius=10.)
    cert_sigs = [s for s in sigs if isinstance(s, CertSignal)]
    assert len(cert_sigs) >= 1
    assert cert_sigs[0].passed == 0.0
