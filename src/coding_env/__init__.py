"""
coding-env: multi-turn code execution environment as ecoframe EnvironmentProtocol.

RLEF-style training: brain generates code → execute → see error → revise.
10x more sample efficient than single-shot code generation (RLEF, ICML 2025).

Usage:
    from coding_env import CodingEnvironment, CODING_MANIFEST
    from ecoframe import TrainingEngine, Field

    field = Field(backend='local')
    env   = CodingEnvironment(field=field)
    env.register_env_signal()   # publish to Field for discovery
    engine = TrainingEngine(brain, env)
    for step, metrics in engine.run(n_steps=100_000):
        log(metrics)
"""
from coding_env.environment import CodingEnvironment, CODING_MANIFEST, HARDWARE
from coding_env.executor    import execute, ExecutionResult
from coding_env.problems    import (builtin_problems, training_problems,
                                    eval_problems, stream_problems)

__version__ = "0.1.0"
__all__ = [
    "CodingEnvironment", "CODING_MANIFEST", "HARDWARE",
    "execute", "ExecutionResult",
    "builtin_problems", "training_problems", "eval_problems", "stream_problems",
]
