"""
Problem sources — plug-and-play with CodingEnvironment.

Usage:
    from coding_env.sources import HumanEvalSource, MBPPSource, ProceduralSource

    env = CodingEnvironment(
        problems=HumanEvalSource().load(),
        eval_problems=HumanEvalSource(split='test').load(),
    )

    # Infinite procedural stream:
    env = CodingEnvironment(
        problems=list(ProceduralSource().sample(n=1000)),
    )
"""
from coding_env.sources.huggingface import HumanEvalSource, MBPPSource, CodeContestsSource
from coding_env.sources.procedural  import ProceduralSource

__all__ = [
    "HumanEvalSource", "MBPPSource", "CodeContestsSource",
    "ProceduralSource",
]
