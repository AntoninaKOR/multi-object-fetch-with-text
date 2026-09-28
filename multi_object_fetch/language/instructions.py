"""Closed-set task instructions for multi-object Fetch."""

from typing import Dict, Tuple


_ACTION = {
    "Reach": "reach",
    "Push": "push",
    "Pick": "pick up",
}

_TARGET = {
    "Red": "the red object",
    "Reddest": "the reddest object",
    "Odd": "the odd one out",
    "OddGroups": "the odd one out",
}


def build_instruction(environment: str, task: str) -> str:
    """Return the canonical English instruction for an environment/task pair."""
    if environment not in _ACTION:
        raise ValueError(f"Unknown environment: {environment}")
    if task not in _TARGET:
        raise ValueError(f"Unknown task: {task}")
    return f"{_ACTION[environment]} {_TARGET[task]}"


INSTRUCTIONS: Dict[Tuple[str, str], str] = {
    (environment, task): build_instruction(environment, task)
    for environment in _ACTION
    for task in _TARGET
}
