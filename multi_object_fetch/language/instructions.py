"""Closed-set task instructions for multi-object Fetch."""

from typing import Dict, Optional, Tuple

# Must stay aligned with ColorsMixin.colors index order in env.py.
COLOR_NAMES = (
    "green",
    "dark slate gray",
    "saddle brown",
    "midnight blue",
    "dark green",
    "dark khaki",
    "turquoise",
    "red",
    "orange",
    "yellow",
    "violet red",
    "blue",
    "spring green",
    "thistle",
    "magenta",
    "dodger blue",
)

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

# Stack tasks name the block to move and the block to put it on.
STACK_TASKS = ("Onto1", "Onto1Fixed")


def build_instruction(
    environment: str, task: str, color_name: Optional[str] = None, base_color_name: Optional[str] = None
) -> str:
    """Return the canonical English instruction for an environment/task pair.

    For Multicolor, pass ``color_name`` (from COLOR_NAMES) chosen for this episode.
    For Stack, also pass ``base_color_name``, the color of the block to stack onto.
    """
    if environment == "Stack":
        if task not in STACK_TASKS:
            raise ValueError(f"Unknown task: {task}")
        for name in (color_name, base_color_name):
            if name not in COLOR_NAMES:
                raise ValueError(f"Unknown color name: {name}")
        return f"put the {color_name} block on the {base_color_name} block"
    if environment not in _ACTION:
        raise ValueError(f"Unknown environment: {environment}")
    if task == "Multicolor":
        if color_name is None:
            raise ValueError("Multicolor requires color_name")
        if color_name not in COLOR_NAMES:
            raise ValueError(f"Unknown color name: {color_name}")
        return f"{_ACTION[environment]} the {color_name} object"
    if task not in _TARGET:
        raise ValueError(f"Unknown task: {task}")
    return f"{_ACTION[environment]} {_TARGET[task]}"


INSTRUCTIONS: Dict[Tuple[str, ...], str] = {
    (environment, task): build_instruction(environment, task)
    for environment in _ACTION
    for task in _TARGET
}

# Episode-varying Multicolor utterances for all environments.
INSTRUCTIONS.update(
    {
        (environment, "Multicolor", color_name): build_instruction(
            environment, "Multicolor", color_name=color_name
        )
        for environment in _ACTION
        for color_name in COLOR_NAMES
    }
)

# Episode-varying Stack utterances: every ordered pair of distinct colors.
INSTRUCTIONS.update(
    {
        ("Stack", task, color_name, base_color_name): build_instruction(
            "Stack", task, color_name=color_name, base_color_name=base_color_name
        )
        for task in STACK_TASKS
        for color_name in COLOR_NAMES
        for base_color_name in COLOR_NAMES
        if color_name != base_color_name
    }
)
