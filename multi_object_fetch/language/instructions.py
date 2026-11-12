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

# Must stay aligned with ColorsMixin.shape_geoms index order in env.py.
SHAPE_NAMES = ("sphere", "cube")
# Language Table block shapes, rendered from meshes (multi_object_fetch/assets/meshes/<name>.stl).
MESH_SHAPE_NAMES = ("moon", "star", "pentagon")
# MulticolorShape: the primitive shapes, then the mesh shapes.
MULTISHAPE_NAMES = SHAPE_NAMES + MESH_SHAPE_NAMES
# MulticolorShape (color name, shape name) pairs kept out of training (split="train") and used as targets for
# validation (split="val"): 4 of the 16 colors per shape, shifting by 4 colors per shape, so every color is held
# out for at most 2 shapes and still appears in training with at least 3.
MULTICOLORSHAPE_HELDOUT = tuple(
    (COLOR_NAMES[(4 * s + k) % len(COLOR_NAMES)], shape)
    for s, shape in enumerate(MULTISHAPE_NAMES)
    for k in range(4)
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

# Stack tasks name the block to move and the block to put it on (for Onto2, the top block of the tower).
STACK_TASKS = ("Onto1", "Onto1Fixed", "Onto2", "Onto2Fixed")


def build_instruction(
    environment: str, task: str, color_name: Optional[str] = None, shape_name: Optional[str] = None,
    base_color_name: Optional[str] = None,
) -> str:
    """Return the canonical English instruction for an environment/task pair.

    For Multicolor, pass ``color_name`` (from COLOR_NAMES) chosen for this episode.
    For MulticolorShape, also pass ``shape_name`` (from MULTISHAPE_NAMES).
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
    if task == "MulticolorShape":
        if color_name not in COLOR_NAMES:
            raise ValueError(f"Unknown color name: {color_name}")
        if shape_name not in MULTISHAPE_NAMES:
            raise ValueError(f"Unknown shape name: {shape_name}")
        return f"{_ACTION[environment]} the {color_name} {shape_name}"
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

# MulticolorShape exists for Reach only.
INSTRUCTIONS.update(
    {
        ("Reach", "MulticolorShape", color_name, shape_name): build_instruction(
            "Reach", "MulticolorShape", color_name=color_name, shape_name=shape_name
        )
        for color_name in COLOR_NAMES
        for shape_name in MULTISHAPE_NAMES
    }
)
