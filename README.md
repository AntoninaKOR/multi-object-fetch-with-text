# Multi-object Fetch

Visual RL environments for multi-object reasoning and manipulation with a Fetch robot. 
The following table shows examples of the different environments (Reach, Push, and Pick) as well as the relational
reasoning tasks (Red, Reddest, Odd, OddGroups). Multicolor samples a random target color each episode and names it in the instruction.

|           |                             Red                             |                            Reddest                             |                           Odd                            |                           OddGroups                            |
|:----------|:-----------------------------------------------------------:|:--------------------------------------------------------------:|:--------------------------------------------------------:|:--------------------------------------------------------------:|
| **Reach** | <img style="width: 128px;" src="docs/images/ReachRed.png">  | <img style="width: 128px;" src="docs/images/ReachReddest.png"> | <img style="width: 128px;" src="docs/images/ReachOdd.png"> | <img style="width: 128px;" src="docs/images/ReachOddGroups.png"> |
| **Push**  |  <img style="width: 128px;" src="docs/images/PushRed.png">  |  <img style="width: 128px;" src="docs/images/PushReddest.png">   | <img style="width: 128px;" src="docs/images/PushOdd.png">  | <img style="width: 128px;" src="docs/images/PushOddGroups.png">  |
| **Pick**  |  <img style="width: 128px;" src="docs/images/PickRed.png">  |  <img style="width: 128px;" src="docs/images/PickReddest.png">   | <img style="width: 128px;" src="docs/images/PickOdd.png">  | <img style="width: 128px;" src="docs/images/PickOddGroups.png">  |

## Installation
The easiest way to install is to use the provided `create_conda_env.sh` script. This creates a conda environment called `mof` with all the necessary dependencies, sets up MuJoCo and copies asset files.
```
./create_conda_env.sh
source ~/.bashrc
conda activate mof
```

## Getting Started
To verify that the installation was successful, environments can be run with a random or user-controlled policy. An example command to control the robot on a pick-and-place task is:
``` 
python examples/run.py --policy user --task Odd --num_distractors 2 --environment Pick
```

Multicolor: each episode picks a random target color; the streamed instruction names that color (e.g. `pick up the turquoise object`):
```
python examples/run.py --policy random --task Multicolor --num_distractors 2 --environment Pick
```

Stack: put one block on another (e.g. `put the red block on the blue block`); both colors are random each episode and the distractors are extra loose blocks. There is no target location: the goal is the top of the base block, wherever it is. In `Onto1` the base is a single block on the table; in `Onto2` it is the top block of a two-block tower that stands ready at the start of the episode (the instruction names the top block). In `Onto1Fixed` / `Onto2Fixed` the base (the whole tower) is welded in place; otherwise every block is free and can be pushed or knocked over (the episode goes on). A tower starts at least 12 cm (xy) from the gripper's start, which is at the height of the tower's second block. Success: the block rests on the base (its center within the half-diagonal of the base's top face horizontally, one block height above it), the tower still stands (each block on top of the one below in the same sense), and the block is released and still; holding it there does not count. Dense reward is 1 on success and otherwise `0.8 * exp(-20 * d) + 0.1 * exp(-10 * g)` (d: block to the top of the base, g: gripper to block); sparse reward is -1 until success. `Staged` (Stack only) is a stepped dense reward where every stage raises a floor: reach the block 0–0.2, grasped 0.4, lifted to the stacking height 0.5, aligned over the base up to +0.2, lowered onto it up to +0.1, on the base 0.8 while held and 0.9 once released, placed 1 (IDs `StackOnto1_2Distractors_Staged-v1` etc.). `ManiSkill` (Stack only) is ManiSkill StackCube-v1's normalized dense reward, `compute_dense_reward / 8`: reach `2(1 - tanh(5g))`, grasped `4 + (1 - tanh(5d))`, on the base `6 + (ungrasp + static) / 2` (finger opening fraction, `1 - tanh(10 v + w)`), placed 8; grasp, on-base and success use this env's definitions. The episode ends early, failed, if the block or a tower block falls off the table (a distractor falling off does not end it). Episodes have 100 steps.
```
python examples/run.py --policy user --task Onto1Fixed --num_distractors 2 --environment Stack
```

MulticolorShape (Reach only): targets and distractors are spheres, cubes, or the Language Table block shapes moon,
star and pentagon, and the instruction names color and shape (e.g. `reach the red star`). With at least one
distractor, one of them has the target's color and another shape; with at least two, another has the target's shape
and another color; every (color, shape) pair appears at most once. `target_shape` in `info` indexes
`MULTISHAPE_NAMES`. The mesh shapes are visual only, like the sites: 8 cm across (the sphere's diameter), never
touched by the gripper. They face the frontview camera, so the outline stays readable, with a random twist per
episode for variety (up to 30 degrees in the outline's plane and 20 degrees out of it; `ReachEnv.MESH_MAX_ROLL` /
`MESH_MAX_TILT`). Cubes are twisted too: any turn about the vertical and up to 20 degrees of tilt (`CUBE_MAX_YAW` /
`CUBE_MAX_TILT`). The mesh shapes are STL files in `multi_object_fetch/assets/meshes`, converted from Language
Table's OBJ blocks (MuJoCo 2.1 cannot read OBJ) with `scripts/convert_language_table_meshes.py`.
```
python examples/run.py --policy random --task MulticolorShape --num_distractors 4 --environment Reach
```

Held-out color-shape pairs, to test generalization to unseen combinations: `_Train` IDs (e.g.
`ReachMulticolorShape_4Distractors_Dense_Train-v1`) never show a held-out pair, on the target or any distractor;
`_Val` IDs always instruct a held-out pair (distractors may use any pair). IDs without a suffix use every pair. The
default held-out pairs (`MULTICOLORSHAPE_HELDOUT`) are 4 colors per shape; every color still appears in training
with at least 3 shapes, and both guaranteed distractors (same color, same shape) exist in every split:

| shape | held-out colors |
|:--|:--|
| sphere | green, dark slate gray, saddle brown, midnight blue |
| cube | dark green, dark khaki, turquoise, red |
| moon | orange, yellow, violet red, blue |
| star | spring green, thistle, magenta, dodger blue |
| pentagon | green, dark slate gray, saddle brown, midnight blue |

Other pairs can be held out with `gym.make(..., split='train' or 'val', heldout_pairs=[('red', 'star'), ...])`.
The `_Train` / `_Val` IDs have a fourth `_`-separated part, so they go through `gym.make`; dynalang's `mof_` task
names (`{Env}{Task}_{n}Distractors_{Reward}-v1`) do not parse them.

The following arguments are used to configure the environments and can be tested in `examples/run.py`:
- `policy`: Whether to control the agent via the keyboard or run a random policy be in `random`, `user`.
- `environment`: Selects the environment to run and can be in `Reach`, `Push`, `Pick`, and `Stack`.
- `task`: Selects which task to run and can be in `Red`, `Reddest`, `Odd`, `OddGroups`, `Multicolor`, and `MulticolorShape` (Reach only), or `Onto1`, `Onto1Fixed`, `Onto2` and `Onto2Fixed` (Stack).
- `num_distractors`: Number of distractor targets/blocks to use in the environment.
- `no_language`: Disable HomeGrid-style parallel token streaming.
- `no_viz`: Disable the on-frame instruction overlay.

By default `examples/run.py` wraps the env in `LanguageWrapper`, which streams one T5 token per step in parallel with actions (`token`, `token_embed` in the observation).

## Language embeddings
Task instructions are a closed set listed in `multi_object_fetch/language/sentences.txt`. Token ids and T5-small embeddings are precomputed into `multi_object_fetch/language/embeds.pkl` so the environment does not need a language model at runtime.

A cached `embeds.pkl` is included in the repo. Rebuild it after editing `sentences.txt`:
```
pip install torch transformers
python scripts/embed_offline.py \
  --infile multi_object_fetch/language/sentences.txt \
  --outfile multi_object_fetch/language/embeds.pkl \
  --model t5
```

## Credits
This repository is an extension of the environments in [fetch-block-construction](https://github.com/richardrl/fetch-block-construction).
The moon, star and pentagon meshes are from [Language Table](https://github.com/google-research/language-table) (Apache 2.0).
