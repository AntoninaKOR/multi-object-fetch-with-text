"""Tests for the per-step metric values that MultiObjectFetchEnv writes into `info` (see docs/metrics.md)."""
from itertools import combinations

import gym
import numpy as np
import pytest

if not hasattr(gym, 'GoalEnv'):
    # gym 0.26 (used by dynalang) removed GoalEnv, which fetch-block-construction still subclasses.
    class GoalEnv(gym.Env):
        def compute_reward(self, achieved_goal, desired_goal, info):
            raise NotImplementedError

    gym.GoalEnv = GoalEnv

from fetch_block_construction.envs.robotics import fetch_env  # noqa: E402
from mujoco_py.builder import MujocoException  # noqa: E402
from mujoco_py.generated import const  # noqa: E402

import multi_object_fetch.env  # noqa: E402  # also registers the gym envs
from multi_object_fetch.language import LanguageWrapper
from multi_object_fetch.language.instructions import (COLOR_NAMES, INSTRUCTIONS, MESH_SHAPE_NAMES,
                                                    MULTICOLORSHAPE_HELDOUT, MULTISHAPE_NAMES)


def make(name, seed=0):
    """Build the env like dynalang does: unwrapped, so the same code runs under gym 0.17 and 0.26."""
    try:
        env = gym.make(name, disable_env_checker=True)
    except TypeError:
        env = gym.make(name)
    env = env.unwrapped
    env.seed(seed)
    return env


def episode_length(name):
    return 50 if name.startswith('Reach') else 100


def move_towards(env, pos, gripper):
    grip = env.unwrapped.sim.data.get_site_xpos('robot0:grip')
    delta = np.clip((pos - grip) / 0.05, -1, 1)
    return np.append(delta, gripper)


FINGERS = set(multi_object_fetch.env.ManipulateEnv.FINGERS)


def check_step(info):
    """Invariants that must hold at every step."""
    assert info['success'] == (info['dist'] < env_threshold())
    assert info['unstable'] is False
    if 'lifted' in info:
        env_cls = multi_object_fetch.env.ManipulateEnv
        contacts = set(info['object_contacts'])
        assert info['grasped'] == (FINGERS.issubset(contacts) and info['between_fingers'])
        assert info['between_fingers'] == (abs(info['finger_gap'] - info['object_width']) < env_cls.GRASP_GAP_TOLERANCE
                                           and abs(info['grasp_center_offset']) < env_cls.GRASP_CENTER_TOLERANCE)
        assert info['lifted'] == (not (contacts - FINGERS) and info['object_lift'] > env_cls.LIFT_THRESHOLD)
        assert info['held'] == (info['grasped'] and info['lifted'])
        if info['placed']:
            assert info['success']
            if info['goal_needs_lift']:
                assert info['held']
            else:
                assert not info['grasped'] and 'table0' in contacts and info['object_speed'] < env_cls.PLACED_SPEED


def run_plan(env, plan):
    """Follow (position, gripper, steps) waypoints; return the info of every step."""
    infos = []
    for pos, gripper, steps in plan:
        for _ in range(steps):
            _, _, _, info = env.step(move_towards(env, np.asarray(pos), gripper))
            check_step(info)
            infos.append(info)
    return infos


def site(env, name):
    return env.sim.data.get_site_xpos(name).copy()


def set_object(env, name, pos=None, yaw=None):
    """Teleport a block (and optionally rotate it about the vertical axis), then settle it."""
    qpos = env.sim.data.get_joint_qpos(f'{name}:joint').copy()
    if pos is not None:
        qpos[:3] = pos
    if yaw is not None:
        qpos[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    env.sim.data.set_joint_qpos(f'{name}:joint', qpos)
    env.sim.forward()


def env_threshold():
    return multi_object_fetch.env.MultiObjectFetchEnv.SUCCESS_THRESHOLD


@pytest.mark.parametrize('name', ['ReachMulticolor_2Distractors_Dense-v1', 'PushMulticolor_3Distractors_Dense-v1',
                                  'PickMulticolor_4Distractors_Sparse-v1', 'PickRed_1Distractors_Dense-v1',
                                  'ReachOdd_2Distractors_Dense-v1', 'PushOddGroups_4Distractors_Dense-v1'])
def test_info_keys_and_constants(name):
    env = LanguageWrapper(make(name))
    env.action_space.seed(0)
    manipulate = not name.startswith('Reach')
    for _ in range(2):
        env.reset()
        _, _, _, info = env.step(env.action_space.sample())
        first = dict(info)
        for _ in range(5):
            _, _, _, info = env.step(env.action_space.sample())
            check_step(info)
            for key in ('num_distractors', 'target_color') + (('goal_height',) if manipulate else ()):
                assert info[key] == first[key]
        assert info['num_distractors'] == int(name.split('_')[1][0])
        if 'Multicolor' in name:
            assert 0 <= info['target_color'] < len(COLOR_NAMES)
            assert info['instruction'] == env.unwrapped.instruction
            assert f" {COLOR_NAMES[info['target_color']]} object" in info['instruction']
        else:
            assert info['target_color'] == -1
        assert ('goal_height' in info) == manipulate
        assert ('object_lift' in info) == manipulate
        assert ('grip_object_dist' in info) == manipulate
        for key in ('lifted', 'held', 'grasped', 'grasped_geometric', 'between_fingers', 'grasp_center_offset',
                    'placed', 'object_contacts', 'finger_gap', 'object_width', 'object_speed', 'goal_needs_lift'):
            assert (key in info) == manipulate, key
        if name.startswith('Push'):
            assert info['goal_height'] == pytest.approx(0.0, abs=1e-6)


def test_reach_scripted_success():
    env = make('ReachMulticolor_2Distractors_Dense-v1')
    successes = 0
    for _ in range(3):
        env.reset()
        dists = []
        for _ in range(episode_length('Reach')):
            target = env.unwrapped.sim.data.get_site_xpos('target0').copy()
            _, _, _, info = env.step(move_towards(env, target, 0.0))
            check_step(info)
            dists.append(info['dist'])
        successes += info['success']
        assert dists[-1] < dists[0]
    assert successes == 3


def pick_plan(obj, goal, grasp_height=0.0):
    grasp = obj + [0, 0, grasp_height]
    return [(grasp + [0, 0, 0.1], 1.0, 15), (grasp, 1.0, 15), (grasp, -1.0, 8),
            (grasp + [0, 0, 0.15], -1.0, 15), (goal + [0, 0, grasp_height], -1.0, 40)]


def air_goal(b):
    return b.goal[2] - b.height_offset > 0.12


def table_goal(b):
    return b.goal[2] - b.height_offset < 0.01


def reset_until(env, accept):
    for _ in range(100):
        env.reset()
        if accept(env):
            return
    raise RuntimeError('no accepted scene')


def test_pick_air_goal_grasp_lift_hold_place():
    env = make('PickRed_0Distractors_Dense-v1', seed=1)
    for _ in range(3):
        reset_until(env, air_goal)
        infos = run_plan(env, pick_plan(site(env, 'object0'), env.goal.copy()))
        assert infos[0]['object_lift'] == pytest.approx(0.0, abs=0.005)  # block starts on the table
        assert not infos[0]['grasped'] and not infos[0]['lifted']
        grasped = [i['grasped'] for i in infos]
        assert any(grasped) and any(i['held'] for i in infos)
        # A centered grasp: the geometric cross-check agrees with the contacts, except on the first
        # contact step, when the fingers are still closing and the gap has not settled yet.
        geometric = [i['grasped_geometric'] for i in infos]
        disagree = [k for k in range(len(infos)) if grasped[k] != geometric[k]]
        assert all(grasped[k] and not grasped[k - 1] for k in disagree), disagree
        held = [i for i in infos if i['held']]
        assert all(abs(i['finger_gap'] - i['object_width']) < 0.005 for i in held)
        assert infos[-1]['goal_needs_lift'] and infos[-1]['success'] and infos[-1]['placed']


def test_pick_table_goal_placed_only_after_release():
    env = make('PickRed_0Distractors_Dense-v1', seed=1)
    reset_until(env, table_goal)
    goal = env.goal.copy()
    infos = run_plan(env, pick_plan(site(env, 'object0'), goal)[:-1] + [
        (goal + [0, 0, 0.005], -1.0, 40),  # lower onto the goal, still holding
        (goal + [0, 0, 0.005], 1.0, 5),  # release
        (goal + [0, 0, 0.15], 1.0, 15),  # move away
    ])
    holding_at_goal = infos[15 + 15 + 8 + 15 + 39]
    assert holding_at_goal['success'] and holding_at_goal['grasped'] and not holding_at_goal['placed']
    assert not infos[-1]['grasped'] and infos[-1]['success'] and infos[-1]['placed']
    assert not infos[-1]['goal_needs_lift']


def test_block_on_distractor_is_not_lifted():
    env = make('PickRed_1Distractors_Dense-v1')
    env.reset()
    other = site(env, 'object1')
    set_object(env, 'object0', pos=other + [0, 0, 2 * 0.045 + 0.001])
    infos = run_plan(env, [(site(env, 'robot0:grip'), 1.0, 10)])
    info = infos[-1]
    assert info['object_lift'] > 0.08  # stacked: 9 cm above the table, which the old height rule called lifted
    assert 'object1' in info['object_contacts']
    assert not info['lifted'] and not info['held'] and not info['grasped']


def test_grasp_near_top_edge():
    """Fingers close on the upper part of the cube: still a grasp, though the center is off the pads."""
    env = make('PickRed_0Distractors_Dense-v1', seed=2)
    reset_until(env, air_goal)
    infos = run_plan(env, pick_plan(site(env, 'object0'), env.goal.copy(), grasp_height=0.03))
    held = [i for i in infos if i['held']]
    assert held, 'the top-edge grasp should hold the cube'
    assert max(i['grip_object_dist'] for i in held) > 0.025  # the old 3 cm rule was borderline here
    # The pads are 11.6 cm tall, so the center is still between them: both checks agree.
    assert all(i['grasped_geometric'] for i in held)


def test_rotated_cube_width_and_grasp():
    env = make('PickRed_0Distractors_Dense-v1', seed=3)
    reset_until(env, air_goal)
    yaw = np.deg2rad(20)
    set_object(env, 'object0', yaw=yaw)
    infos = run_plan(env, [(site(env, 'object0') + [0, 0, 0.1], 1.0, 15)])
    expected = 0.09 * (np.cos(yaw) + np.sin(yaw))  # width of the rotated cube along the closing axis
    assert infos[-1]['object_width'] == pytest.approx(expected, abs=0.004)
    infos = run_plan(env, pick_plan(site(env, 'object0'), env.goal.copy())[1:])
    assert any(i['held'] for i in infos)


def test_closed_fingers_beside_cube_are_not_a_grasp():
    env = make('PickRed_0Distractors_Dense-v1', seed=4)
    env.reset()
    beside = site(env, 'object0') - [0.12, 0, 0]  # pads (2 cm) and cube (4.5 cm) half-widths do not overlap
    infos = run_plan(env, [(beside + [0, 0, 0.1], 1.0, 15), (beside, 1.0, 15), (beside, -1.0, 10)])
    assert abs(site(env, 'robot0:grip')[0] - beside[0]) < 0.01  # the gripper reached the spot
    assert not any(i['grasped'] or i['grasped_geometric'] or i['lifted'] for i in infos)


def test_edge_pinch_is_a_contact_grasp_but_not_geometric():
    """Fingers overlapping the cube's edge by ~5 mm: both fingers squeeze it, but its center is not between the pads."""
    env = make('PickRed_0Distractors_Dense-v1', seed=4)
    env.reset()
    obj = site(env, 'object0')
    edge = obj + [0.045 + 0.02 - 0.005, 0, 0]
    infos = run_plan(env, [(edge + [0, 0, 0.1], 1.0, 15), (edge, 1.0, 15), (edge, -1.0, 10),
                           (edge + [0, 0, 0.15], -1.0, 15)])
    pinched = infos[39]
    assert pinched['grasped'] and not pinched['grasped_geometric']
    # It is a real grasp: the pinch lifts the cube with the fingers alone.
    assert any(i['held'] for i in infos) and max(i['object_lift'] for i in infos) > 0.1


def test_push_with_closed_fingers_is_not_a_grasp():
    """Pushing with the (blocked, closed) fingers presses both fingertips on one face of the block."""
    env = make('PushRed_0Distractors_Dense-v1', seed=0)
    env.reset()
    obj, goal = site(env, 'object0'), env.goal.copy()
    u = np.append((goal - obj)[:2] / np.linalg.norm((goal - obj)[:2]), 0)
    behind = obj - u * 0.09
    infos = run_plan(env, [(site(env, 'robot0:grip') + [0, 0, 0.15], 1.0, 4), (behind + [0, 0, 0.15], 1.0, 8),
                           (behind + [0, 0, 0.005], 1.0, 6), (goal + [0, 0, 0.005], 1.0, 20)])
    both_touch = [i for i in infos if FINGERS.issubset(i['object_contacts'])]
    assert both_touch, 'the push should press both fingertips against the block'
    assert all(i['finger_gap'] < 0.05 for i in both_touch)  # closed fingers, far narrower than the 9 cm block
    assert not any(i['grasped'] or i['lifted'] or i['held'] for i in infos)


def test_dropped_block_is_lifted_but_not_held():
    env = make('PickRed_0Distractors_Dense-v1', seed=1)
    reset_until(env, air_goal)
    obj = site(env, 'object0')
    infos = run_plan(env, pick_plan(obj, env.goal.copy())[:4] + [(obj + [0, 0, 0.15], 1.0, 15)])
    falling = [i for i in infos[46:] if i['lifted'] and not i['grasped']]
    assert falling, 'right after release the block is in the air without a grasp'
    assert not any(i['held'] for i in falling)
    assert not infos[-1]['lifted']  # back on the table


@pytest.mark.parametrize('name', ['ReachMulticolor_2Distractors_Dense-v1', 'PickMulticolor_2Distractors_Dense-v1'])
def test_unstable_step(name, monkeypatch):
    env = make(name)
    env.reset()
    env.step(env.action_space.sample())

    def explode(self, action):
        raise MujocoException('forced instability')

    monkeypatch.setattr(fetch_env.FetchEnv, 'step', explode)
    _, reward, done, info = env.step(env.action_space.sample())
    assert done and reward == 0.0
    assert info['success'] is False and info['unstable'] is True
    for key in ('dist', 'object_lift', 'grip_object_dist', 'lifted', 'held', 'grasped', 'placed'):
        assert key not in info
    assert info['num_distractors'] == 2
    assert 0 <= info['target_color'] < len(COLOR_NAMES)
    assert ('goal_height' in info) == name.startswith('Pick')


STACK_TASKS = ['Onto1', 'Onto1Fixed', 'Onto2', 'Onto2Fixed']


def stack_plan(obj, goal):
    """Grasp object0, carry it high (above the goal), lower it onto the base and let go.

    The approach stays at the goal height so that the gripper does not sweep a tall tower over."""
    above = goal + [0, 0, 0.08]
    approach = obj + [0, 0, max(0.1, goal[2] - obj[2])]
    lift = obj + [0, 0, max(0.2, above[2] - obj[2])]
    return [(approach, 1.0, 15), (obj, 1.0, 15), (obj, -1.0, 8), (lift, -1.0, 15),
            (above, -1.0, 30), (goal, -1.0, 20), (goal, 1.0, 10),
            (above, 1.0, 15)]


def run_stack_plan(env, plan):
    infos, rewards = [], []
    for pos, gripper, steps in plan:
        for _ in range(steps):
            _, reward, done, info = env.step(move_towards(env, np.asarray(pos), gripper))
            check_stack_step(env, info)
            assert not done
            infos.append(info)
            rewards.append(reward)
    return infos, rewards


def check_stack_step(env, info):
    assert info['success'] == info['placed']
    assert info['placed'] == (info['on_base'] and info['base_resting'] and env.base_name in info['object_contacts']
                              and not info['grasped'] and info['object_speed'] < env.PLACED_SPEED)
    assert info['dist'] == pytest.approx(np.linalg.norm(site(env, 'object0') - env.goal), abs=1e-6)


@pytest.mark.parametrize('task', STACK_TASKS)
@pytest.mark.parametrize('num_distractors', [0, 2, 4])
def test_stack_scene(task, num_distractors):
    env = LanguageWrapper(make(f'Stack{task}_{num_distractors}Distractors_Dense-v1'))
    b = env.unwrapped
    height = int(task[len('Onto')])
    assert b.tower_height == height and b.tower_names == [f'object{i}' for i in range(1, height + 1)]
    assert b.num_blocks == num_distractors + 1 + height
    for name in b.tower_names:  # Fixed: the whole tower is welded
        assert (f'{name}:joint' in b.sim.model.joint_names) != task.endswith('Fixed')
    for _ in range(5):
        env.reset()
        assert b.sim.model.site_rgba[b.sim.model.site_name2id('target0')][3] == 0  # no target location
        rgba = [tuple(b.sim.model.geom_rgba[b.sim.model.geom_name2id(n)][:3]) for n in b.object_names]
        assert len(set(rgba)) == b.num_blocks
        assert np.allclose(rgba[0], b.colors[b.target_color]) and np.allclose(rgba[height], b.colors[b.base_color])
        assert b.instruction == f'put the {COLOR_NAMES[b.target_color]} block on the {COLOR_NAMES[b.base_color]} block'
        assert np.allclose(b.goal, site(b, b.base_name) + [0, 0, 2 * b.object_size])
        assert b.goal[2] == pytest.approx(b.height_offset + 2 * b.object_size * height, abs=0.001)
        bottom = site(b, 'object1')
        if height > 1:  # clear of the gripper, which starts at the height of the second block
            assert np.linalg.norm(bottom[:2] - b.initial_gripper_xpos[:2]) >= b.TOWER_GRIPPER_CLEARANCE
        for level, name in enumerate(b.tower_names):  # stacked straight up
            assert np.allclose(site(b, name), bottom + [0, 0, 2 * b.object_size * level], atol=0.001)
        scattered = [site(b, n) for n in b.object_names if n not in b.tower_names[1:]]
        for i in range(len(scattered)):
            assert scattered[i][2] == pytest.approx(b.height_offset, abs=0.001)
            for j in range(i):
                assert np.linalg.norm(scattered[i][:2] - scattered[j][:2]) >= 2 * np.sqrt(2) * b.object_size - 1e-9
        _, reward, done, info = env.step(np.zeros(4))
        check_stack_step(b, info)
        assert 0 <= reward < 0.9 and not done
        assert not info['success'] and not info['on_base'] and info['base_resting'] and not info['dropped']
        assert info['instruction'] == b.instruction
        assert (info['target_color'], info['base_color']) == (b.target_color, b.base_color)
        assert info['fixed_base'] == task.endswith('Fixed') and info['goal_needs_lift']
        assert info['tower_height'] == height
        assert info['goal_height'] == pytest.approx(2 * b.object_size * height)


@pytest.mark.parametrize('task', STACK_TASKS)
def test_stack_scripted_success(task):
    env = make(f'Stack{task}_2Distractors_Dense-v1', seed=1)
    for _ in range(3):
        env.reset()
        infos, rewards = run_stack_plan(env, stack_plan(site(env, 'object0'), env.goal.copy()))
        assert any(i['held'] for i in infos)
        holding = [k for k, i in enumerate(infos) if i['on_base'] and i['grasped']]
        assert holding and not any(infos[k]['success'] for k in holding)  # not placed while still held
        assert max(rewards[k] for k in holding) < 0.9
        assert infos[-1]['success'] and infos[-1]['placed'] and not infos[-1]['grasped']
        assert rewards[-1] == 1.0
        assert infos[-1]['base_shift'] < 0.01


def test_stack_sparse_reward():
    env = make('StackOnto1Fixed_0Distractors_Sparse-v1', seed=1)
    env.reset()
    _, rewards = run_stack_plan(env, stack_plan(site(env, 'object0'), env.goal.copy()))
    assert rewards[0] == -1.0 and rewards[-1] == 0.0


@pytest.mark.parametrize('task', ['Onto1', 'Onto1Fixed', 'Onto2Fixed'])
def test_stack_push_base(task):
    """The gripper pushes into the base from the side: a welded base stays, a free one slides (the goal follows)."""
    env = make(f'Stack{task}_0Distractors_Dense-v1')
    env.reset()
    for level, name in enumerate(env.tower_names):
        env._set_object_xy(name, [1.3, 0.65], level)
    env._set_object_xy('object0', [1.3, 0.95])  # out of the way
    env.base_start_xy = np.array([1.3, 0.65])
    side = site(env, env.base_name) + [0, -0.12, 0]
    infos, _ = run_stack_plan(env, [(side + [0, 0, 0.1], -1.0, 15), (side, -1.0, 15), (side + [0, 0.17, 0], -1.0, 20)])
    shift = infos[-1]['base_shift']
    if task.endswith('Fixed'):
        assert shift == 0.0 and infos[-1]['base_resting']
    else:
        assert shift > 0.03
        assert np.allclose(env.goal, site(env, env.base_name) + [0, 0, 2 * env.object_size])


def test_stack_free_base_knocked_over_continues():
    """Tipping the free base (on another face) is not a drop; a block lying on the knocked base is not placed."""
    env = make('StackOnto1_0Distractors_Dense-v1')
    env.reset()
    set_object(env, 'object1', site(env, 'object1') + [0, 0, 0.05])
    qpos = env.sim.data.get_joint_qpos('object1:joint').copy()
    qpos[3:] = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0, 0]  # 60 degrees about x: falls onto a side face
    env.sim.data.set_joint_qpos('object1:joint', qpos)
    env.sim.forward()
    for _ in range(25):
        _, _, done, info = env.step(np.zeros(4))
        assert not done and not info['dropped']
    assert abs(env.sim.data.get_site_xmat('object1')[2, 2]) < 0.05  # its old top now faces sideways
    assert info['base_resting'] and not info['success']
    assert np.allclose(env.goal, site(env, 'object1') + [0, 0, 2 * env.object_size])


@pytest.mark.parametrize('task', ['Onto2'])
def test_stack_tower_knocked_apart_continues(task):
    """The base came off the tower onto the table: not a drop, but the tower no longer stands, so object0 resting
    on the base is not placed; the goal follows the base."""
    env = make(f'Stack{task}_0Distractors_Dense-v1')
    env.reset()
    env._set_object_xy('object1', [1.3, 0.65])
    env._set_object_xy('object2', [1.3, 0.85])  # beside the tower's bottom block
    env._set_object_xy('object0', [1.3, 0.85], level=1)  # resting on the base
    for _ in range(25):
        _, _, done, info = env.step(np.zeros(4))
        assert not done and not info['dropped']
    assert info['on_base'] and 'object2' in info['object_contacts']
    assert not info['base_resting'] and not info['success']
    assert np.allclose(env.goal, site(env, 'object2') + [0, 0, 2 * env.object_size])


@pytest.mark.parametrize('task, name', [('Onto1', 'object0'), ('Onto1', 'object1'), ('Onto2', 'object1'),
                                        ('Onto2', 'object2'), ('Onto2Fixed', 'object0')])
def test_stack_dropped_block_ends_episode(task, name):
    env = make(f'Stack{task}_1Distractors_Dense-v1')
    env.reset()
    set_object(env, name, [1.3, 0.25, env.height_offset - 0.2])  # beside the table, below its top
    _, _, done, info = env.step(np.zeros(4))
    assert done and info['dropped'] and not info['success']


@pytest.mark.parametrize('task, name', [('Onto1', 'object2'), ('Onto2', 'object3')])
def test_stack_distractor_off_table_does_not_end_episode(task, name):
    env = make(f'Stack{task}_1Distractors_Dense-v1')
    env.reset()
    set_object(env, name, [1.3, 0.25, env.height_offset - 0.2])
    _, _, done, info = env.step(np.zeros(4))
    assert not done and not info['dropped']


def test_stack_instructions_are_embedded():
    from multi_object_fetch.language.instructions import INSTRUCTIONS
    env = LanguageWrapper(make('StackOnto1_0Distractors_Dense-v1'))
    for key, sentence in INSTRUCTIONS.items():
        if key[0] == 'Stack':
            assert sentence in env.token_cache and sentence in env.embed_cache, sentence


@pytest.mark.parametrize('offset, success', [((0.4, 0.4), True),  # stable over a corner region (0.57 side away)
                                             ((0.6, 0.0), False)])  # center past the edge: tips off
def test_stack_xy_tolerance(offset, success):
    env = make('StackOnto1Fixed_0Distractors_Dense-v1')
    env.reset()
    side = 2 * env.object_size
    env._set_object_xy('object1', [1.3, 0.75])  # table center: a block tipping off the base stays on the table
    env._set_object_xy('object0', [1.3, 1.0])
    base = site(env, 'object1')
    away = base + [0, -0.25, 0.3]  # the gripper starts right beside the base
    for _ in range(15):
        env.step(move_towards(env, away, 0.0))
    set_object(env, 'object0', base + [offset[0] * side, offset[1] * side, side + 0.001])
    for _ in range(25):
        _, _, done, info = env.step(move_towards(env, away, 0.0))
        check_stack_step(env, info)
    assert info['success'] == success and info['on_base'] == success and not done


@pytest.mark.parametrize('task', STACK_TASKS)
def test_stack_staged_reward(task):
    """Every stage of a scripted stack raises the staged reward; letting go on the base never lowers it."""
    env = make(f'Stack{task}_2Distractors_Staged-v1', seed=1)
    env.reset()
    obj, goal = site(env, 'object0'), env.goal.copy()
    phases = {}
    for k, (pos, gripper, steps) in enumerate(stack_plan(obj, goal)):
        for _ in range(steps):
            _, reward, done, info = env.step(move_towards(env, np.asarray(pos), gripper))
            check_stack_step(env, info)
            assert reward == pytest.approx(env.staged_reward()) and 0.0 <= reward <= 1.0 and not done
            if info['success']:
                assert reward == 1.0
            elif info['on_base']:
                assert reward == (0.8 if info['grasped'] else 0.9)
            elif info['grasped']:
                assert 0.4 <= reward < 0.8
            else:
                assert reward <= 0.2
            phases.setdefault(k, []).append(reward)
    approach, grasp, lift, carry, lower, release, retreat = (phases[k] for k in range(1, 8))
    assert approach[-1] > phases[0][0]  # closer to the block
    assert grasp[-1] == pytest.approx(0.4, abs=0.01)  # grasped, still on the table
    assert lift[-1] > 0.5  # at the stacking height, not yet over the base
    assert carry[-1] > lift[-1]  # aligned over the base
    assert lower[-1] == 0.8  # held on the base
    assert min(release + retreat) >= 0.8 and retreat[-1] == 1.0


@pytest.mark.parametrize('task', STACK_TASKS)
def test_stack_maniskill_reward(task):
    """ManiSkill StackCube-v1's normalized reward: reach <= 0.25 < grasped [0.5, 0.625] < on base [0.75, 0.875] < 1."""
    env = make(f'Stack{task}_2Distractors_ManiSkill-v1', seed=1)
    env.reset()
    phases = {}
    for k, (pos, gripper, steps) in enumerate(stack_plan(site(env, 'object0'), env.goal.copy())):
        for _ in range(steps):
            _, reward, done, info = env.step(move_towards(env, np.asarray(pos), gripper))
            check_stack_step(env, info)
            assert reward == pytest.approx(env.maniskill_reward()) and not done
            if info['success']:
                assert reward == 1.0
            elif info['on_base']:
                assert 0.75 <= reward <= 0.875
            elif info['grasped']:
                assert 0.5 <= reward <= 0.625
                assert reward == pytest.approx((5 - np.tanh(5 * info['dist'])) / 8)
            else:
                assert 0 <= reward <= 0.25
            phases.setdefault(k, []).append(reward)
    assert phases[1][-1] > phases[0][0]  # closer to the block
    assert phases[4][-1] > phases[3][0]  # carried toward the goal
    assert phases[7][-1] == 1.0


SHAPE = 'ReachMulticolorShape_{}Distractors_Dense-v1'


def site_objects(env):
    """(color index, shape index, position, bounding radius) of target0, then each distractor."""
    objects = []
    for name in ['target0'] + [f'distractor{i}' for i in range(env.num_distractors)]:
        site_id = env.sim.model.site_name2id(name)
        color = int(np.argmin(np.abs(env.colors - env.sim.model.site_rgba[site_id][:3]).sum(1)))
        shape = env.shape_geoms.index(int(env.sim.model.site_type[site_id]))
        objects.append((color, shape, site(env, name), env.site_radius(name)))
    return objects


@pytest.mark.parametrize('name', ['ReachMulticolor_2Distractors_Dense-v1', 'ReachRed_4Distractors_Dense-v1',
                                  'PickMulticolor_2Distractors_Dense-v1'])
def test_other_tasks_have_no_shape(name):
    env = make(name)
    env.reset()
    _, _, _, info = env.step(env.action_space.sample())
    assert info['target_shape'] == -1
    if name.startswith('Reach'):
        assert all(o[1] == 0 for o in site_objects(env))  # spheres only


def test_every_instruction_is_embedded():
    env = LanguageWrapper(make(SHAPE.format(1)))
    for instruction in INSTRUCTIONS.values():
        assert len(env._tokenize(instruction)) == len(env._embed(instruction)) == len(env._pieces(instruction))


def multishape_objects(env):
    """(color index, MULTISHAPE_NAMES index, position, bounding radius) of target0, then each distractor.

    Checks that exactly one of the site and its mesh geoms is visible, and that the mesh body sits on the site.
    """
    m = env.sim.model
    objects = []
    for name in ['target0'] + [f'distractor{i}' for i in range(env.num_distractors)]:
        site_id = m.site_name2id(name)
        geoms = {shape: m.geom_name2id(f'{name}:{shape}') for shape in MESH_SHAPE_NAMES}
        shown = [shape for shape, g in geoms.items() if m.geom_rgba[g][3] == 1 and m.geom_group[g] == 0]
        hidden = [shape for shape, g in geoms.items() if m.geom_rgba[g][3] == 0 and m.geom_group[g] == 3]
        assert len(shown) + len(hidden) == len(MESH_SHAPE_NAMES) and len(shown) <= 1
        if shown:
            assert m.site_rgba[site_id][3] == 0 and m.site_group[site_id] == 3
            assert env.mesh_shapes[name] == shown[0]
            rgb, shape = m.geom_rgba[geoms[shown[0]]][:3], MULTISHAPE_NAMES.index(shown[0])
        else:
            assert m.site_rgba[site_id][3] == 1 and m.site_group[site_id] == 0
            assert name not in env.mesh_shapes
            rgb, shape = m.site_rgba[site_id][:3], env.shape_geoms.index(int(m.site_type[site_id]))
        assert np.allclose(env.sim.data.get_body_xpos(f'{name}:shapes'), site(env, name))
        color = int(np.argmin(np.abs(env.colors - rgb).sum(1)))
        objects.append((color, shape, site(env, name), env.site_radius(name)))
    return objects


@pytest.mark.parametrize('num_distractors', [0, 1, 2, 4])
def test_shape_scene(num_distractors):
    env = make(SHAPE.format(num_distractors))
    target_shapes = set()
    for _ in range(40):
        env.reset()
        objects = multishape_objects(env)
        color, shape = objects[0][:2]
        assert (color, shape) == (env.target_color, env.target_shape)
        assert env.instruction == f"reach the {COLOR_NAMES[color]} {MULTISHAPE_NAMES[shape]}"
        pairs = [o[:2] for o in objects]
        assert len(set(pairs)) == len(pairs)  # the instruction names exactly one object
        if num_distractors >= 1:
            assert any(c == color and s != shape for c, s in pairs)  # same color, other shape
        if num_distractors >= 2:
            assert any(c != color and s == shape for c, s in pairs)  # same shape, other color
        for (*_, pos_a, r_a), (*_, pos_b, r_b) in combinations(objects, 2):
            assert np.linalg.norm(pos_a - pos_b) >= r_a + r_b - 1e-6  # bounding spheres do not overlap
        target_shapes.add(shape)
    assert target_shapes == set(range(len(MULTISHAPE_NAMES)))


def test_mesh_radius_matches_compiled_mesh():
    env = make(SHAPE.format(0))
    env.reset()
    m = env.sim.model
    for shape in MESH_SHAPE_NAMES:
        g = m.geom_name2id(f'target0:{shape}')
        mesh = m.geom_dataid[g]
        verts = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
        world = env.sim.data.geom_xpos[g] + verts @ env.sim.data.geom_xmat[g].reshape(3, 3).T
        # The compiler recenters the mesh; the shape is still centered on the site and fits its bounding sphere.
        assert np.linalg.norm(world - site(env, 'target0'), axis=1).max() == pytest.approx(
            env.target_size * env.mesh_radii[shape], rel=1e-4)
        assert np.ptp(world, axis=0).max() <= 2 * env.target_size * env.mesh_radii[shape]


def test_shape_info_and_language():
    env = LanguageWrapper(make(SHAPE.format(2)))
    env.action_space.seed(0)
    for _ in range(3):
        env.reset()
        _, _, _, first = env.step(env.action_space.sample())
        for _ in range(5):
            _, _, _, info = env.step(env.action_space.sample())
            check_step(info)
            assert (info['target_color'], info['target_shape']) == (first['target_color'], first['target_shape'])
        assert info['instruction'] == env.unwrapped.instruction
        assert info['instruction'].endswith(
            f"{COLOR_NAMES[info['target_color']]} {MULTISHAPE_NAMES[info['target_shape']]}")


def test_shape_scripted_success():
    env = make(SHAPE.format(4))
    assert env.spec.max_episode_steps == 100
    for _ in range(3):
        env.reset()
        for _ in range(episode_length('Reach')):
            _, _, _, info = env.step(move_towards(env, site(env, 'target0'), 0.0))
            check_step(info)
        assert info['success']


def test_mesh_shapes_are_twisted_but_still_face_the_camera():
    env = make(SHAPE.format(4))
    m, d = env.sim.model, env.sim.data
    cam_rot = d.cam_xmat[m.camera_name2id('frontview')].reshape(3, 3)
    tilts, rolls = [], []
    for _ in range(20):
        env.reset()
        for name in ['target0'] + [f'distractor{i}' for i in range(4)]:
            # Shape frame relative to the camera frame (identity when untwisted).
            rel = cam_rot.T @ d.get_body_xmat(f'{name}:shapes')
            tilts.append(np.degrees(np.arccos(np.clip(rel[2, 2], -1, 1))))  # outline normal vs view axis
            rolls.append(np.degrees(np.arctan2(rel[1, 0], rel[0, 0])))
    tilts, rolls = np.array(tilts), np.array(rolls)
    max_tilt = np.degrees(np.hypot(env.MESH_MAX_TILT, env.MESH_MAX_TILT))
    assert tilts.max() <= max_tilt + 1e-6 and tilts.max() > 5  # twisted, but never far from facing the camera
    assert np.abs(rolls).max() <= np.degrees(env.MESH_MAX_ROLL) + 10 and np.std(rolls) > 5
    # Twists are drawn per episode and site, not shared.
    assert len(np.unique(np.round(rolls, 3))) == len(rolls)


def test_cubes_are_twisted():
    env = make(SHAPE.format(4))
    m, d = env.sim.model, env.sim.data
    yaws, tilts = [], []
    for _ in range(30):
        env.reset()
        for name in ['target0'] + [f'distractor{i}' for i in range(4)]:
            site_id = m.site_name2id(name)
            rot = d.site_xmat[site_id].reshape(3, 3)
            if name in env.mesh_shapes or m.site_type[site_id] != const.GEOM_BOX:
                assert np.allclose(m.site_quat[site_id], [1, 0, 0, 0])  # only cubes are twisted
                continue
            tilts.append(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))  # cube's z vs world up
            yaws.append(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    assert len(yaws) > 5
    max_tilt = np.degrees(np.hypot(env.CUBE_MAX_TILT, env.CUBE_MAX_TILT))
    assert max(tilts) <= max_tilt + 1e-6 and max(tilts) > 5
    assert np.std(yaws) > 10 and len(np.unique(np.round(yaws, 3))) == len(yaws)


def test_other_reach_tasks_have_no_mesh_shapes():
    env = make('ReachMulticolor_2Distractors_Dense-v1')
    env.reset()
    assert env.mesh_shapes == {} and env.sim.model.nmesh == make(SHAPE.format(2)).sim.model.nmesh - 3


HELDOUT = {(COLOR_NAMES.index(c), MULTISHAPE_NAMES.index(sh)) for c, sh in MULTICOLORSHAPE_HELDOUT}


def test_heldout_split_covers_every_color_and_shape():
    assert len(HELDOUT) == len(MULTICOLORSHAPE_HELDOUT) == 4 * len(MULTISHAPE_NAMES)
    train = {(c, sh) for c in range(len(COLOR_NAMES)) for sh in range(len(MULTISHAPE_NAMES))} - HELDOUT
    for sh in range(len(MULTISHAPE_NAMES)):
        assert sum(1 for p in train if p[1] == sh) == len(COLOR_NAMES) - 4
    for c in range(len(COLOR_NAMES)):
        assert sum(1 for p in train if p[0] == c) >= 3


@pytest.mark.parametrize('split', ['Train', 'Val'])
@pytest.mark.parametrize('num_distractors', [1, 4])
def test_shape_split(split, num_distractors):
    env = make(f'ReachMulticolorShape_{num_distractors}Distractors_Dense_{split}-v1')
    assert env.split == split.lower()
    for _ in range(40):
        env.reset()
        objects = multishape_objects(env)
        pairs = [o[:2] for o in objects]
        color, shape = pairs[0]
        assert ((color, shape) in HELDOUT) == (split == 'Val')
        if split == 'Train':
            assert not set(pairs) & HELDOUT  # held-out pairs never appear in training, not even as distractors
        assert len(set(pairs)) == len(pairs)
        assert any(c == color and sh != shape for c, sh in pairs)  # same color, other shape
        if num_distractors >= 2:
            assert any(c != color and sh == shape for c, sh in pairs)  # same shape, other color
        assert env.instruction == f"reach the {COLOR_NAMES[color]} {MULTISHAPE_NAMES[shape]}"


def test_shape_custom_heldout_pairs():
    heldout = [('red', 'star'), ('blue', 'moon')]
    for split, check in [('val', lambda t, ps: t in heldout), ('train', lambda t, ps: not set(ps) & set(heldout))]:
        env = gym.make('ReachMulticolorShape_2Distractors_Dense-v1', split=split, heldout_pairs=heldout).unwrapped
        env.seed(0)
        for _ in range(20):
            env.reset()
            names = [(COLOR_NAMES[c], MULTISHAPE_NAMES[sh]) for c, sh, *_ in multishape_objects(env)]
            assert check(names[0], names)


def test_shape_split_errors():
    with pytest.raises(ValueError):
        gym.make('ReachMulticolorShape_2Distractors_Dense-v1', split='test')
    with pytest.raises(ValueError):
        gym.make('ReachMulticolor_2Distractors_Dense-v1', split='train')
    # Holding out every red pair but one leaves no same-color distractor for a red target in training.
    env = gym.make('ReachMulticolorShape_1Distractors_Dense-v1', split='train',
                   heldout_pairs=[('red', sh) for sh in MULTISHAPE_NAMES[1:]]).unwrapped
    env.target_pairs = [(COLOR_NAMES.index('red'), 0)]
    with pytest.raises(ValueError, match='same-color'):
        env.reset()
