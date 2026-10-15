"""Tests for the per-step metric values that MultiObjectFetchEnv writes into `info` (see docs/metrics.md)."""
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

import multi_object_fetch.env  # noqa: E402  # also registers the gym envs
from multi_object_fetch.language import LanguageWrapper
from multi_object_fetch.language.instructions import COLOR_NAMES


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
        assert info['grasped'] == FINGERS.issubset(contacts)
        assert info['lifted'] == (bool(contacts & FINGERS) and not (contacts - FINGERS)
                                  and info['object_lift'] > env_cls.LIFT_THRESHOLD)
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
        for key in ('lifted', 'held', 'grasped', 'grasped_geometric', 'placed', 'object_contacts',
                    'finger_gap', 'object_width', 'object_speed', 'goal_needs_lift'):
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
