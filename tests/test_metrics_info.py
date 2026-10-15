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

import multi_object_fetch  # noqa: E402,F401  # gym registrations
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


def check_step(info):
    assert info['success'] == (info['dist'] < env_threshold())
    assert info['unstable'] is False


def env_threshold():
    return multi_object_fetch.env.MultiObjectFetchEnv.SUCCESS_THRESHOLD


@pytest.mark.parametrize('name', ['ReachMulticolor_2Distractors_Dense-v1', 'PushMulticolor_3Distractors_Dense-v1',
                                  'PickMulticolor_4Distractors_Sparse-v1', 'PickRed_1Distractors_Dense-v1',
                                  'ReachOdd_2Distractors_Dense-v1', 'PushOddGroups_4Distractors_Dense-v1'])
def test_info_keys_and_constants(name):
    env = LanguageWrapper(make(name))
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


def test_pick_scripted_lift_and_hold():
    env = make('PickRed_0Distractors_Dense-v1', seed=1)
    lifted_episodes = 0
    for _ in range(3):
        env.reset()
        u = env.unwrapped
        obj = u.sim.data.get_site_xpos('object0').copy()
        goal = u.goal.copy()
        plan = [(obj + [0, 0, 0.1], 1.0, 15), (obj, 1.0, 15), (obj, -1.0, 8),
                (obj + [0, 0, 0.15], -1.0, 15), (goal, -1.0, 40)]
        infos = []
        for pos, gripper, steps in plan:
            for _ in range(steps):
                _, _, _, info = env.step(move_towards(env, pos, gripper))
                check_step(info)
                infos.append(info)
        lifts = np.array([i['object_lift'] for i in infos])
        grips = np.array([i['grip_object_dist'] for i in infos])
        held = (lifts > 0.02) & (grips < 0.03)
        assert lifts[0] == pytest.approx(0.0, abs=0.005)  # block starts resting on the table
        lifted_episodes += held.any()
        print('max lift %.3f, held %s, final dist %.3f, goal height %.3f, success %s' % (
            lifts.max(), held.any(), infos[-1]['dist'], infos[-1]['goal_height'], infos[-1]['success']))
    assert lifted_episodes >= 2


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
    assert 'dist' not in info and 'object_lift' not in info and 'grip_object_dist' not in info
    assert info['num_distractors'] == 2
    assert 0 <= info['target_color'] < len(COLOR_NAMES)
    assert ('goal_height' in info) == name.startswith('Pick')
