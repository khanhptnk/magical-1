"""Tools for saving and loading trajectories without requiring the `imitation`
or `tensorflow` packages to be installed."""
import gzip
from pickle import Unpickler
from typing import List, NamedTuple, Optional

import gym
import numpy as np

from milbench.benchmarks import DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS


class MILBenchTrajectory(NamedTuple):
    """Duplicate of `imitation.util.rollout.Trajectory`. It's here so that I
    can unpickle trajectories semi-faithfully without needing to have
    `imitation` installed."""

    acts: np.ndarray
    obs: np.ndarray
    rews: np.ndarray
    infos: Optional[List[dict]]


class _TrajRewriteUnpickler(Unpickler):
    """Custom unpickler that replaces references to `Trajectory` class in
    `imitation` with custom trajectory class in this module."""
    def find_class(self, module, name):
        # print('find_class(%r, %r)' % (module, name))
        if (module, name) == ('imitation.util.rollout', 'Trajectory'):
            return MILBenchTrajectory
        return super().find_class(module, name)


def load_demos(demo_paths, rewrite_traj_cls=True):
    """Use GzipFile & pickle to load a list of demo dictionaries from a series
    of file paths."""
    demo_dicts = []
    n_demos = len(demo_paths)
    for d_num, d_path in enumerate(demo_paths, start=1):
        print(f"Loading '{d_path}' ({d_num}/{n_demos})")
        with gzip.GzipFile(d_path, 'rb') as fp:
            if rewrite_traj_cls:
                unpickler = _TrajRewriteUnpickler(fp)
            else:
                unpickler = Unpickler(fp)
            demo_dicts.append(unpickler.load())
    return demo_dicts


def splice_in_preproc_name(base_env_name, preproc_name):
    """Splice the name of a preprocessor into a milbench benchmark name. e.g.
    you might start with "MoveToCorner-Demo-v0" and insert "LoResStack" to end
    up with "MoveToCorner-Demo-LoResStack-v0". Will do a sanity check to ensure
    that the preprocessor actually exists."""
    prefix, version = base_env_name.rsplit('-', 1)
    assert preproc_name in DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS, \
        f"no preprocessor named '{preproc_name}', options are " \
        f"{', '.join(DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS)}"
    env_name = f'{prefix}-{preproc_name}-{version}'
    return env_name


class _MockDemoEnv(gym.Wrapper):
    """Mock Gym environment that just returns an observation"""
    def __init__(self, orig_env, trajectory):
        super().__init__(orig_env)
        self._idx = 0
        self._traj = trajectory
        self._traj_length = len(self._traj.acts)

    def reset(self):
        self._idx = 0
        return self._traj.obs[self._idx]

    def step(self, action):
        rew = self._traj.rews[self._idx]
        info = self._traj.infos[self._idx] or {}
        info['_mock_demo_act'] = self._traj.acts[self._idx]
        self._idx += 1
        # ignore action, return next obs
        obs = self._traj.obs[self._idx]
        # it's okay if we run one over the end
        done = self._idx >= self._traj_length
        return obs, rew, done, info


def preprocess_demos_with_wrapper(trajectories, orig_env_name, preproc_name):
    """Preprocess trajectories using one of the built-in environment
    preprocessing pipelines.

    Args:
        trajectories ([Trajectory]): list of trajectories to process.
        orig_env_name (str): name of original environment where trajectories
            were collected. This function will instantiate a temporary instance
            of that environment to get access to an observation space and other
            metadata.
        preproc_name (str): name of preprocessor to apply. Should be available
            in `milbench.benchmarks.DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS`.

    Returns:
        rv_trajectories ([Trajectory]): equivalent list of trajectories that
            have each been preprocessed with the given wrapper."""
    wrapper = DEFAULT_PREPROC_ENTRY_POINT_WRAPPERS[preproc_name]
    orig_env = gym.make(orig_env_name)
    wrapped_constructor = wrapper(_MockDemoEnv)
    rv_trajectories = []
    for traj in trajectories:
        mock_env = wrapped_constructor(orig_env=orig_env, trajectory=traj)
        obs = mock_env.reset()
        values = {
            'obs': [],
            'acts': [],
            'rews': [],
            'infos': [],
        }
        values['obs'].append(obs)
        done = False
        while not done:
            obs, rew, done, info = mock_env.step(None)
            acts = info['_mock_demo_act']
            del info['_mock_demo_act']
            values['obs'].append(obs)
            values['acts'].append(acts)
            values['rews'].append(rew)
            values['infos'].append(info)
        # turn obs, acts, and rews into numpy arrays
        stack_values = {
            k: np.stack(vs, axis=0)
            for k, vs in values.items() if k in ['obs', 'acts', 'rews']
        }
        # keep infos as a list (hard to get at elements otherwise)
        stack_values['infos'] = values.get('infos')
        # use type(traj) to preserve either Trajectory namedtuple type (from
        # imitation module), or MILBenchTrajectory type (from milbench module)
        new_traj = type(traj)(**stack_values)
        rv_trajectories.append(new_traj)
    return rv_trajectories
