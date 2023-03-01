import math
import warnings
from collections import OrderedDict

import pymunk as pm

from gym.spaces import Box, Dict, Discrete
from gym.utils import EzPickle
import numpy as np

from magical import geom
from magical.base_env import BaseEnv, ez_init
import magical.entities as en


class PickAndPlaceEnv(BaseEnv, EzPickle):

    NUM_OBJECTS = 1

    @ez_init()
    def __init__(self,
                 rand_shape_colour=False,
                 rand_shape_type=False,
                 rand_poses=False,
                 debug_reward=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.rand_shape_colour = rand_shape_colour
        self.rand_shape_type = rand_shape_type
        self.rand_poses = rand_poses
        self.debug_reward = debug_reward
        self.observation_space['target_type'] = Box(low=0, high=en.SHAPE_TYPES.shape[0], shape=(1,))
        self.observation_space['target_colour'] = Box(low=0, high=en.SHAPE_COLOURS.shape[0], shape=(1,))
        self.observation_space['object_position'] = Box(low=-2, high=2, shape=(2,))
        self.observation_space['goal_position'] = Box(low=-2, high=2, shape=(2,))

    def on_reset(self):
        # make the robot
        robot_pos = np.asarray((0., 0.))
        robot_angle = 0.55 * math.pi
        robot = self._make_robot(robot_pos, robot_angle)
        self.add_entities([robot])

        self.shapes = []
        for _ in range(self.NUM_OBJECTS):
            shape_pos = np.asarray((0.1, -0.65))
            shape_angle = 0.13 * math.pi
            shape_colour = 'red'
            shape_type = en.ShapeType.SQUARE
            if self.rand_shape_colour:
                shape_colour = self.rng.choice(
                    np.asarray(en.SHAPE_COLOURS, dtype='object'))
            if self.rand_shape_type:
                shape_type = self.rng.choice(
                    np.asarray(en.SHAPE_TYPES, dtype='object'))

            shape = self._make_shape(
                shape_type=shape_type,
                colour_name=shape_colour,
                init_pos=shape_pos,
                init_angle=shape_angle)

            self.shapes.append(shape)

        self.add_entities(self.shapes)
        target_shape = self.shapes[0]
        self.target_type = target_shape.shape_type
        self.target_colour = target_shape.colour_name
        self.target_type_id = np.asarray([en.SHAPE_TYPES.tolist().index(self.target_type)])
        self.target_colour_id = np.asarray([en.SHAPE_COLOURS.tolist().index(self.target_colour)])
        if self.config.one_goal:
            self.goal_position = np.array([-1., 1.])
        else:
            self.goal_position = self.rng.rand(2) * 2 - 1

        self.valid_target_shapes = []
        for shape in self.shapes:
            if shape.shape_type == self.target_type and shape.colour_name == self.target_colour:
                self.valid_target_shapes.append(shape)
        self.target_shape = self.rng.choice(self.valid_target_shapes)

        if self.rand_poses:
            geom.pm_randomise_all_poses(
                self._space, (self._robot, *self.shapes),
                self.ARENA_BOUNDS_LRBT,
                self.rng,
                rand_pos=True,
                rand_rot=True)
                #rel_pos_linf_limits=self.JITTER_POS_BOUND,
                #rel_rot_limits=self.JITTER_ROT_BOUND)

    def _get_direction_to_target_object(self):
        cur_pos = self.robot.robot_body.position
        obj_pos = self.target_shape.shape_body.position
        dir = np.asarray(obj_pos - cur_pos)
        return dir

    def _get_direction_to_goal(self):
        cur_pos = self.robot.robot_body.position
        goal_pos = pm.vec2d.Vec2d(*self.goal_position.tolist())
        dir = np.asarray(goal_pos - cur_pos)
        return dir

    def _add_more_info(self, obs):
        obs['target_type'] = self.target_type_id
        obs['target_colour'] = self.target_colour_id
        obs['object_position'] = np.asarray(self.target_shape.shape_body.position)
        obs['goal_position'] = self.goal_position

    def reset(self):
        obs = super().reset()
        self._add_more_info(obs)
        return obs

    def score_on_end_of_traj(self):
        best_score = -1e9
        for shape in self.valid_target_shapes:
            target_shape = self.target_shape
            shape_pos = np.asarray(target_shape.shape_body.position)
            # target is top left
            dist = np.linalg.norm(self.goal_position - shape_pos)
            #succeed_dist = np.sqrt(2) / 2
            succeed_dist = self.SHAPE_RAD
            furthest_dist = np.sqrt(2)
            drange = (furthest_dist - succeed_dist)
            score = min(1.0, max(0.0, furthest_dist - dist) / drange)
            assert 0 <= score <= 1
            best_score = max(best_score, score)
        return best_score

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        if self.debug_reward:
            # dense reward for training RL
            rew = self.debug_shaped_reward()
        self._add_more_info(obs)
        #self.render(mode='human')
        return obs, rew, done, info

    def debug_shaped_reward(self):
        shape_pos = np.asarray(self.target_shape.shape_body.position)
        # shape_pos[0] is meant to be 0, shape_pos[1] is meant to be 1
        shape_to_goal_dist = np.linalg.norm(shape_pos - self.goal_position)
        # encourage the robot to get close to the shape, and the shape to get
        # close to the goal
        robot_pos = np.asarray(self.robot.robot_body.position)
        robot_to_shape_dist = np.linalg.norm(robot_pos - shape_pos)
        """
        shaping = -shape_to_target_dist / 5 \
            - max(robot_to_shape_dist, self.SHAPE_RAD) / 10
        return shaping + self.score_on_end_of_traj()
        """

        robot_to_goal_dist = np.linalg.norm(robot_pos - self.goal_position)

        if self.config.task == 'go_to_goal':
            reward = -robot_to_goal_dist / 20
        elif self.config.task == 'pick':
            reward = -robot_to_shape_dist / 20
        else:
            reward = -shape_to_goal_dist / 5 - max(robot_to_shape_dist, 0.2) / 20

        return reward
