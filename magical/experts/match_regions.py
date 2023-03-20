import os
import sys
import math

from collections import deque
import numpy as np
import pymunk as pm
import torch

from magical.base_env import BaseEnv
from magical.experts.base import BaseExpert, PickAndPlaceExpert
from magical.entities import RobotAction as RA


class MatchRegionsExpert(BaseExpert):

    def _predict(self, observations, not_used_deterministic=None):
        actions = [self._next_action(i) for i in range(len(observations))]
        return actions

    def predict(self, observations, not_used_deterministic=None):
        return self._predict(observations)

    def forward(self, observations, not_used_deterministic=None):
        return self._predict(observations)

    def reset(self, venv):
        self.venv = venv
        self.history = [deque(maxlen=10) for _ in range(venv.num_envs)]
        self.pick_and_place_expert = [None] * venv.num_envs
        self.picked_shape = (None, None)

    def _is_inside(self, sensor, shape):
        sensor_x = sensor.position.x - sensor.w / 2
        sensor_y = sensor.position.y + sensor.h / 2
        return sensor_x + BaseEnv.SHAPE_RAD <= shape.position.x <= sensor_x + sensor.w - BaseEnv.SHAPE_RAD and \
               sensor_y - sensor.h + BaseEnv.SHAPE_RAD <= shape.position.y <= sensor_y - BaseEnv.SHAPE_RAD

    def _can_put(self, point, rad):
        r = BaseEnv.SHAPE_RAD
        return -1 + r <= point.x <= 1 - r and -1 + r <= point.y <= 1 - r

    def _next_action(self, id):
        state = self.env_method(id, 'get_state')
        robot = state.robot
        target_shapes = state.target_shapes
        distractor_shapes = state.distractor_shapes
        sensor = state.sensor

        expert = self.pick_and_place_expert[id]

        if expert is None:
            self.picked_shape = (None, None)
            for i, shape in enumerate(distractor_shapes):
                if self._is_inside(sensor, shape):
                    self.picked_shape = ('distractor', i)
                    break

            if self.picked_shape[0] is None:
                for i, shape in enumerate(target_shapes):
                    if not self._is_inside(sensor, shape):
                        self.picked_shape = ('target', i)
                        break
                if self.picked_shape[0] is None:
                    return 0

                # check if a distractor is hindering the path to a target
                target_shape = target_shapes[self.
                hindering_distractor_shape = None
                robot_to_target_vector = (target_shape.position - robot.position).normalized()
                robot_to_target_dist = target_shape.position.get_distance(robot.position)
                for i, shape in enumerate(distractor_shapes):
                    projected_distractor = shape.position.projection(robot_to_target_vector)
                    projected_distractor_to_distractor_dist = shape.get_distance(projected_distractor)
                    robot_to_distractor_dist = shape.position.get_distance(robot.position)
                    if projected_distractor_to_distractor_dist < BaseEnv.SHAPE_RAD and \
                       robot_to_distractor_dist < robot_to_target_dist:


                goal_pos = sensor.position
            else:
                # choose where to put distractor
                r = BaseEnv.SHAPE_RAD
                points = [ pm.vec2d.Vec2d(sensor.position.x - sensor.w / 2 - r, sensor.position.y - sensor.h / 2 - r),
                           pm.vec2d.Vec2d(sensor.position.x - sensor.w / 2 - r, sensor.position.y + sensor.h / 2 + r),
                           pm.vec2d.Vec2d(sensor.position.x + sensor.w / 2 + r, sensor.position.y - sensor.h / 2 - r),
                           pm.vec2d.Vec2d(sensor.position.x + sensor.w / 2 + r, sensor.position.y + sensor.h / 2 + r)]
                best_p = (None, None)
                distractor_shape = distractor_shapes[self.picked_shape[1]]
                for p in points:
                    if self._can_put(p):
                        d = p.get_distance(distractor_shape)
                        if d < best_corner[0]:
                            best_p = (d, p)
                assert best_p[0] is not None
                goal_pos = best_p[1]
            expert = PickAndPlaceExpert(goal_pos)

        if self.picked_shape[0] == 'target':
            shape = target_shapes[self.picked_shape[1]]
        else:
            assert self.picked_shape[0] == 'distractor'
            shape = distractor_shapes[self.picked_shape[1]]

        act_flags, done = expert.predict(robot, shape)
        self.pick_and_place_expert[id] = None if done else expert

        action = self.env_method(id, 'flags_to_action', act_flags)
        input()
        return action



