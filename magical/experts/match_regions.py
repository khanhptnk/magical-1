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
        self.picked_shape = [None] * venv.num_envs

    def _is_inside(self, sensor, point):
        sensor_x = sensor.position.x - sensor.w / 2
        sensor_y = sensor.position.y + sensor.h / 2
        return sensor_x + BaseEnv.SHAPE_RAD <= point.x <= sensor_x + sensor.w - BaseEnv.SHAPE_RAD and \
               sensor_y - sensor.h + BaseEnv.SHAPE_RAD <= point.y <= sensor_y - BaseEnv.SHAPE_RAD

    def _can_put(self, point, rad):
        r = BaseEnv.SHAPE_RAD
        return -1 + r <= point.x <= 1 - r and -1 + r <= point.y <= 1 - r

    def _find_obstacle(self, robot, target, candidate_obstables):
        robot_to_target_vector = (target.position - robot.position).normalized()
        robot_to_target_dist = target.position.get_distance(robot.position)
        for i, s in enumerate(candidate_obstables):
            shape_pos = s.position - robot.position
            project = shape_pos.projection(robot_to_target_vector)
            shape_to_project_dist = project.get_distance(shape_pos)

            robot_to_shape_vector = shape_pos.normalized()
            robot_to_project_dist = project.length

            angle_diff = robot_to_target_vector.get_angle_between(robot_to_shape_vector)
            #print(s)
            #print(angle_diff, robot_to_project_dist, robot_to_target_dist, shape_to_project_dist, BaseEnv.SHAPE_RAD)
            #print()
            if abs(angle_diff) < math.pi / 2 and robot_to_project_dist < robot_to_target_dist and \
               shape_to_project_dist < BaseEnv.SHAPE_RAD * 2.5:
                return s, i
        return None, None


    def _next_action(self, id):
        state = self.env_method(id, 'get_state')
        robot = state.robot
        target_shapes = state.target_shapes
        distractor_shapes = state.distractor_shapes
        sensor = state.sensor

        expert = self.pick_and_place_expert[id]
        picked_shape = self.picked_shape[id]

        should_stop = True
        for i, shape in enumerate(target_shapes):
            if not self._is_inside(sensor, shape.position):
                should_stop = False
        if should_stop:
            return 0

        if expert is None:
            picked_shape = None
            for i, shape in enumerate(distractor_shapes):
                if self._is_inside(sensor, shape.position):
                    picked_shape = ('distractor', i)
                    break

            if picked_shape is None:
                for i, shape in enumerate(target_shapes):
                    if not self._is_inside(sensor, shape.position):
                        picked_shape = ('target', i)
                        break
                if picked_shape is None:
                    return 0

                target_shape = target_shapes[picked_shape[1]]
                obstacle, obstacle_id = self._find_obstacle(robot, target_shape, distractor_shapes)
                if obstacle is not None:
                    # move the obstacle to the robot's current position
                    picked_shape = ('distractor', obstacle_id)
                    goal_pos = robot.position
                else:
                    # no obstacle, move the target shape inside goal region
                    goal_pos = sensor.position
            else:
                # choose where to put distractor
                r = BaseEnv.SHAPE_RAD
                points = [ pm.vec2d.Vec2d(sensor.position.x - sensor.w / 2 - r, sensor.position.y - sensor.h / 2 - r),
                           pm.vec2d.Vec2d(sensor.position.x - sensor.w / 2 - r, sensor.position.y + sensor.h / 2 + r),
                           pm.vec2d.Vec2d(sensor.position.x + sensor.w / 2 + r, sensor.position.y - sensor.h / 2 - r),
                           pm.vec2d.Vec2d(sensor.position.x + sensor.w / 2 + r, sensor.position.y + sensor.h / 2 + r)]
                best_p = (1e9, None)
                distractor_shape = distractor_shapes[picked_shape[1]]
                for p in points:
                    if self._can_put(p, r):
                        d = p.get_distance(distractor_shape.position)
                        if d < best_p[0]:
                            best_p = (d, p)
                assert best_p[0] is not None
                goal_pos = best_p[1]
            expert = PickAndPlaceExpert(goal_pos, dist_eps=0.15)

        if picked_shape[0] == 'target':
            shape = target_shapes[picked_shape[1]]
        else:
            assert picked_shape[0] == 'distractor'
            shape = distractor_shapes[picked_shape[1]]

        act_flags, done = expert.predict(robot, shape)
        if done:
            self.pick_and_place_expert[id] = None
            self.picked_shape[id] = None
        else:
            self.pick_and_place_expert[id] = expert
            self.picked_shape[id] = picked_shape

        action = self.env_method(id, 'flags_to_action', act_flags)
        #input()
        return action



