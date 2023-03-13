import os
import sys
import math

from collections import deque
import numpy as np
import pymunk as pm
import torch


from magical.experts.base import BaseExpert
from magical.entities import RobotAction as RA


class MoveToCornerExpert(BaseExpert):

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
        self.should_close = [False] * venv.num_envs

    def _next_action(self, id):

        state = self.env_method(id, 'get_state')
        robot_pos = state.robot.position
        robot_angle = state.robot.angle
        robot_rotation_vector = state.robot.rotation_vector
        shape_pos = state.shape.position
        shape_angle = state.shape.angle
        robot_vector = robot_rotation_vector.rotated_degrees(90)

        history = self.history[id]

        robot_to_shape_dist = shape_pos.get_distance(robot_pos)
        to_shape_vector = (shape_pos - robot_pos).normalized()

        if not self.should_close[id] and robot_to_shape_dist < 0.4 and len(history) >= 3:
            l = len(history)
            for i in range(l - 3, l - 1):
                d_now = history[i]['robot_to_shape_dist']
                d_next = history[i + 1]['robot_to_shape_dist']
                if abs(d_next - d_now) > 0.01:
                    self.should_close[id] = True

        act_flags = [RA.NONE, RA.NONE, RA.OPEN]

        if self.should_close[id]:
            act_flags[2] = RA.CLOSE
            goal_pos = pm.vec2d.Vec2d((-1., 1.))
            shape_to_goal_dist = goal_pos.get_distance(shape_pos)
            to_goal_vector = (goal_pos - robot_pos).normalized()
            diff_angle = robot_vector.get_angle_between(to_goal_vector)
            target_dist = shape_to_goal_dist
        else:
            diff_angle = robot_vector.get_angle_between(to_shape_vector)
            target_dist = robot_to_shape_dist

        angle_eps = math.pi / 20

        if abs(diff_angle) < angle_eps:
            act_flags[0] = RA.UP
        elif diff_angle < 0 and diff_angle >= -math.pi:
            act_flags[1] = RA.RIGHT
        else:
            act_flags[1] = RA.LEFT

        cnt_down = 0
        i = len(history) - 1
        while i >= 0 and history[i]['action'][0] == RA.DOWN:
            cnt_down += 1
            i -= 1

        should_back_up = False

        # continue back up
        if cnt_down > 0 and cnt_down < 10:
            should_back_up = True

        cnt_unmoved = 0
        i = len(history) - 1
        while i > 0:
            p1 = history[i]['robot_pos']
            p2 = history[i - 1]['robot_pos']
            dp = p1.get_distance(p2)
            a1 = history[i]['robot_angle']
            a2 = history[i - 1]['robot_angle']
            da = abs(a1 - a2)
            if dp > 0.01 or da > 0.01:
                break
            cnt_unmoved += 1
            i -= 1

        if target_dist >= 0.3 and cnt_unmoved >= 3:
            should_back_up = True

        if should_back_up:
            act_flags[0] = RA.DOWN
            center = pm.vec2d.Vec2d((0, 0))
            to_center_vector = (center - robot_pos).normalized()
            diff_angle = (-robot_vector).get_angle_between(to_center_vector)
            if diff_angle < 0 and diff_angle >= -math.pi:
                act_flags[1] = RA.RIGHT
            else:
                act_flags[1] = RA.LEFT

        history.append({
            'action': act_flags,
            'robot_pos': robot_pos,
            'robot_angle': robot_angle,
            'robot_to_shape_dist': robot_to_shape_dist
        })

        action = self.env_method(id, 'flags_to_action', act_flags)

        #input()

        return action



