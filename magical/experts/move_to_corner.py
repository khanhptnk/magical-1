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
        self.history = deque(maxlen=10)
        self.should_hold = False

    def _next_action(self, id):

        env_state = self.get_env_attr(id, 'get_state')

        robot_pos = env_state['robot']['pos']
        robot_angle = env_state['robot']['angle']
        robot_rotation_vector = env_state['robot']['rotation_vector']

        shape_pos = env_state['shape']['pos']
        shape_angle = env_state['shape']['angle']

        robot_vector = robot_rotation_vector.rotated_degrees(90)

        robot_to_shape_dist = shape_pos.get_distance(robot_pos)
        to_shape_vector = (shape_pos - robot_pos).normalized()
        diff_angle = robot_vector.get_angle_between(to_shape_vector)

        if not self.should_hold and robot_to_shape_dist < 0.4 and len(self.history) >= 3:
            l = len(self.history)
            for i in range(l - 3, l - 1):
                d_now = self.history[i]['robot_to_shape_dist']
                d_next = self.history[i + 1]['robot_to_shape_dist']
                if abs(d_next - d_now) > 0.01:
                    self.should_hold = True

        act_flags = [RA.NONE, RA.NONE, RA.OPEN]

        if self.should_hold:
            act_flags[2] = RA.CLOSE
            goal_pos = pm.vec2d.Vec2d((-1., 1.))
            shape_to_goal_dist = goal_pos.get_distance(shape_pos)
            to_goal_vector = (goal_pos - robot_pos).normalized()
            diff_angle = robot_vector.get_angle_between(to_goal_vector)
            target_dist = shape_to_goal_dist
        else:
            target_dist = robot_to_shape_dist

        angle_eps = math.pi / 20

        if abs(diff_angle) < angle_eps:
            act_flags[0] = RA.UP
        elif diff_angle < 0 and diff_angle >= -math.pi:
            act_flags[1] = RA.RIGHT
        else:
            act_flags[1] = RA.LEFT

        cnt_down = 0
        i = len(self.history) - 1
        while i >= 0 and self.history[i]['action'][0] == RA.DOWN:
            cnt_down += 1
            i -= 1

        should_back_up = False

        # continue back up
        if cnt_down > 0 and cnt_down < 10:
            should_back_up = True

        cnt_unmoved = 0
        i = len(self.history) - 1
        while i > 0:
            p1 = self.history[i]['robot_pos']
            p2 = self.history[i - 1]['robot_pos']
            dp = p1.get_distance(p2)
            a1 = self.history[i]['robot_angle']
            a2 = self.history[i - 1]['robot_angle']
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
            target_vector = (center - robot_pos).normalized()
            diff_angle = (-robot_vector).get_angle_between(target_vector)
            if diff_angle < 0 and diff_angle >= -math.pi:
                act_flags[1] = RA.RIGHT
            else:
                act_flags[1] = RA.LEFT


        self.history.append({
            'action': act_flags,
            'robot_pos': robot_pos,
            'robot_angle': robot_angle,
            'robot_to_shape_dist': robot_to_shape_dist
        })

        action = self.env_method(id, 'flags_to_action', act_flags)

        #input()

        return action



