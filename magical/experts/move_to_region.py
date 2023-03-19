import os
import sys
import math

from collections import deque
import numpy as np
import pymunk as pm
import torch


from magical.experts.base import BaseExpert
from magical.entities import RobotAction as RA


class MoveToRegionExpert(BaseExpert):

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

    def _next_action(self, id):

        state = self.env_method(id, 'get_state')
        robot_pos = state.robot.position
        robot_angle = state.robot.angle
        robot_rotation_vector = state.robot.rotation_vector
        robot_vector = robot_rotation_vector.rotated_degrees(90)
        goal_pos = state.goal.position

        history = self.history[id]

        robot_to_goal_dist = goal_pos.get_distance(robot_pos)
        to_goal_vector = (goal_pos - robot_pos).normalized()

        act_flags = [RA.NONE, RA.NONE, RA.OPEN]

        diff_angle = robot_vector.get_angle_between(to_goal_vector)
        target_dist = robot_to_goal_dist

        if target_dist > 0.05:

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
            'robot_to_goal_dist': robot_to_goal_dist
        })

        action = self.env_method(id, 'flags_to_action', act_flags)

        #input()

        return action



