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

    def _predict(self, observations, deterministic=False):
        actions = [self._next_action(env) for env in self.envs]
        return actions

    def predict(self, observations, deterministic=False):
        return self._predict(observations, deterministic)

    def reset(self, envs):
        self.envs = envs
        self.history = deque(maxlen=10)
        self.should_hold = False

    def _next_action(self, env):

        robot_body = env.robot.robot_body
        robot_pos = robot_body.position
        robot_angle = robot_body.angle

        robot_vector = robot_body.rotation_vector.rotated_degrees(90)

        target_pos = env.target_shape.shape_body.position
        target_dist = target_pos.get_distance(robot_pos)
        target_vector = (target_pos - robot_pos).normalized()
        diff_angle = robot_vector.get_angle_between(target_vector)

        if not self.should_hold and target_dist < 0.4 and len(self.history) >= 3:
            l = len(self.history)
            for i in range(l - 3, l - 1):
                d_now = self.history[i]['dist_to_target']
                d_next = self.history[i + 1]['dist_to_target']
                if abs(d_next - d_now) > 0.01:
                    self.should_hold = True

        act_flags = [RA.NONE, RA.NONE, RA.OPEN]

        if self.should_hold:
            act_flags[2] = RA.CLOSE
            target_pos = pm.vec2d.Vec2d((-1., 1.))
            target_dist = target_pos.get_distance(env.target_shape.shape_body.position)
            target_vector = (target_pos - robot_pos).normalized()
            diff_angle = robot_vector.get_angle_between(target_vector)

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
            origin = pm.vec2d.Vec2d((0, 0))
            target_vector = (origin - robot_pos).normalized()
            diff_angle = (-robot_vector).get_angle_between(target_vector)
            if diff_angle < 0 and diff_angle >= -math.pi:
                act_flags[1] = RA.RIGHT
            else:
                act_flags[1] = RA.LEFT


        self.history.append({
            'action': act_flags,
            'robot_pos': robot_pos,
            'robot_angle': robot_angle,
            'dist_to_target': target_dist
        })

        action = env.flags_to_action(act_flags)

        #input()

        return action



