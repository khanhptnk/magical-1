import os
import sys
import math
from collections import deque
import pymunk as pm

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from magical.entities import RobotAction as RA


class BaseExpert(BasePolicy):

    def __init__(self, observation_space,
                       action_space,
                       not_used_features_extractor_class):

        super().__init__(observation_space,
                         action_space,
                         not_used_features_extractor_class)

    def get_env_attr(self, id, attr_name):
        return self.venv.get_attr(attr_name, indices=[id])[0]

    def env_method(self, id, method_name, *method_args, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=[id], **method_kwargs)[0]


class PickAndPlaceExpert:

    ANGLE_EPS = math.pi / 20

    def __init__(self, goal_pos, dist_eps=0.05):
        self.goal_pos = goal_pos
        self.history = deque(maxlen=20)
        self.should_close = False
        self.shape_at_goal = False
        self.done = False
        self.dist_eps = dist_eps

    def update_history(self, act_flags, robot_pos, robot_angle, robot_to_shape_dist):
        self.history.append({
            'action': act_flags,
            'robot_pos': robot_pos,
            'robot_angle': robot_angle,
            'robot_to_shape_dist': robot_to_shape_dist
        })

    def predict(self, robot, shape):

        history = self.history

        robot_pos = robot.position
        robot_angle = robot.angle
        robot_rotation_vector = robot.rotation_vector
        robot_vector = robot_rotation_vector.rotated_degrees(90)

        shape_pos = shape.position
        shape_angle = shape.angle

        goal_pos = self.goal_pos
        robot_to_shape_dist = shape_pos.get_distance(robot_pos)

        act_flags = [RA.NONE, RA.NONE, RA.OPEN]

        if self.shape_at_goal:
            cnt_down = 0
            i = len(history) - 1
            while i >= 0 and history[i]['action'][0] == RA.DOWN:
                cnt_down += 1
                i -= 1
            # continue back up
            if cnt_down < 5:
                act_flags[0] = RA.DOWN
            else:
                self.done = True
            self.update_history(act_flags, robot_pos, robot_angle, robot_to_shape_dist)
            return act_flags, self.done

        if not self.should_close and robot_to_shape_dist < 0.4 and len(history) >= 3:
            l = len(history)
            for i in range(l - 3, l - 1):
                d_now = history[i]['robot_to_shape_dist']
                d_next = history[i + 1]['robot_to_shape_dist']
                if abs(d_next - d_now) > 0.01:
                    self.should_close = True

        if self.should_close:

            robot_to_shape_dist = robot_pos.get_distance(shape_pos)
            if robot_to_shape_dist > 0.5:
                return act_flags, True

            act_flags[2] = RA.CLOSE
            shape_to_goal_dist = goal_pos.get_distance(shape_pos)
            if shape_to_goal_dist < self.dist_eps:
                self.shape_at_goal = True
                self.should_close = False
                act_flags[0] = RA.DOWN
                act_flags[2] = RA.OPEN
                self.update_history(act_flags, robot_pos, robot_angle, robot_to_shape_dist)
                return act_flags, self.done

            shape_to_goal_vector = (goal_pos - shape_pos).normalized()
            diff_angle = robot_vector.get_angle_between(shape_to_goal_vector)
            target_dist = shape_to_goal_dist
            angle_eps = self.ANGLE_EPS * 3
        else:
            robot_to_shape_vector = (shape_pos - robot_pos).normalized()
            diff_angle = robot_vector.get_angle_between(robot_to_shape_vector)
            target_dist = robot_to_shape_dist
            angle_eps = self.ANGLE_EPS

        robot_to_goal_dist = robot_pos.get_distance(goal_pos)
        if abs(diff_angle) < angle_eps:
            act_flags[0] = RA.UP
        elif self.should_close and robot_to_goal_dist < 0.2:
            act_flags[0] = RA.DOWN
        elif diff_angle < 0 and diff_angle >= -math.pi:
            act_flags[1] = RA.RIGHT
        else:
            act_flags[1] = RA.LEFT

        should_back_up = False

        cnt_down = 0
        i = len(history) - 1
        while i >= 0 and history[i]['action'][0] == RA.DOWN:
            cnt_down += 1
            i -= 1

        # continue back up
        if cnt_down > 0 and cnt_down < 10:
            should_back_up = True

        # not changing pose at all
        cnt_unmoved_pose = 0
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
            cnt_unmoved_pose += 1
            i -= 1

        if cnt_unmoved_pose >= 3:
            should_back_up = True

        # not changing position
        cnt_unmoved = 0
        i = len(history) - 1
        while i > 0:
            p1 = history[i]['robot_pos']
            p2 = history[i - 1]['robot_pos']
            dp = p1.get_distance(p2)
            if dp > 0.01:
                break
            cnt_unmoved += 1
            i -= 1

        if cnt_unmoved >= 19:
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

        self.update_history(act_flags, robot_pos, robot_angle, robot_to_shape_dist)

        return act_flags, self.done
