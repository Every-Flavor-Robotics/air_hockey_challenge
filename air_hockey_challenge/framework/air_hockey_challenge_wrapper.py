from copy import deepcopy

# gym
import gymnasium as gym
import numpy as np
from mushroom_rl.core import Environment

from air_hockey_challenge.constraints import *
from air_hockey_challenge.environments import position_control_wrapper as position
from air_hockey_challenge.utils import robot_to_world


class AirHockeyChallengeWrapper(Environment):
    def __init__(
        self, env, custom_reward_function=None, interpolation_order=3, **kwargs
    ):
        """
        Environment Constructor

        Args:
            env [string]:
                The string to specify the running environments. Available environments: [3dof-hit, 3dof-defend].
                [7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be available once the corresponding stage starts.
            custom_reward_function [callable]:
                You can customize your reward function here.
            interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
        """

        env_dict = {
            "tournament": position.IiwaPositionTournament,
            "7dof-hit": position.IiwaPositionHit,
            "7dof-defend": position.IiwaPositionDefend,
            "7dof-prepare": position.IiwaPositionPrepare,
            "3dof-hit": position.PlanarPositionHit,
            "3dof-defend": position.PlanarPositionDefend,
        }

        if env == "tournament" and type(interpolation_order) != tuple:
            interpolation_order = (interpolation_order, interpolation_order)

        self.base_env = env_dict[env](interpolation_order=interpolation_order, **kwargs)
        self.env_name = env
        self.env_info = self.base_env.env_info

        if custom_reward_function:
            self.base_env.reward = (
                lambda state, action, next_state, absorbing: custom_reward_function(
                    self.base_env, state, action, next_state, absorbing
                )
            )

        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add(EndEffectorConstraint(self.env_info))
        if "7dof" in self.env_name or self.env_name == "tournament":
            constraint_list.add(LinkConstraint(self.env_info))

        self.env_info["constraints"] = constraint_list
        self.env_info["env_name"] = self.env_name

        super().__init__(self.base_env.info)

    def step(self, action):
        obs, reward, done, info = self.base_env.step(action)

        if "tournament" in self.env_name:
            info["constraints_value"] = list()
            info["jerk"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs) / 2) : (i + 1) * int(len(obs) / 2)]
                info["constraints_value"].append(
                    deepcopy(
                        self.env_info["constraints"].fun(
                            obs_agent[self.env_info["joint_pos_ids"]],
                            obs_agent[self.env_info["joint_vel_ids"]],
                        )
                    )
                )
                info["jerk"].append(
                    self.base_env.jerk[
                        i
                        * self.env_info["robot"]["n_joints"] : (i + 1)
                        * self.env_info["robot"]["n_joints"]
                    ]
                )

            info["score"] = self.base_env.score
            info["faults"] = self.base_env.faults

        else:
            info["constraints_value"] = deepcopy(
                self.env_info["constraints"].fun(
                    obs[self.env_info["joint_pos_ids"]],
                    obs[self.env_info["joint_vel_ids"]],
                )
            )
            info["jerk"] = self.base_env.jerk
            info["success"] = self.check_success(obs)

        return obs, reward, done, info

    def render(self, record=False):
        return self.base_env.render(record=record)

    def reset(self, state=None):
        return self.base_env.reset(state)

    def check_success(self, obs):
        puck_pos, puck_vel = self.base_env.get_puck(obs)

        puck_pos, _ = robot_to_world(
            self.base_env.env_info["robot"]["base_frame"][0], translation=puck_pos
        )
        success = 0

        if "hit" in self.env_name:
            if (
                puck_pos[0] - self.base_env.env_info["table"]["length"] / 2 > 0
                and np.abs(puck_pos[1])
                - self.base_env.env_info["table"]["goal_width"] / 2
                < 0
            ):
                success = 1

        elif "defend" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.2 and puck_vel[0] < 0.1:
                success = 1

        elif "prepare" in self.env_name:
            if (
                -0.8 < puck_pos[0] <= -0.2
                and np.abs(puck_pos[1]) < 0.39105
                and puck_vel[0] < 0.1
            ):
                success = 1
        return success


# Create a gym environment wrapper


class AirHockeyChallengeGymWrapper(gym.Env):

    def __init__(
        self, env, custom_reward_function=None, interpolation_order=3, **kwargs
    ):

        if custom_reward_function is None:

            def custom_reward_function(self, state, action, next_state, absorbing):
                puck_pos = next_state[0:2]

                # Compute error from (1.948/2, 0)
                error = np.linalg.norm(puck_pos - np.array([1.948 / 2, 0]))

                return -error

        self.env = AirHockeyChallengeWrapper(
            env, custom_reward_function, interpolation_order, **kwargs
        )

        # print(type(self.env.env_info["rl_info"].action_space))
        # exit()

        # Convert mushroom rl Box to gym Box
        self.action_space = gym.spaces.Box(
            low=self.env.env_info["rl_info"].action_space.low,
            high=self.env.env_info["rl_info"].action_space.high,
            dtype=np.float32,
        )

        # Convert mushroom rl Box to gym Box
        self.observation_space = gym.spaces.Box(
            low=self.env.env_info["rl_info"].observation_space.low,
            high=self.env.env_info["rl_info"].observation_space.high,
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

    def reset(self, seed=None):
        return self.env.reset(), {}

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        pass


if __name__ == "__main__":
    env = AirHockeyChallengeWrapper(env="7dof-hit")
    env.reset()

    R = 0.0
    J = 0.0
    gamma = 1.0
    steps = 0
    while True:
        action = np.random.uniform(-1, 1, (2, env.env_info["robot"]["n_joints"])) * 3
        observation, reward, done, info = env.step(action)
        env.render()
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.0
            J = 0.0
            gamma = 1.0
            steps = 0
            env.reset()
