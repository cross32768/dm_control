import cv2
import gym
import numpy as np


class GymWrapper(object):

    metadata = {'render.modes': ['rgb_array']}
    reward_range = (-np.inf, np.inf)

    def __init__(self, env, render_size=(64, 64), camera_id=0):
        self._env = env
        self._render_size = render_size
        self._camera_id = camera_id

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        obs_spec = self._env.observation_spec()
        return gym.spaces.Box(0, 255, obs_spec['pixels'].shape, dtype=np.uint8)

    @property
    def action_space(self):
        action_spec = self._env.action_spec()
        return gym.spaces.Box(
            action_spec.minimum, action_spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation['pixels']
        reward = time_step.reward or 0
        done = time_step.last()
        info = None
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = time_step.observation['pixels']
        return obs
    
    def get_extended_observation(self, render_kwargs=None):
        obs = self._env.physics.render(render_kwargs)
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        del args
        del kwargs
        return self._env.physics.render(
            *self._render_size, camera_id=self._camera_id)
