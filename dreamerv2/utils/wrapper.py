import gym
import numpy as np
from gym.spaces import Box

import minatar


# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# from dreamerv2.training.converter import Converter


class DeepMindWrapperPong(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.observation_space = Box(
            low=0, high=255, shape=(1, 80, 80), dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        pre_obs = self.pre(obs)
        expanded = np.expand_dims(pre_obs, axis=0)
        return expanded

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        pre_obs = self.pre(obs)
        expanded = np.expand_dims(pre_obs, axis=0)

        return expanded, rew, done, info

    def preprocess_single(self, image, bkg_color=np.array([144, 72, 17])):
        # print('image[34:-16:2,::2].shape: ', image[34:-16:2,::2].shape)
        img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1)
        return img

    def pre(self, obs):
        p_obs = self.preprocess_single(obs)
        bw_obs = self.make_bw_frame(p_obs)
        return bw_obs

    def make_bw_frame(self, p_obs):
        p_obs = p_obs.astype(int)
        ball_index = np.where(p_obs == 158)
        pads_index_right = np.where(p_obs == 61)
        pads_index_left = np.where(p_obs == 45)
        bw_obs = np.zeros(p_obs.shape)
        bw_obs[ball_index] = 255
        bw_obs[pads_index_right] = 255
        bw_obs[pads_index_left] = 255
        return bw_obs


class GymMinAtarCompact(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env_name = env_name
        self.env = minatar.Environment(env_name)
        self.minimal_actions = self.env.minimal_action_set()
        h, w, c = self.env.compact_state_shape()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        self.observation_space = gym.spaces.MultiBinary((c, h, w))

    def reset(self):
        self.env.reset()
        return self.env.compact_state().transpose(2, 0, 1)

    def step(self, index, a_prime=None):
        '''index is the action id, considering only the set of minimal actions'''
        action = self.minimal_actions[index]
        if a_prime is not None:
            a_prime = self.minimal_actions[a_prime]
        r, terminal = self.env.act(action, a_prime)
        self.game_over = terminal
        return self.env.compact_state().transpose(2, 0, 1), r, terminal, {}

    def seed(self, seed='None'):
        self.env = minatar.Environment(self.env_name, random_seed=seed)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.state()
        elif mode == 'human':
            # print("bonjooor")
            self.env.display_state(self.display_time)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0


class GymMinAtar(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env_name = env_name
        self.env = minatar.Environment(env_name)
        self.minimal_actions = self.env.minimal_action_set()
        h, w, c = self.env.state_shape()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        self.observation_space = gym.spaces.MultiBinary((c, h, w))

    def reset(self):
        self.env.reset()
        return self.env.state().transpose(2, 0, 1)

    def step(self, index):
        '''index is the action id, considering only the set of minimal actions'''
        action = self.minimal_actions[index]
        r, terminal = self.env.act(action)
        self.game_over = terminal
        return self.env.state().transpose(2, 0, 1), r, terminal, {}

    def seed(self, seed='None'):
        self.env = minatar.Environment(self.env_name, random_seed=seed)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.state()
        elif mode == 'human':
            self.env.display_state(self.display_time)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0


class breakoutPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        '''index 2 (trail) is removed, which gives ball's direction'''
        super(breakoutPOMDP, self).__init__(env)
        c, h, w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c - 1, h, w))

    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)


class asterixPOMDP(gym.ObservationWrapper):
    '''index 2 (trail) is removed, which gives ball's direction'''

    def __init__(self, env):
        super(asterixPOMDP, self).__init__(env)
        c, h, w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c - 1, h, w))

    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)


class freewayPOMDP(gym.ObservationWrapper):
    '''index 2-6 (trail and speed) are removed, which gives cars' speed and direction'''

    def __init__(self, env):
        super(freewayPOMDP, self).__init__(env)
        c, h, w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c - 5, h, w))

    def observation(self, observation):
        return np.stack([observation[0], observation[1]], axis=0)


class space_invadersPOMDP(gym.ObservationWrapper):
    '''index 2-3 (trail) are removed, which gives aliens' direction'''

    def __init__(self, env):
        super(space_invadersPOMDP, self).__init__(env)
        c, h, w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c - 2, h, w))

    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[4], observation[5]], axis=0)


class seaquestPOMDP(gym.ObservationWrapper):
    '''index 3 (trail) is removed, which gives enemy and driver's direction'''

    def __init__(self, env):
        super(seaquestPOMDP, self).__init__(env)
        c, h, w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c - 1, h, w))

    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[2], observation[4], observation[5], observation[6],
                         observation[7], observation[8], observation[9]], axis=0)


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=1):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class Converter10x10(gym.Wrapper):
    def __init__(self, env, converter):
        super(Converter10x10, self).__init__(env)
        self.converter = converter
        self.action_map = ['n', 'l', 'u', 'r', 'd', 'f']

        self.minimal_actions = self.minimal_action_set()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        # self.observation_space = gym.spaces.MultiBinary((1, 10, 10))

    @property
    def observation_space(self):
        return gym.spaces.MultiBinary((1, 10, 10))

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        obs = self.converter.convert_to_compact(obs)
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = self.converter.convert_to_compact(obs)
        return obs

    def minimal_action_set(self):
        minimal_actions = ['n', 'l', 'r']
        return [self.action_map.index(x) for x in minimal_actions]


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)

    def step(self, action, action_prime=None):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if action_prime is not None:
            index_prime = np.argmax(action_prime).astype(int)
            reference_prime = np.zeros_like(action_prime)
            reference_prime[index_prime] = 1
            return self.env.step(index, index_prime)
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    # def render(self, mode="human", **kwargs):
    #     self.env.game.display_state(50)
    #     return super().render(mode, **kwargs)
    #
    # def close(self):
    #     self.env.game.close_display()
    #     return super().close()

    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference
