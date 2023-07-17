import gym
import numpy as np
from gym.spaces import Discrete, Box


class PhysicWrapper(gym.Wrapper):
    def __init__(self, env):
        super(PhysicWrapper, self).__init__(env)
        self.env.observation_space = Box(
            low=-80, high=80, shape=(1, 10), dtype=np.uint8
        )
        self.state = np.zeros((10))

    @property
    def observation_space(self):

        return self.env.observation_space

    def reset(self, **kwargs):
        self.env.reset()
        return self.state

    def render(self, mode="human", **kwargs):
        return super().render(mode, **kwargs)

    def close(self):
        return super().close()

    def step(self, action):
        obs1, rew1, done1, info1 = self.env.step(action)
        obs2, rew2, done2, info2 = self.env.step(action)

        pr_pos1, pl_pos1, ball_pos1 = self.find_positions(obs1)
        pr_pos2, pl_pos2, ball_pos2 = self.find_positions(obs2)
        x_pl = pl_pos2[0]
        y_pl = pl_pos2[1]
        dx_pl = pl_pos2[0] - pl_pos1[0]
        x_pr = pr_pos2[0]
        y_pr = pr_pos2[1]
        dx_pr = pr_pos2[0] - pr_pos1[0]

        x_ball = ball_pos2[0]
        y_ball = ball_pos2[1]
        dx_ball = ball_pos2[0] - ball_pos1[0]
        dy_ball = ball_pos2[1] - ball_pos1[1]

        self.state = np.array([x_pl, y_pl, dx_pl, x_pr, y_pr, dx_pr, x_ball, y_ball, dx_ball, dy_ball])

        return self.state, rew1 + rew2, done1 or done2, info2

    def find_positions(self, obs):
        p_obs = self.preprocess_single(obs)
        bw_obs = self.make_bw_frame(p_obs)
        pr_pos, bw_obs = self.find_paddle_right(bw_obs, (-1, -1))
        pl_pos, bw_obs = self.find_paddle_left(bw_obs, (-1, -1))
        ball_pos = self.find_ball(bw_obs, (-1, -1))
        return pr_pos, pl_pos, ball_pos

    def preprocess_single(self, image, bkg_color=np.array([144, 72, 17])):
        # print('image[34:-16:2,::2].shape: ', image[34:-16:2,::2].shape)
        img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1)
        return img

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

    def find_ball(self, bw_obs, previous_position: tuple):
        mask = np.ones((8, 8))
        for i in range(10):
            for j in range(10):
                res = bw_obs[i * 8:(8 + i * 8), j * 8:(8 + j * 8)] * mask
                if res.sum() == 2 * 255:
                    if j != previous_position[1]:
                        return int(i * 8), int(j * 8)
        return previous_position

    def find_paddle_right(self, bw_obs, previous_position: tuple):
        # plt.imshow(bw_obs)
        bw_obs2 = bw_obs[:, 70:72]
        mask = np.ones((8, 2))
        for i in range(bw_obs2.shape[0] - 8):
            res = bw_obs2[i:(8 + i), :] * mask
            if res.sum() == 16 * 255:
                bw_obs[i:(8 + i), 70:72] = 0
                return (i, 70), bw_obs
        return previous_position, bw_obs

    def find_paddle_left(self, bw_obs, previous_position: tuple):
        bw_obs2 = bw_obs[:, 8:10]

        mask = np.ones((8, 2))
        for i in range(bw_obs2.shape[0] - 8):
            res = bw_obs2[i:(8 + i), :] * mask
            if res.sum() == 16 * 255:
                bw_obs[i:(8 + i), 9:11] = 0
                return (i, 9), bw_obs
        return previous_position, bw_obs
