import numpy as np
import torch
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
import gym

from dreamerv2.utils import OneHotAction


class Evaluator(object):
    '''
    used this only for minigrid envs
    '''

    def __init__(
            self,
            config,
            device,
            is_compact,
            is_real_gym,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.is_compact = is_compact
        self.is_real_gym = is_real_gym

    def load_model(self, config, model_path):
        saved_dict = torch.load(model_path)
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size']
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size * class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size

        if config.pixel:
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device).eval()
            self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(
                self.device).eval()
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()

        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor,
                                               config.expl).to(self.device).eval()
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type,
                         config.rssm_info).to(self.device).eval()

        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])

    def eval_saved_agent(self, env, model_path):
        self.load_model(self.config, model_path)
        eval_episode = self.config.eval_episode
        eval_scores = []
        if self.is_real_gym:
            env = gym.make("Pong-v0", render_mode='rgb_array')
            env = OneHotAction(env)
        for e in range(eval_episode):
            obs, score = env.reset(), 0
            if self.is_real_gym and self.is_compact:
                obs = convert_to_compact(obs)
            elif self.is_real_gym:
                obs = three_channel_converter(obs)
            done = False
            prev_rssmstate = self.RSSM._init_rssm_state(1)
            prev_action = torch.zeros(1, self.action_size).to(self.device)
            while not done:
                with torch.no_grad():
                    embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
                    _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    model_state = self.RSSM.get_model_state(posterior_rssm_state)
                    action, _ = self.ActionModel(model_state)
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                    # if (action.squeeze(0).cpu().numpy() == [0,0,1]).all() == False :
                    # print("sssss")
                    # pass

                    # print("action to perform " + str(action.squeeze(0).cpu().numpy()))
                next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
                if False:
                    env.render()
                score += rew
                if self.is_real_gym and self.is_compact:
                    obs = convert_to_compact(obs)
                elif self.is_real_gym:
                    obs = three_channel_converter(obs)
                else:
                    obs = next_obs
            eval_scores.append(score)
        print('average evaluation score for model at ' + model_path + ' = ' + str(np.mean(eval_scores)))
        env.close()
        return np.mean(eval_scores)



# todo fix this by importing
def three_channel_converter(obs):
    channeled = np.zeros((3, 10, 10))
    compact = convert_to_compact(obs)
    compact = compact / 255
    channeled[0, :, 0] = compact[:, 1]
    channeled[1, :, 9] = compact[:, 8]
    channeled[2] = compact
    channeled[2, :, 1] = 0
    channeled[2, :, 8] = 0
    return channeled


def convert_to_compact(frame):
    result = np.zeros((10, 10))
    p_obs = preprocess_single(frame)
    bw_obs = make_bw_frame(p_obs)
    converted_obs = conv2dpong(bw_obs)
    result[:, 0] = converted_obs[:, 1]

    result[:, 9] = converted_obs[:, 8]

    return converted_obs


def make_bw_frame(p_obs):
    p_obs = p_obs.astype(int)
    ball_index = np.where(p_obs == 158)
    pads_index_right = np.where(p_obs == 61)
    pads_index_left = np.where(p_obs == 45)
    bw_obs = np.zeros(p_obs.shape)
    bw_obs[ball_index] = 255
    bw_obs[pads_index_right] = 255
    bw_obs[pads_index_left] = 255
    return bw_obs


def conv2dpong(input):
    conv = np.zeros((10, 10))
    i = 0
    j = 0
    # for c in range(100):
    for i in range(10):
        for j in range(10):
            conv[i, j] = input[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].sum()

    conv[np.where(conv > 0)] = 255
    return conv


# plt.imshow(conv)


def preprocess_single(image, bkg_color=np.array([144, 72, 17])):
    # print('image[34:-16:2,::2].shape: ', image[34:-16:2,::2].shape)
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1)
    return img
