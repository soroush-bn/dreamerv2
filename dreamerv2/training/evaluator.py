import numpy as np
import torch
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
import gym
import matplotlib.pyplot as plt
from PIL import Image
from dreamerv2.training.converter import Converter
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
        self.timestep = 0

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
        converter = Converter((0, 9), (0, 0), (4, 4))
        eval_scores = []
        if self.is_real_gym:
            env = gym.make("Pong-v0", render_mode='human')
            env = OneHotAction(env)
        for e in range(eval_episode):
            obs, score = env.reset(), 0
            if self.is_real_gym and self.is_compact:
                obs = converter.convert_to_compact(obs)
            elif self.is_real_gym:
                obs = converter.three_channel_converter(obs)

            done = False
            prev_rssmstate = self.RSSM._init_rssm_state(1)
            prev_action = torch.zeros(1, self.action_size).to(self.device)
            while not done:
                self.timestep += 1
                with torch.no_grad():
                    embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
                    _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    model_state = self.RSSM.get_model_state(posterior_rssm_state)
                    action, _ = self.ActionModel(model_state)
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action

                    # if (action.squeeze(0).cpu().numpy() == [0,1,0]).all() == False :
                    # print(action.squeeze(0).cpu().numpy())
                    # pass

                    # print("action to perform " + str(action.squeeze(0).cpu().numpy()))
                if self.is_real_gym:
                    not_changed = True
                    while not_changed or not done:
                        np_action = action.squeeze(0).cpu().numpy()
                        a = np.array([np_action[0], 0, np_action[2], np_action[1], 0, 0])
                        next_obs, rew, done, _ = env.step(a)
                        if not np.array_equal(obs, converter.convert_to_compact(next_obs)):
                            not_changed = False
                else:
                    next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
                self.get_imagined_obs(horizon=5, posterior_rssm_state=posterior_rssm_state, obs=next_obs)

                # frame_skip = 8
                # for i in range(frame_skip):
                #     np_action = action.squeeze(0).cpu().numpy()
                #     a = np.array([np_action[0], 0, np_action[2], np_action[1], 0, 0])
                #     next_obs, rew, done, _ = env.step(a)

                if True:
                    if self.is_real_gym:
                        env.render(mode="rgb_array")
                    else:
                        env.render()
                score += rew
                if self.is_real_gym and self.is_compact:
                    obs = converter.convert_to_compact(next_obs)
                elif self.is_real_gym:
                    obs = converter.three_channel_concverter(next_obs)
                else:
                    obs = next_obs
            eval_scores.append(score)
        print('average evaluation score for model at ' + model_path + ' = ' + str(np.mean(eval_scores)))
        env.close()
        return np.mean(eval_scores)

    def get_imagined_obs(self, horizon, posterior_rssm_state, obs):
        next_rssm_states, imag_log_probs, action_entropy = self.RSSM.rollout_imagination(horizon,
                                                                                         self.ActionModel,
                                                                                         posterior_rssm_state)

        obs = Image.fromarray(obs)
        if obs.mode != 'RGB':
            obs = obs.convert('RGB')
        # obs = obs.resize((80,80))
        name = "E:\\projects\\dreamerv2-minatar\\dreamerv2\\training\\imaginations_real\\" + str(self.timestep) + ".jpeg"
        obs.save(name)
        _, x = self.ObsDecoder.forward(torch.cat((next_rssm_states.deter, next_rssm_states.stoch), dim=-1))
        for i in range(horizon):
            img = x[i].squeeze(0).cpu().detach().numpy()
            img = Image.fromarray(img*255)
            if img.mode != 'RGB':
                img= img.convert('RGB')
            img = img.resize((80, 80))

            name = "E:\\projects\\dreamerv2-minatar\\dreamerv2\\training\\imaginations_real\\" + str(
                self.timestep) + "_" + str(i) +".jpeg"
            img.save(name)
