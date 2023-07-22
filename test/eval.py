import argparse
import os
import sys

import gym
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dreamerv2.utils.wrapper import GymMinAtar, GymMinAtarCompact, OneHotAction, breakoutPOMDP, space_invadersPOMDP, \
    seaquestPOMDP, \
    asterixPOMDP, freewayPOMDP, AtariARIWrapper, Usefulram
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.evaluator import Evaluator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

pomdp_wrappers = {
    'breakout': breakoutPOMDP,
    'seaquest': seaquestPOMDP,
    'space_invaders': space_invadersPOMDP,
    'asterix': asterixPOMDP,
    'freeway': freewayPOMDP,
}


def main(args):
    print(args)
    compact = True if args.compact == 1 else False
    real_gym = True if args.gym== 1 else False
    env_name = args.env
    # if args.pomdp == 1:
    #     exp_id = args.id + '_pomdp'
    #     PomdpWrapper = pomdp_wrappers[env_name]
    #     if compact:
    #         env = PomdpWrapper(OneHotAction(GymMinAtarCompact(env_name)))
    #     else:
    #         env = PomdpWrapper(OneHotAction(GymMinAtar(env_name)))
    #     print('using partial state info')
    # else:
    #     exp_id = args.id
    #     if compact:
    #         env = OneHotAction(GymMinAtarCompact(env_name))
    #     else :
    #         env = OneHotAction(GymMinAtar(env_name))
    #     print('using complete state info')
    env = gym.make("Pong-ram-v0",render_mode = "human")
    env = Usefulram(AtariARIWrapper(env))
    env = OneHotAction(env)
    if args.eval_episode == 1:
        eval_render = True
    else:
        eval_render = False

    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using :', device)
    exp_id = args.id
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')

    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = bool
    action_dtype = np.float32

    config = MinAtarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
        model_dir=model_dir,
        eval_episode=args.eval_episode,
        eval_render=eval_render
    )

    evaluator = Evaluator(config, device,compact,real_gym)
    best_score = 0

    for f in sorted(os.listdir(model_dir)):
        eval_score = evaluator.eval_saved_agent(env, os.path.join(model_dir, f))
        if eval_score > best_score:
            print('..saving model number')
            best_score = eval_score

    print('best mean evaluation score amongst stored models is : ', best_score)


if __name__ == "__main__":
    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="pong", type=str, help='mini atari env name')
    parser.add_argument('--eval_episode', type=int, default=5, help='number of episodes to eval')
    parser.add_argument("--id", type=str, default='ram_8tayi', help='Experiment ID')
    parser.add_argument("--eval_render", default=0, type=int, help='to render while evaluation')
    parser.add_argument("--pomdp", default=0, type=int, help='partial observation flag')
    parser.add_argument("--gym", type=int, default=1, help="run real gym env or minatar")
    parser.add_argument("--compact", type=int, default=1, help="run compact or channeled")
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    args = parser.parse_args()
    main(args)
