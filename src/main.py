from utils import evaluate_policy, str2bool
from datetime import datetime
from actor import SACD_agent
import gymnasium as gym
import os, shutil
import argparse
import torch
import utils
import threading
from PIL import Image
import numpy as np
import actor
import wandb

from embedder import Embedder, RewardModel

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--env_name', type=str, default="CartPole-v1", help='environment name, default:CartPole-v1')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=True, help='Render in 2D RGB array or human video')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--model_index', type=int, default=50, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--max_train_steps', type=int, default=10000, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e3, help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=100, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--hid_shape', type=list, default=[200,200], help='Hidden net shape')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
parser.add_argument('--distance_type', type=str, default="euclidean", help = "distance metric, either 'euclidean' or 'cosine'")
parser.add_argument("--mode", type=str, default="image", help="image or text mode")
parser.add_argument("--dump_every", type=int, default=5, help= "frequency at which to dump rewards in the replaybuffer")
parser.add_argument("--model", type=str, default = "CLIP", help="embedding model (CLIP, DINOV2)")
parser.add_argument("--automatic_entropy_tuning", type=bool, default=False, help= "Automatically adjust alpha (default:False)")
parser.add_argument("--policy", type=str, default="Gaussian", help="Policy type for SAC (Gaussian, Deterministic)")
parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument("--visualize_every", type=int, default=int(1e5), help="wandb visualization frequency")
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='Value target update per no. of updates per step (default: 1)')


opt = parser.parse_args()
opt.dvc = "cuda:0" if torch.cuda.is_available() else "cpu"

from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
DEFAULT_CAMERA_CONFIG = {
"distance": 1.35,
"azimuth": 160,
"elevation": -35.0,
"lookat": np.array([0, 0.6, -0.05]),
}

def main():
    # Create Env
    env, eval_env = utils.get_env(env_name=opt.env_name)

    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = utils.get_action_dim(env)
    #TODO : change this to a normal value
    #TODO: citation ?
    opt.max_e_steps = 250
    run = wandb.init(
    project="thesis",  # Specify your project
    config=opt,
    )

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed: {opt.seed}")
    print(opt)
    print(
        'Algorithm: SACD',
        ' Env:', opt.env_name,
        ' state_dim:', opt.state_dim,
        ' action_dim:', opt.action_dim,
        ' Random Seed:', opt.seed,
        ' max_e_steps:', opt.max_e_steps,
        '\n'
    )

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = datetime.now().strftime("%Y%m%d_%H%M%S")
        writepath = f'runs/SACD_{opt.env_name}_{timenow}'
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Build model directory
    if not os.path.exists('model'):
        os.mkdir('model')

    # Create Agent, Embedder and Reward Model
    agent = actor.choose_agent(opt, opt.env_name, env)
    print("Agent created")
    
    embedder = Embedder(**vars(opt))
    print("Embedder created")

    reward_model = RewardModel(embedder, goal_path=utils.get_goal_path(opt.env_name), **vars(opt))
    print("Reward model created")

    if opt.Loadmodel:
        agent.load(opt.ModelIdex, opt.env_name)

    total_steps : int = 0
    s, _ = env.reset(seed=env_seed)
    goal_embedding = reward_model.get_current_goal_embedding().unsqueeze(0)
    env.mujoco_renderer = MujocoRenderer(env.model, env.data, DEFAULT_CAMERA_CONFIG, width=400, height=400)
    states = []
    actions = []
    dws = []
    depictions = []
    count = 0
    #rewards = torch.empty(opt.dump_every, device = opt.dvc)
    while total_steps < opt.max_train_steps:
        s, info = env.reset(seed=env_seed)  # avoid overfitting seed
        env_seed += 1
        done = False


        # Interaction & training
        while not done:
            states.append(s)
            if total_steps % opt.dump_every == 0 and total_steps != 0:
                #print(total_steps, rewards.size(), len(depictions))
                #rewards = 1 - torch.exp(-rewards) # normalization
                rewards = reward_model.compute_rewards(depictions, goal_embedding)
                agent.dump_infos_to_replay_buffer(states, actions, rewards, dws)
                states = [s]
                actions = []
                dws = []
                depictions = []
                count = 0

            if total_steps < opt.random_steps:
                a = env.action_space.sample()
            else:
                a = agent.select_action(s, deterministic=False)

            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)
            depictions.append(env.render())
            #print(len(depictions))
            actions.append(a)
            #rewards[count] = r
            dws.append(dw)
            count += 1

            s = s_next

            # Training updates
            if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                for _ in range(opt.update_every):
                    agent.train()

            # Evaluation & logging
            if total_steps % opt.eval_interval == 0:
                avg_env_r, success_rate = evaluate_policy(eval_env, agent, turns=3)
                wandb.log({"average reward": avg_env_r, "success_rate": success_rate})
                if opt.write:
                    writer.add_scalar('ep_r', avg_env_r, global_step=total_steps)
                    writer.add_scalar('alpha', agent.alpha, global_step=total_steps)
                    writer.add_scalar('H_mean', agent.H_mean, global_step=total_steps)
                print(
                    'EnvName:', opt.env_name,
                    'seed:', opt.seed,
                    f'steps: {int(total_steps)}',
                    'score:', int(avg_env_r)
                )

            total_steps += 1
            if total_steps % opt.visualize_every == 0 and total_steps != 0:
                utils.visualize_episode(agent, env, reward_model, goal_embedding, total_steps)
            # Save model periodically
            if total_steps % opt.save_interval == 0:
                agent.save(int(total_steps/1000), opt.env_name)

    # Close environments
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
