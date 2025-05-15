import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import math
import torch
# model imports
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import numpy as np
from PIL import Image
import argparse
import metaworld
import random
import gymnasium as gym


QUERIES = {
    "CartPole-v1" : "What is in this picture ? The goal of the agent is to keep the pole upright. Is the pole upright in this picture ? If not, edit this picture, while preserving the proportions, such that the pole is upright. If it is already upright, do not do anything." 
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def save_env_image(env):
	# whatever this is, fix later, too lazy rn ngl
	path = "../assets/img.png"
	img = env.render()
	img = Image.fromarray(img)
	img.save(path)
	return path

# TODO : modularize
# def get_goal_embedding(mode, env_name = "CartPole-v1", sys_path_to_goal= None):
# 	if mode == "image":
# 		img = Image.open(sys_path_to_goal)
# 		return get_image_embedding(img)
# 	elif mode == "text":
# 		query = QUERIES[env_name]
# 		return get_text_embedding(clip.tokenize([query]).to(device))



def get_goal_embedding(env, query = "a cartpole standing upright"):
    embedding = get_text_embedding(clip.tokenize([query]).to(device))
    return embedding



def get_text_embedding(tokens, model):
	with torch.no_grad():
		return model.encode_text(tokens)



def get_env(env_name):
    all_envs = gym.envs.registry.keys()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name, render_mode="rgb_array")
        eval_env = gym.make(env_name)

    elif env_name in metaworld.ML1.ENV_NAMES:
        ml1 = metaworld.ML1(env_name)
        env = ml1.train_classes[env_name](render_mode="rgb_array")
        task = random.choice(ml1.train_tasks)
        env.set_task(task)

        eval_env = ml1.train_classes[env_name](render_mode="rgb_array")
        task = random.choice(ml1.train_tasks)
        eval_env.set_task(task)

    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env, eval_env

#TODO get action dimension based on the environment
def get_action_dim(env):
    if isinstance(env.action_space, gym.spaces.Box):
        return env.action_space.shape[0]
    elif isinstance(env.action_space, gym.spaces.Discrete):
        return env.action_space.n
    else:
        raise NotImplementedError("action space not implemented yet")




#TODO choose action based on the environment
# continuous action space -> continuous SAC ig
# discrete action space -> discrete SAC
def choose_agent(opt, env_name, env):
	if isinstance(env.action_space, gym.spaces.Box):
		#TODO return agent that has a continuous policy
		pass
	elif isinstance(env.action_space, gym.spaces.Discrete):
		return 

def evaluate_policy(env, agent, turns = 3):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		while not done:
			# Take deterministic actions at test time
			a = agent.select_action(s, deterministic=True)
			s_next, r, dw, tr, info = env.step(a)
			done = (dw or tr)
			total_scores += r
			s = s_next
	return int(total_scores/turns)


#You can just ignore 'str2bool'. Is not related to the RL.
def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	

def compute_distance(a, b, dist_type = "euclidean"):
	"""
	distance between a and b is bounded between 0 and infty
	"""
	if dist_type == "euclidean":
		return torch._euclidean_dist(a,b)
	elif dist_type == "cosine":
		sim = torch.cosine_similarity(a,b)
		return (1 - sim) / (1+ sim + 1e-6)


def compute_reward(a, b, dist_type = "euclidean"):
	"""
	bijection from [0, infty) to [0,1)
	"""
	return torch.exp(-compute_distance(a,b,dist_type=dist_type))

def compute_rewards(rgb_imgs, goal, model):
	with torch.inference_mode():
		embeddings = model.forward_image(rgb_imgs)
		#print(embeddings)
		#print(embeddings.size(), goal.size(), rgb_imgs.size())
		rewards = compute_reward(embeddings, goal)
		# L2 norm squared
		return rewards


def get_goal_path(env_name):
	return "assets/" + env_name + "-goal"