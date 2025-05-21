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
import wandb
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from collections.abc import Iterable, Sized
import imageio

HAND_LOW = (-0.5,  0.40, 0.05)
HAND_HIGH = ( 0.5,  1.00, 0.50)



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
    env_ids = [env_spec for env_spec in all_envs]

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




def evaluate_policy(env, agent, turns = 10):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		successes = 0
		while not done:
			# Take deterministic actions at test time
			a = agent.select_action(state=s, deterministic=True)
			s_next, r, dw, tr, info = env.step(a)
			done = (dw or tr)
			total_scores += r
			s = s_next
			successes += dw
	return int(total_scores/turns), int(successes/turns)


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

def visualize_episode(agent, env, reward_model, goal_embedding, n_ep):
	sim   = env.sim
	model = sim.model
	data  = sim.data
	site_id = model.site_name2id("robot0:gripper_site") # <- assuming it is the position of the gripper
	s, _ = env.reset()
	done = False
	positions = []
	depictions = []
	env_rewards = []
	object_pos = []
	target_pos = []
	while not done:
		# Take deterministic actions at test time
		a = agent.select_action(state=s, deterministic=True)
		depictions.append(env.render())

		eef_pos = data.site_xpos[site_id]   # numpy array of shape (3,)
		positions.append(eef_pos) # we just want the (x,y) coordinates
		s_next, r, dw, tr, _ = env.step(a)
		done = (dw or tr)
		env_rewards.append(r)
		object_pos.append(env._get_pos_objects())
		target_pos.append(env._get_pos_goal())
		s = s_next
	subjective_rewards = reward_model.compute_rewards(depictions, goal_embedding)
	frames = create_episode_video(agent_pos=positions, object_pos=object_pos, target_pos=target_pos, env_rewards=env_rewards, subjective_rewards=subjective_rewards)
	output_path = "/visualizations/episode" + str(n_ep)
	write_video_imageio(frames, output_path=output_path)




def create_episode_video(agent_pos, object_pos, goal_pos, corner_1 = HAND_HIGH, corner_2 = HAND_LOW, **kwargs):
	# get the number of frames
	length = len(agent_pos)
	# create each frame
	frames = []
	pos_range = (corner_1 - corner_2)[:2] * 500

	# We proceed in 3 steps
	# 1) engineer the map where the agent is located
	# 2) create the side pannel where the textual information will be located
	# 3) concatenate the two together
	# ====== INGREDIENTS FOR STEP 1 =====
	videodims = int(pos_range)  # exclude z coordinate, view from above
	agent_pos[:,:2] *= 500
	object_pos[:,:2] *= 500
	goal_pos[:,:2] *= 500
	# round the coordinates
	agent_pos = int(agent_pos)
	object_pos = int(object_pos)
	goal_pos = int(goal_pos)
	
	# associate a color to the z coordinate
	z_min = HAND_LOW[2]
	z_max = HAND_HIGH[2]
	z_range = z_max - z_min
	agent_z_color = int( (agent_pos[:,:2] / z_range))
	obj_z_color = int( (object_pos[:,:2] / z_range))
	goal_z_color = int( (goal_pos[:,:2] / z_range))

	# matplotlib colormap for z coordinates
	cmap = plt.get_cmap('viridis')
	nkeys = len(kwargs)

	# separate between arrays of size n, and just pure strings
	lists = []
	not_lists = []
	for k, val in kwargs.item():
		if isinstance(val, Sized):
			assert len(val) == length
			lists.append((k,val))
		else:
			not_lists.append((k,val))

	bar = make_colorbar(cv2.COLORMAP_VIRIDIS, width = 20, height = videodims[1],
					 	vmin=-1.0, vmax=1.0, ticks=7)
	
	padding_size = (15, videodims[1])

	padding = np.ones((*padding_size, 3), dtype = np.uint8)

		

	images = []
	for i in range(length):
		blank_map = np.ones((*videodims, 3), dtype = np.uint8)
		cv2.rectangle(blank_map, (0,0), int(pos_range), (0,0,0), 1)
		cv2.circle(blank_map, agent_pos[i][:2], radius=2, color=cmap(agent_z_color[i])[:3])
		cv2.circle(blank_map, agent_pos[i][:2], radius=1, color=(255,0,0))

		cv2.circle(blank_map, object_pos[i][:2], radius=2, color=cmap(obj_z_color[i])[:3])
		cv2.circle(blank_map, object_pos[i][:2], radius=1, color=(0,0,255))

		cv2.circle(blank_map, goal_pos[i][:2], radius=2, color=cmap(goal_z_color[i])[:3])
		cv2.circle(blank_map, goal_pos[i][:2], radius=1, color=(0,255,0))

		text_pannel_size = (50, videodims[1])
		text_pannel = np.ones((*text_pannel_size, 3), dtype=np.uint8)

		# first we process strings, then arrays of size n

		count = 0

		for k, val in not_lists:
			string = "%s: %s" % (k, val)
			cv2.putText(text_pannel, string, (0, count * text_pannel_size[1]/nkeys))
			count += 1

		for k, val in lists:
			# we assume val is always an array containing n values depending on each time step
			# we also assume that each entry in val is a floating point number
			s = "%s : %.4f" % (k, val[i])
			cv2.putText(blank_map, s, (0, count * text_pannel_size / nkeys),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
			count += 1

		images.append(np.concatenate([blank_map, padding, bar, padding, text_pannel], axis=1))
	return images
	

	


def make_colorbar(cmap=cv2.COLORMAP_JET,
                  width=30, height=256,
                  vmin=0, vmax=1.0,
                  ticks=5, tick_font_scale=0.5):
    """
    Returns a colorbar image of shape (height, width, 3) mapping [vmin,vmax]
    through the specified OpenCV colormap.
    """
    # 1) Create a vertical gradient [0..255] as uint8
    grad = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
    bar  = np.repeat(grad, width, axis=1)          # shape (H, W)
    
    # 2) Apply colormap
    bar_color = cv2.applyColorMap(bar, cmap)       # shape (H, W, 3)
    
    # 3) Add tick labels
    for i in range(ticks):
        y = int(i * (height-1) / (ticks-1))
        val = vmin + (vmax - vmin) * (1 - y/(height-1))
        txt = f"{val:.2f}"
        cv2.putText(bar_color, txt, (width+5, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    tick_font_scale, (255,255,255), 1, cv2.LINE_AA)
    return bar_color
		





def write_video_imageio(frames, output_path, fps=30):
    """
    Writes a video using imageio.
    
    Parameters:
    - frames: list of np.ndarray, each with shape (H, W, 3) and dtype uint8 (RGB).
    - output_path: str, path to save the output video (e.g., 'output.mp4').
    - fps: int or float, frames per second.
    """
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            # If your frames are BGR, convert to RGB for imageio:
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame)
    print(f"Video saved to {output_path}")

# Usage:
# write_video_imageio(frames, 'output_imageio.mp4', fps=24)




