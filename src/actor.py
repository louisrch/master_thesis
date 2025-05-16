import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from networks import Double_Q_Net, Policy_Net, GaussianPolicy, DeterministicPolicy, ReplayBuffer
from utils import hard_update, soft_update
import os

import gymnasium as gym


class Actor:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))

    def save(self):
        #TODO implement
        pass

    def load(self):
        #TODO implement
        pass

    def dump_infos_into_replay_buffer(self, states, actions, rewards, dws):
        # assertion that we can extract states - next_states pairs
        assert len(states) == len(actions) +1
        next_states = states[1:]
        states = states[:-1]
        self.replay_buffer.addAll(states, actions, rewards, next_states, dws)



class SACD_agent:
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.H_mean = 0
        self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))

        self.actor = Policy_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.q_critic = Double_Q_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        if self.adaptive_alpha:
            # We use 0.6 because the recommended 0.98 will cause alpha explosion.
            self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))  # H(discrete)>0
            self.log_alpha = torch.tensor(
                np.log(self.alpha),
                dtype=torch.float,
                requires_grad=True,
                dvc=self.dvc
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from (s_dim,) to (1, s_dim)
            probs = self.actor(state)
            if deterministic:
                return probs.argmax(-1).item()
            return Categorical(probs).sample().item()

    def train(self):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        # ------------------------------------------ Train Critic ---------------------------------------- #
        '''Compute the target soft Q value'''
        with torch.no_grad():
            next_probs = self.actor(s_next)  # [b,a_dim]
            next_log_probs = torch.log(next_probs + 1e-8)  # [b,a_dim]
            next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(
                next_probs * (min_next_q_all - self.alpha * next_log_probs),
                dim=1,
                keepdim=True
            )  # [b,1]
            target_Q = r + (~dw) * self.gamma * v_next

        '''Update soft Q net'''
        q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
        q1 = q1_all.gather(1, a)
        q2 = q2_all.gather(1, a)  # [b,1]
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ------------------------------------------ Train Actor ---------------------------------------- #
        probs = self.actor(s)  # [b,a_dim]
        log_probs = torch.log(probs + 1e-8)  # [b,a_dim]
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
        min_q_all = torch.min(q1_all, q2_all)

        a_loss = torch.sum(
            probs * (self.alpha * log_probs - min_q_all),
            dim=1
        )  # [b,]

        self.actor_optimizer.zero_grad()
        a_loss.mean().backward()
        self.actor_optimizer.step()

        # ------------------------------------------ Train Alpha ---------------------------------------- #
        if self.adaptive_alpha:
            with torch.no_grad():
                self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

        # ------------------------------------------ Update Target Net ---------------------------------- #
        for param, target_param in zip(
            self.q_critic.parameters(), self.q_critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, timestep, EnvName):
        torch.save(
            self.actor.state_dict(),
            f"./model/sacd_actor_{timestep}_{EnvName}.pth"
        )
        torch.save(
            self.q_critic.state_dict(),
            f"./model/sacd_critic_{timestep}_{EnvName}.pth"
        )

    def load(self, timestep, EnvName):
        self.actor.load_state_dict(
            torch.load(
                f"./model/sacd_actor_{timestep}_{EnvName}.pth",
                map_location=self.dvc
            )
        )
        self.q_critic.load_state_dict(
            torch.load(
                f"./model/sacd_critic_{timestep}_{EnvName}.pth",
                map_location=self.dvc
            )
        )

    def dump_infos_to_replay_buffer(self, states, actions, rewards, dws):
        """
        states : list of N+1 1xS numpy arrays
        actions : list of N actions
        depictions : list of N+1 depiction of states, 224x224x3 numpy arrays
        dws : list of episode completions
        """
        next_states = states[1:]
        states = states[:-1]
        self.replay_buffer.addAll(states, actions, rewards, next_states, dws)


class SAC(object):
    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)
        
        self.policy_type = self.policy
        self.target_update_interval = self.target_update_interval
        self.automatic_entropy_tuning = self.automatic_entropy_tuning
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.dvc)
        self.update_count = 0
        self.device = self.dvc

        self.critic = Double_Q_Net(self.state_dim, self.action_dim, self.hid_shape).to(dvc=self.dvc)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = Double_Q_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.dvc)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, dvc=self.dvc)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy(num_inputs=self.state_dim,
                                        num_actions=self.action_dim, 
                                        hidden_dim=self.hid_shape
                                        ).to(self.dvc)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs=self.state_dim,
                                                num_actions=self.action_dim,
                                                hidden_dim=self.hid_shape
                                                ).to(self.dvc)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.dvc).unsqueeze(0)
        if deterministic is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def train(self):
        self.update_parameters(self.replay_buffer, self.batch_size, self.update_count)
        self.update_count += 1

    def update_parameters(self, memory : ReplayBuffer, batch_size : int, updates : int):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.dvc)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.dvc)
        action_batch = torch.FloatTensor(action_batch).to(self.dvc)
        reward_batch = torch.FloatTensor(reward_batch).to(self.dvc).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.dvc).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.dvc)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name : str, suffix : str ="", ckpt_path : str = None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path : str, evaluate : bool =False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
    
    def dump_infos_to_replay_buffer(self, states, actions, rewards, dws):
        """
        states : list of N+1 1xS numpy arrays
        actions : list of N actions
        depictions : list of N+1 depiction of states, 224x224x3 numpy arrays
        dws : list of episode completions
        """
        next_states = states[1:]
        states = states[:-1]
        self.replay_buffer.addAll(states, actions, rewards, next_states, dws)



ACTOR_DICT = {
	'assembly-v2' : SAC,
	'basketball-v2':SAC,
	'bin-picking-v2' : SAC,
	'box-close-v2' : SAC,
	'button-press-topdown-v2' : SAC,
	'button-press-topdown-wall-v2' : SAC,
	'button-press-v2' : SAC,
	'button-press-wall-v2' : SAC,
	'coffee-button-v2' : SAC,
	'coffee-pull-v2' : SAC,
	'coffee-push-v2' : SAC, 
	'dial-turn-v2' : SAC, 
	'disassemble-v2' : SAC, 
	'door-close-v2': SAC,
    'door-lock-v2': SAC,
    'door-open-v2': SAC,
    'door-unlock-v2': SAC,
    'hand-insert-v2': SAC,
    'drawer-close-v2': SAC,
    'drawer-open-v2': SAC,
    'faucet-open-v2': SAC,
    'faucet-close-v2': SAC,
    'hammer-v2': SAC,
    'handle-press-side-v2': SAC,
    'handle-press-v2': SAC,
    'handle-pull-side-v2': SAC,
    'handle-pull-v2': SAC,
    'lever-pull-v2': SAC,
    'pick-place-wall-v2': SAC,
    'pick-out-of-hole-v2': SAC,
    'pick-place-v2': SAC,
    'plate-slide-v2': SAC,
    'plate-slide-side-v2': SAC,
    'plate-slide-back-v2': SAC,
    'plate-slide-back-side-v2': SAC,
    'peg-insert-side-v2': SAC,
    'peg-unplug-side-v2': SAC,
    'soccer-v2': SAC,
    'stick-push-v2': SAC,
    'stick-pull-v2': SAC,
    'push-v2': SAC,
    'push-wall-v2': SAC,
    'push-back-v2': SAC,
    'reach-v2': SAC,
    'reach-wall-v2': SAC,
    'shelf-place-v2': SAC,
    'sweep-into-v2': SAC,
    'sweep-v2': SAC,
    'window-open-v2': SAC,
    'window-close-v2': SAC
}



#TODO choose action based on the environment
# continuous action space -> continuous SAC ig
# discrete action space -> discrete SAC
def choose_agent(opt, env_name, env):
    if isinstance(env.action_space, gym.spaces.Box):
        #TODO return agent that has a continuous policy
        return SAC(**vars(opt))
    elif isinstance(env.action_space, gym.spaces.Discrete):
        return SACD_agent(**vars(opt))
    else:
        raise NotImplementedError("Actor selection of this kind is not implemented yet")

