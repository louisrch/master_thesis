import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from networks import Double_Q_Net, Policy_Net


class SACD_agent:
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        print(self.dvc)
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
                device=self.dvc
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


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.long, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def addAll(self, s_array, a_array, r_array, s_next_array, dw_array):
        # print(len(s_array), len(a_array), len(r_array), len(s_next_array))
        begin = self.ptr
        end = (self.ptr + len(a_array)) % self.max_size
        end_array = min(self.max_size, self.ptr + len(a_array))

        states = torch.from_numpy(np.stack(s_array, axis=0)).to(self.dvc)
        actions = torch.tensor(a_array).unsqueeze(-1).to(self.dvc)
        rewards = r_array.clone().detach().float().to(self.dvc)
        next_states = torch.from_numpy(np.stack(s_next_array, axis=0)).to(self.dvc)
        dws = torch.tensor(dw_array).unsqueeze(-1).to(self.dvc)

        if begin + len(a_array) <= self.max_size:
            slice_idx = slice(begin, begin + len(a_array))
            self.s[slice_idx] = states
            self.a[slice_idx] = actions
            self.r[slice_idx] = rewards
            self.s_next[slice_idx] = next_states
            self.dw[slice_idx] = dws
            self.ptr = (begin + len(a_array)) % self.max_size
            self.size = min(self.size + len(a_array), self.max_size)
        else:
            head = self.max_size - begin
            tail = len(a_array) - head

            self.s[begin:] = states[:head]
            self.a[begin:] = actions[:head]
            self.r[begin:] = rewards[:head]
            self.s_next[begin:] = next_states[:head]
            self.dw[begin:] = dws[:head]

            self.s[:tail] = states[head:]
            self.a[:tail] = actions[head:]
            self.r[:tail] = rewards[head:]
            self.s_next[:tail] = next_states[head:]
            self.dw[:tail] = dws[head:]
            self.ptr = tail

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
