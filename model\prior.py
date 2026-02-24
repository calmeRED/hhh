import pdb

import torch
import torch.nn as nn


class PriorNetwork(nn.Module):
    def __init__(self, obs_dim, n_agents, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, obs, agent_id):
        inp = torch.cat([obs, agent_id], dim=-1) # 自身已经属于某agent(i)了，再拼接一个agent_id编码(j)来决定ij是否通信(有边)
        pdb.set_trace()

        return self.net(inp).squeeze(-1)