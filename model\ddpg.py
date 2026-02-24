import pdb

import torch
from model.mlp_block import MLPModel
from utils.utils_model import onehot_from_logits, gumbel_softmax
class DDPG:
    """单智能体的 DDPG（Actor‑Critic）实现，适配离散动作空间。"""
    def __init__(self,
                 obs_dim,          # 状态向量维度
                 action_dis_dim,     # 离散动作数量（即 one‑hot 长度）
                 action_con_dim,     # 连续动作数量（后面可过 tanh）
                 critic_input_dim,   # Critic 输入维度 = sum(state) + sum(action_dis) + sum(action_con)
                 hidden_dim,
                 actor_lr,
                 critic_lr,
                 device):
        # ---------- Actor ----------
        # self.actor = TwoLayerFC(obs_dim, action_dim, hidden_dim).to(device)
        # self.target_actor = TwoLayerFC(obs_dim, action_dim,
        #                                hidden_dim).to(device)
        self.actor = MLPModel(input_dim=obs_dim, num_outputs=action_dis_dim+action_con_dim, num_layers=6, hidden_dim=hidden_dim).to(device)
        self.target_actor = MLPModel(input_dim=obs_dim, num_outputs=action_dis_dim+action_con_dim, num_layers=6, hidden_dim=hidden_dim).to(device)

        # ---------- Critic ----------
        # self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        # self.target_critic = TwoLayerFC(critic_input_dim, 1,
        #                                 hidden_dim).to(device)
        self.critic = MLPModel(input_dim=critic_input_dim, num_outputs=1, num_layers=6, hidden_dim=hidden_dim).to(device)
        self.target_critic = MLPModel(input_dim=critic_input_dim, num_outputs=1, num_layers=6, hidden_dim=hidden_dim).to(device)

        # 同步 target 参数（初始化时直接拷贝）
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        # ---------- Optimizer ----------
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ---------- 离散-连续动作数量 ----------
        self.action_dis_dim = action_dis_dim
        self.action_con_dim = action_con_dim

    # 7.1 采样动作
    # ------------------------------------------------------------------
    def take_action(self, state, explore=False):
        """
        给定单个智能体的状态张量，返回一个离散动作的 one‑hot 向量（numpy）。
        - explore=True  使用 Gumbel‑Softmax 进行随机探索（可微）。
        - explore=False 使用 ε‑greedy 的确定性 one‑hot。
        """
        action = self.actor(state)
        dis, con = action[:, :self.action_dis_dim], action[:, self.action_dis_dim:]
        if dis.numel() > 0:
            if explore: # 训练时使用soft
                # 通过 Gumbel‑Softmax 产生可微的离散采样
                dis = gumbel_softmax(dis)
            else:
                # 直接取最大 logits（并做 ε‑greedy）
                dis = onehot_from_logits(dis)
        con = torch.tanh(con)
        action = torch.cat([dis, con], dim=-1) # TODO -1维？
        # 返回 numpy 形式，去掉 batch 维度


        return action.detach().cpu().numpy()[0]

    # ------------------------------------------------------------------
    # 7.2 软更新（Polyak averaging）
    # ------------------------------------------------------------------
    def soft_update(self, net, target_net, tau):
        """
        将 net 参数以比例 tau 融合进 target_net（软更新）。
        target = (1 - tau) * target + tau * net
        """
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)