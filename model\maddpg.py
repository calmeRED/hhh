import pdb

import torch
from model.ddpg import DDPG
from utils.utils_model import onehot_from_logits, gumbel_softmax
class MADDPG:
    """
    Multi‑Agent DDPG 框架（论文《MADDPG》）。
    每个智能体都有独立的 Actor / Critic，
    Critic 的输入会拼接所有 agents 的状态和动作，
    实现 centralized training & decentralized execution。
    """

    def __init__(self,
                 # env,  # 用于获取 agents 数量
                 device,
                 actor_lr,
                 critic_lr,
                 hidden_dim,
                 obs_dims,  # list: 每个 agent 的状态维度
                 action_dis_dims,
                 action_con_dims,
                 critic_input_dim,  # 所有 agents 状态+动作的总维度
                 gamma,
                 tau):
        self.agents = []  # 存放每个智能体的 DDPG 实例
        for i in range(len(obs_dims)):
            self.agents.append(
                DDPG(obs_dims[i],
                     action_dis_dims[i],
                     action_con_dims[i],
                     critic_input_dim,
                     hidden_dim,
                     actor_lr,
                     critic_lr,
                     device)
            )
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新系数
        self.critic_criterion = torch.nn.MSELoss()  # TD‑error L2 loss
        self.device = device

    # ------------------------------------------------------------------
    # 8.1 只读属性，方便外部访问 Actor / Target‑Actor
    # ------------------------------------------------------------------
    @property
    def policies(self):
        """返回所有智能体的当前 Actor 网络（list）"""
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        """返回所有智能体的目标 Actor 网络（list）"""
        return [agt.target_actor for agt in self.agents]

    # ------------------------------------------------------------------
    # 8.2 统一的动作采样接口（对外部环境使用）
    # ------------------------------------------------------------------
    def take_action(self, states, explore):
        """
        为环境中的每个智能体采样一个动作。
        - states: list of raw numpy observations, 长度 = n_agents
        - explore: bool，是否开启探索（Gumbel‑Softmax）
        返回: list of one‑hot 动作（numpy），可直接喂给 env.step()
        """
        # 将每个观测转为 torch Tensor，保持 batch=1 维度。在强化学习中，神经网络模型通常要求输入是 批量形式（batched），即使你只是推理一个样本
        states = [
            states[i].unsqueeze(0) # HACK .detach().clone()
            for i in range(len(states))
        ]
        # 逐个智能体调用其 DDPG 的 take_action
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    # ------------------------------------------------------------------
    # 8.3 单个智能体的网络更新（一次梯度步）
    # ------------------------------------------------------------------
    def update(self, sample, i_agent):
        obs, enhanced_obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]
        # ---------- Critic 更新 ----------
        cur_agent.critic_optimizer.zero_grad()
        # 1) 计算目标网络的动作（使用目标 Actor + ε‑greedy/one‑hot）
        # HACK obs、next_obs额外加入了I2C的通信feature
        target_critic_value = None
        # 改动0203
        with torch.no_grad():
            all_target_act = [pi(enhanced_obs) for pi, enhanced_obs in zip(self.target_policies, enhanced_obs) ]
            # 2) 拼接 next_obs + all_target_act 作为目标 Critic 的输入
            target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
            # 3) 计算 TD‑target：r + γ * Q'(s', a')
            target_critic_value = rew[i_agent].view(-1, 1) + \
                                  self.gamma * cur_agent.target_critic(target_critic_input) * \
                                  (1 - done[i_agent].view(-1, 1))
        # 4) 当前 Critic 的估计 Q(s,a)
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        # 5) MSE loss 并反向传播
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        # print(f"critic_value: {critic_value}, target_critic_value.detach(): {target_critic_value.detach()}")
        ret_tderror = (critic_value.detach().cpu() - target_critic_value.detach().cpu())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()
        # ---------- Actor 更新 已区分连续离散----------
        cur_agent.actor_optimizer.zero_grad()
        # 1) 当前智能体的 Actor 前向得到 logits
        cur_actor_out = cur_agent.actor(enhanced_obs[i_agent])  # TODO使用prior生成的msg增强
        # 2) 对 logits 使用 Gumbel‑Softmax 产生可微的离散动作（用于后向计算）
        # 注意：这里需要区分离散和连续动作
        dis_logits, con_logits = cur_actor_out[:, :cur_agent.action_dis_dim], cur_actor_out[:, cur_agent.action_dis_dim:]
        # 离散动作处理
        if dis_logits.numel() > 0:
            dis_action = gumbel_softmax(dis_logits)
        else:
            dis_action = torch.zeros(0, device=cur_actor_out.device)
        # 连续动作处理
        con_action = torch.tanh(con_logits)  # 连续动作限制在[-1,1]范围内
        # 组合动作
        cur_act_vf_in = torch.cat([dis_action, con_action], dim=-1)
        # 3) 为所有智能体准备动作向量：
        #    - 本智能体使用上一步的软采样（保持可微）
        #    - 其他智能体使用确定性 one‑hot（不需要梯度）
        all_actor_acs = []
        for i, (pi, _enhanced_obs) in enumerate(zip(self.policies, enhanced_obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                # 对于其他智能体，需要正确处理离散和连续动作
                pi_output = pi(_enhanced_obs)
                dis_logits_other, con_logits_other = pi_output[:, :self.agents[i].action_dis_dim], pi_output[:, self.agents[i].action_dis_dim:]
                # 离散动作处理
                if dis_logits_other.numel() > 0:
                    dis_action_other = onehot_from_logits(dis_logits_other, eps=0.01)
                else:
                    dis_action_other = torch.zeros(0, device=pi_output.device)
                # 连续动作处理
                con_action_other = torch.tanh(con_logits_other)
                # 组合动作
                action_other = torch.cat([dis_action_other, con_action_other], dim=-1)
                # 改动0203
                all_actor_acs.append(action_other.detach())
        # 4) 拼接状态 + 所有动作，得到当前 Actor 的价值估计
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        # 5) Actor 的目标是最大化 Critic 的 Q 值（即最小化 -Q）
        actor_critic_value = cur_agent.critic(vf_in)
        actor_loss = -actor_critic_value.mean()
        # # 6) 为了提升数值稳定性，加上 L2 正则（权重衰减的简易实现）
        # actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        # 7) 添加梯度裁剪防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(cur_agent.actor.parameters(), max_norm=1.0)
        # 8) 反向传播并更新 Actor 参数
        actor_loss.backward()
        cur_agent.actor_optimizer.step()
        # for name, param in cur_agent.actor.named_parameters():
        #     if param.grad is not None:
        #         print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")
        #     else:
        #         print(f"  {name}: no grad")


        def check_gradients_and_outputs(agent, agent_name):
            # 检查 Critic 梯度
            critic_grad_norm = 0
            for param in agent.critic.parameters():
                if param.grad is not None:
                    critic_grad_norm += param.grad.data.norm(2).item() ** 2
            critic_grad_norm = critic_grad_norm ** 0.5

            # 检查 Actor 梯度
            actor_grad_norm = 0
            for param in agent.actor.parameters():
                if param.grad is not None:
                    actor_grad_norm += param.grad.data.norm(2).item() ** 2
            actor_grad_norm = actor_grad_norm ** 0.5

            return {
                'critic_grad_norm': critic_grad_norm,
                'actor_grad_norm': actor_grad_norm,
            }

        agent_stats = {}
        for i, agent in enumerate(self.agents):
            agent_stats[f'agent_{i}'] = check_gradients_and_outputs(agent, f'Agent {i}')
        agent_stats.update({"critic_loss":critic_loss.item(), "actor_loss":actor_loss.item()})
        return ret_tderror.squeeze(), agent_stats

    # ------------------------------------------------------------------
    # 8.4 所有智能体的软目标网络更新（每次训练结束后调用）
    # ------------------------------------------------------------------
    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)