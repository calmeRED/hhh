# -*- coding: utf-8 -*-
# trainer.py (modified Trainer implementation)
import os
import pdb
import pickle
import warnings
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
from datetime import datetime
from pprint import pprint as pp
from tqdm import trange

from buffer.per_replay_buffer import MultiAgentExclusivePER
from env.dummyenv import DummyEnv
from env.fmu_env_itms import FMUITMS
from model.mlp_block import MLPModel
from model.maddpg import MADDPG
from utils.utils_env import fill_observation, construct_action_dict, fill_list_with_dict, scale_actions
from utils.utils_i2c import generate_msg_observation
from utils.utils_misc import C_to_K, K_to_C, press_scroll_lock
from utils.utils_reward import RewardCalculator

from buffer.replay_buffer import ReplayBuffer
from buffer.kl_buffer import KLBuffer
from utils.utils_klvalue import get_kl_value, build_kl_sample
from utils.utils_config import get_config
from config.base_config import config


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        log_folder = config["log_folder"]
        # 增加时间戳子目录，避免覆盖
        self.log_dir = os.path.join(log_folder, datetime.now().strftime("%Y%m%d-%H%M"))
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # 是否启用 I2C 模块（prior + message）
        self.use_i2c = config["use_i2c"]
        self.use_agent_buffer = config["use_agent_buffer"]

        # 初始化 buffers / networks
        if self.use_i2c:
            # prior nets (每个 agent 一个)
            self.prior_nets = torch.nn.ModuleList([
                MLPModel(input_dim=config["state_dims"][i] + config["n_agents"],
                         num_outputs=2,
                         num_layers=config["i2c_num_layers"],
                         hidden_dim=config["i2c_hidden_dim"]).to(self.device)
                for i in range(config["n_agents"])
            ])
            self.prior_nets_optimizers = [
                torch.optim.Adam(pn.parameters(), lr=config["prior_lr"])
                for pn in self.prior_nets
            ]
            # KL 缓冲区, 每个 agent 一个
            self.prior_buffers = []
            self.obs_onehot_dim = len(self.config["obs_dict"])
            for agent_i in range(self.obs_onehot_dim):
                obs_dim = len(self.config["obs_dict"][agent_i])
                buf = KLBuffer(
                    buffer_size=self.config["prior_buffer_size"],
                    obs_dim=obs_dim,
                    obs_onehot_dim=self.obs_onehot_dim,
                    percentile=self.config["prior_buffer_percentile"]
                )
                self.prior_buffers.append(buf)

            # message nets: 每个 agent 输入为所有其他 agent 的 obs concat
            self.message_nets = [
                MLPModel(sum(len(self.config["obs_dict"][ii]) for ii in range(self.config["n_agents"])),
                         self.config["message_feature_dim"]).to(self.device)
                for agent_i in range(self.config["n_agents"])
            ]
        else:
            self.prior_nets = None
            self.prior_nets_optimizers = None
            self.prior_buffers = None
            self.message_nets = None

        # MADDPG 初始化：如果 use_i2c, 增加 message feature 到 obs dim
        if self.use_i2c:
            enhanced_obs_dims = self.config["enhanced_obs_dims"]
            self.maddpg = MADDPG(
                device=self.device,
                actor_lr=config["actor_lr"],
                critic_lr=config["critic_lr"],
                hidden_dim=config["hidden_dim"],
                obs_dims=enhanced_obs_dims,
                action_dis_dims=config["action_dis_dims"],
                action_con_dims=config["action_con_dims"],
                critic_input_dim=config["critic_input_dim"],
                gamma=config["gamma"],
                tau=config["tau"]
            )
            # 把 message_nets 参数并入 actor optimizer（若 MADDPG 的 agent 有 actor_optimizer）
            for i, agent in enumerate(self.maddpg.agents):
                agent.actor_optimizer = torch.optim.Adam(
                    list(agent.actor.parameters()) + list(self.message_nets[i].parameters()),
                    lr=self.config["actor_lr"]
                )
        else:
            self.maddpg = MADDPG(
                device=self.device,
                actor_lr=config["actor_lr"],
                critic_lr=config["critic_lr"],
                hidden_dim=config["hidden_dim"],
                obs_dims=config["obs_dims"],
                action_dis_dims=config["action_dis_dims"],
                action_con_dims=config["action_con_dims"],
                critic_input_dim=config["critic_input_dim"],
                gamma=config["gamma"],
                tau=config["tau"]
            )

        # replay buffer：使用仓库内的 ReplayBuffer（可能签名不同，下面做兼容处理）
        self.replay_buffer = ReplayBuffer(self.config["buffer_size"])

        # 其它组件
        self.reward_calculator = RewardCalculator(
            config["T_cabin_set"][0],
            config["T_bat_set"][0],
            config["T_motor_set"][0]
        )
        observation_list = [item for d in [config["obs_dict"], config["action_con_str_dict"], config["action_dis_str_dict"], [config["reward_dict"]]] for sublist
                            in d for item in sublist if isinstance(item, str) and item != "T_epsilon"]


        if config.get("fmu_path", False):
            self.env = FMUITMS(fmu_path=config["fmu_path"], step_size=config["fmu_step_size"])
            self.config["obs_dict"][0][0] = self.config["T_cabin_set"][0]
            self.config["obs_dict"][2][0] = self.config["T_bat_set"][0]
            self.config["obs_dict"][2][1] = self.config["T_motor_set"][0]

        else: # dummy
            self.env = DummyEnv(observation_list, config["action_con_str_dict"], config["action_dis_str_dict"], config["action_bounds"])
        self.step_count = 0
        self.episode_count = 0

        # tqdm 进度条引用（run() 中创建，并用于 log_metrics）
        self.pbar = None

        # 内部日志统计槽
        self.metrics = {}

    def get_hard_labels(self):
        """
        从 replay buffer 中采样若干 transitions，用当前 maddpg + prior 计算 KL，并保存到 prior buffer 中。
        返回一个 bool list 指示每个 prior_buffer 是否已满（可用于后续训练 prior）。
        """
        # 采样
        # print("getting labels")
        states, actions_dict, _, _, _ = self.replay_buffer.sample(self.config["prior_buffer_size"])
        per_agent_obs = []
        per_agent_actions = []
        n_agents = self.config["n_agents"]
        for agent_i in range(n_agents):
            agent_obs = [states[b][agent_i] for b in range(self.config["prior_buffer_size"])]
            agent_acts = [
                [actions_dict[b][k] for k in self.config["action_merged_dict"][agent_i]]
                for b in range(self.config["prior_buffer_size"])
            ]
            per_agent_obs.append(torch.FloatTensor(np.vstack(agent_obs)).to(self.device))
            per_agent_actions.append(torch.FloatTensor(np.vstack(agent_acts)).to(self.device))

        is_full_list = [False] * len(self.prior_buffers)
        # states 可能是 numpy array，里面每个 entry 是一个每-agent observation 列表/array
        for i in trange(len(states)):
            for agent_i in range(len(self.prior_buffers)):
                with torch.no_grad():
                    obs_i_rep, comm_id, KL_vals = get_kl_value(
                        agents=self.maddpg.agents,
                        obs_n=per_agent_obs,
                        act_n=per_agent_actions,
                        agent_i=agent_i,
                        action_bounds=self.config["action_bounds"],
                        action_sep_num=self.config["action_sep_num"],
                        action_merged_dict=self.config["action_merged_dict"],
                        temperature=self.config["lambda_temp"]
                    )
                obs_inputs, obs_onehot_inputs, KL_values = build_kl_sample(obs_i_rep, comm_id, KL_vals)
                # 将样本写入 prior buffer
                is_full = self.prior_buffers[agent_i].insert(
                    obs_inputs=obs_inputs.cpu().numpy(),  # [B, obs_dim_i]
                    obs_onehot_inputs=obs_onehot_inputs.cpu().numpy(),
                    KL_values=KL_values.cpu().numpy()
                )
                is_full_list[agent_i] = is_full
        return is_full_list

    def update_prior(self):
        """
        在 prior_buffers 足够满之后训练 prior_nets（分类器），采用交叉熵损失
        """
        if not self.use_i2c:
            return

        for agent_i in range(len(self.prior_buffers)):
            total_loss = 0.0
            total_cnt = 0
            for _ in trange(self.config["prior_train_iter"]):
                buffer = self.prior_buffers[agent_i]
                # 若 buffer 没有 __len__ 或长度小于 batch，跳过
                cur_pbuffer_len = buffer.step if buffer.step > 0 else (buffer.num_steps if np.any(buffer.labels) else 0)
                if cur_pbuffer_len < self.config["prior_train_batch_size"]:
                    continue

                obs_inputs, obs_onehot_inputs, labels = buffer.get_samples(self.config["prior_train_batch_size"])
                obs_inputs = torch.tensor(obs_inputs, dtype=torch.float32, device=self.device)
                obs_onehot_inputs = torch.tensor(obs_onehot_inputs, dtype=torch.float32, device=self.device)
                labels = torch.tensor(labels, dtype=torch.long, device=self.device)  # [B]
                prior_net = self.prior_nets[agent_i]
                optimizer = self.prior_nets_optimizers[agent_i]
                logits = prior_net(torch.cat([obs_inputs, obs_onehot_inputs], dim=-1))  # [B, 2]
                loss = F.cross_entropy(logits, labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_cnt += 1
            agent_i_loss_avg = total_loss / total_cnt if total_cnt > 0 else 0.0
            # 记录到 TensorBoard
            self.writer.add_scalar(f"prior/agent_{agent_i}_loss", agent_i_loss_avg, self.step_count)

    @torch.no_grad()
    def take_action(self, all_obs, explore=True, training=True):
        """
        all_obs: per-agent observations，可能是 list/tuple 或 numpy array
        返回：与 construct_action_dict 兼容的动作表示（list of per-agent action arrays or tensors）
        说明：
          - 若使用 use_i2c，则先通过 prior_nets 计算 communication decisions，再由 message_nets 生成 message_feature，
            将 message_feature 拼接到各 agent 的 obs 上并传给 maddpg 获取动作
          - maddpg.act 或 maddpg.get_actions 的接口并不统一：这里先尝试调用 maddpg.act，否则尝试 maddpg.get_actions 或各 agent 的 act
        """
        # 标准化 obs 格式为 list of per-agent torch tensors [batch dim = None => single sample]
        obs_list = []
        for o in all_obs:
            arr = np.asarray(o, dtype=np.float32)
            obs_list.append(torch.tensor(arr, dtype=torch.float32, device=self.device)) # .unsqueeze(0)

        # message features
        msg_features = [None] * len(obs_list)
        maddpg_inputs = []
        if self.use_i2c:
            # generate_msg_observation 接受 prior_nets 与 obs batch
            prior_inputs, _ = generate_msg_observation(self.prior_nets, obs_list, self.device)  # , for_replay=False
            # prior_inputs: per-agent observations for message_nets
            for i in range(len(obs_list)):
                # message_nets[i] 期待输入 size = sum(others obs dim)
                msg_features[i] = self.message_nets[i](prior_inputs[i])
            # 准备输入给 MADDPG：拼接 message feature 到每个 agent 的 obs（如果有）
            for i, o in enumerate(obs_list):
                input_i = torch.cat([o, msg_features[i][0]], dim=-1) # HACK [0]
                maddpg_inputs.append(input_i)
        else:
            maddpg_inputs = obs_list
            # 尝试调用 maddpg 的动作接口
        actions = self.maddpg.take_action(maddpg_inputs, explore=explore)

        # actions 现在为 list of np arrays or torch tensors: 转成 list of numpy arrays
        actions_out = []
        for a in actions:
            if isinstance(a, torch.Tensor):
                a = a.squeeze(0).cpu().numpy()
            actions_out.append(np.asarray(a))

        return actions_out

    def generate_init_dict(self, spec,seed=None):
        if seed is not None:np.random.seed(seed)
        out={}
        for k,(lo,hi,st) in spec.items():
            lo=float(lo)
            hi=float(hi)
            st=float(st)
            if lo>hi or st<=0:
                out[k] = lo
                continue
            n=int(np.floor((hi-lo)/st))+1
            if n<=0:raise ValueError
            idx=np.random.randint(0,n)
            v=lo+idx*st
            if np.isclose(st,round(st)):v=float(int(round(v)))
            out[k]=v
        return out

    def run_episode(self):
        """
        运行一个 episode：
         - env.reset, step loop
         - 收集 transitions 到 replay buffer（兼容两种 add 签名）
         - 在 replay buffer 足够时做 MADDPG 更新（batch）, 并在合适时训练 prior/message（two-stage）
         - 记录 metrics、更新 step_count/episode_count
        返回：episode_reward (list)
        """
        _spec = self.generate_init_dict(self.config["env_reset_dict"])
        if isinstance(self.env, DummyEnv):
            obs_raw = self.env.reset()
        else:
            obs_raw = self.env.reset(_spec)
        episode_reward = [0.0] * self.config["n_agents"]
        done = False

        for step in trange(self.config["episode_iter"]):
            obs = fill_observation(self.config["obs_dict"], obs_raw)
            # take_action 期望 per-agent obs list
            actions = self.take_action(obs, explore=True, training=True)
            # 构造动作 dict 并缩放
            action_dict = construct_action_dict(actions, self.config["action_con_str_dict"], self.config["action_dis_str_dict"])
            action_dict = scale_actions(action_dict, self.config["action_bounds"])
            next_obs_raw, term, trunc = self.env.step(action_dict)
            done = any((term, trunc))

            rewards = [
                self.reward_calculator.calculate_cabin_reward(obs_raw["cabinVolume.summary.T"], obs_raw["TableDC3.Pe"]),
                self.reward_calculator.calculate_refrigerant_reward(obs_raw["TableDC.Pe"]),
                self.reward_calculator.calculate_coolant_reward(
                    obs_raw["battery.Batt_top[1].T"],
                    obs_raw["machine.heatCapacitor.T"],
                    obs_raw["TableDC1.Pe"],
                    obs_raw["TableDC2.Pe"]
                )
            ]

            _fill_next_obs = fill_observation(self.config["obs_dict"], next_obs_raw)
            if done:
                # 如果 done，则设为 0 向量，避免后续 NN 输入异常
                _fill_next_obs = [np.zeros_like(np.asarray(x)) for x in _fill_next_obs]

            # 将数据写入 replay buffer：兼容 add 接口
            # 我们将 state/action/reward/next_state/done 结构化写入
            self.replay_buffer.add(obs , action_dict, rewards, _fill_next_obs, [True] * self.config["n_agents"] if done else [False] * self.config["n_agents"])
            obs_raw = next_obs_raw
            for i, r in enumerate(rewards):
                episode_reward[i] += r

            # 训练 MADDPG：当 buffer 达到 batch 大小时
            if self.replay_buffer.size() >= self.config["batch_size"]:
                states_b, actions_dict_b, rewards_b, next_states_b, dones_b = self.replay_buffer.sample(self.config["batch_size"])
                # 将样本转换成适合 maddpg.update 的格式
                # 非常依赖 maddpg.update 的实现：这里将每项转换为 per-agent torch tensors
                # states_b: [B, n_agents, obs_dim] 或 [B, n_agents, ...]
                # 我们尝试把它转换成 list of per-agent tensors [agent_i -> tensor(B, obs_dim_i)]
                # 如果 states_b 是 np.array with shape (B, n_agents, obs_dim_i)
                B = len(states_b)
                # convert lists -> numpy arrays if needed
                if isinstance(states_b, np.ndarray):
                    states_arr = states_b
                else:
                    states_arr = np.asarray(states_b, dtype=object)
                # build per-agent lists
                per_agent_obs = []
                per_agent_next_obs = []
                per_agent_actions = []
                per_agent_rewards = []
                per_agent_dones = []
                n_agents = self.config["n_agents"]
                # Try to handle typical shape: states_b[B][n_agents][obs_vec]
                for agent_i in range(n_agents):
                    agent_obs = [states_b[b][agent_i] for b in range(B)]
                    agent_next_obs = [next_states_b[b][agent_i] for b in range(B)]
                    agent_acts = [
                        [actions_dict_b[b][k] for k in self.config["action_merged_dict"][agent_i]]
                        for b in range(B)
                    ]
                    # agent_acts = [actions_b[b][agent_i] for b in range(B)]

                    agent_rews = [rewards_b[b][agent_i] for b in range(B)]
                    agent_dns = [dones_b[b][agent_i] for b in range(B)]
                    per_agent_obs.append(torch.FloatTensor(np.vstack(agent_obs)).to(self.device))
                    per_agent_next_obs.append(torch.FloatTensor(np.vstack(agent_next_obs)).to(self.device))
                    # actions might be dicts or vectors: convert to tensor if numeric
                    per_agent_actions.append(torch.FloatTensor(np.vstack(agent_acts)).to(self.device))
                    per_agent_rewards.append(torch.FloatTensor(np.asarray(agent_rews, dtype=np.float32)).unsqueeze(-1).to(self.device))
                    per_agent_dones.append(torch.FloatTensor(np.asarray(agent_dns, dtype=np.float32)).unsqueeze(-1).to(self.device))

                # 如果 use_i2c: 需要生成 message features for the batch
                msg_obs_batch = None
                per_agent_enhanced_obs = []
                if self.use_i2c:
                    msg_obs_batch, _ = generate_msg_observation(self.prior_nets, per_agent_obs, self.device) # , for_replay=False

                # 对每个 agent 进行 update（assume maddpg.update takes a batch and agent index）
                all_td_errors = [[] for _ in range(self.config["n_agents"])]
                agent_stats_list = []
                per_agent_enhanced_obs = []
                if self.use_i2c:
                    for agent_i in range(self.config["n_agents"]):
                        _obs_i = per_agent_obs[agent_i]
                        _msg_feature_i = self.message_nets[agent_i](msg_obs_batch[agent_i])
                        _enhanced_obs_i = torch.cat([_obs_i, _msg_feature_i], dim=-1)
                        per_agent_enhanced_obs.append(_enhanced_obs_i)
                else:
                    per_agent_enhanced_obs = per_agent_obs
                for agent_i in range(self.config["n_agents"]):
                    # call maddpg.update, which we assume returns (td_errors, critic_loss, actor_loss, agent_stats)
                    rt, agent_stats = self.maddpg.update([per_agent_obs,
                                                                  per_agent_enhanced_obs,
                                                                  per_agent_actions,
                                                                  per_agent_rewards,
                                                                  per_agent_next_obs,
                                                                  per_agent_dones],
                                                                  agent_i)
                    all_td_errors[agent_i] = rt
                    agent_stats_list.append(agent_stats)
                # 更新 target networks
                self.maddpg.update_all_targets()
                self.step_count += 1
                metrics = self.build_metrics(
                    episode_reward=episode_reward,
                    all_td_errors=all_td_errors,
                    agent_stats_list=agent_stats_list,
                    actions=actions,
                    prior_last_losses=getattr(self, "prior_last_losses", None),
                    prior_last_accs=getattr(self, "prior_last_accs", None),
                )
                self.log_metrics(metrics, global_step=self.step_count, mode="marl")
            if done:
                print("done!")
                break
        if self.use_i2c and self.replay_buffer.size() >= self.config["prior_buffer_size"] and self.episode_count % self.config["prior_update_frequency"] == 0 and self.episode_count > 1:
                print("updating prior.")
                is_full_list = self.get_hard_labels()
                if all(is_full_list):
                    self.update_prior()

        self.episode_count += 1
        # episode-level log
        self.writer.add_scalar("episode/avg_reward", float(np.mean(episode_reward)), self.episode_count)
        return episode_reward

    def evaluate_episode(self):
        """
        评估一个 episode（不探索）。返回 per-agent reward sums。
        """
        _spec = self.generate_init_dict(self.config["env_reset_dict"])
        if isinstance(self.env, DummyEnv):
            obs_raw = self.env.reset()
        else:
            obs_raw = self.env.reset(_spec)

        episode_reward = [0.0] * self.config["n_agents"]
        done = False

        for step in trange(self.config["episode_iter"]):
            obs = fill_observation(self.config["obs_dict"], obs_raw)
            actions = self.take_action(obs, explore=False, training=False)
            action_dict = construct_action_dict(actions, self.config["action_con_str_dict"], self.config["action_dis_str_dict"])
            action_dict = scale_actions(action_dict, self.config["action_bounds"])
            next_obs_raw, term, trunc = self.env.step(action_dict)
            done = any((term, trunc))

            rewards = [
                self.reward_calculator.calculate_cabin_reward(obs_raw["cabinVolume.summary.T"], obs_raw["TableDC3.Pe"]),
                self.reward_calculator.calculate_refrigerant_reward(obs_raw["TableDC.Pe"]),
                self.reward_calculator.calculate_coolant_reward(
                    obs_raw["battery.Batt_top[1].T"],
                    obs_raw["machine.heatCapacitor.T"],
                    obs_raw["TableDC1.Pe"],
                    obs_raw["TableDC2.Pe"]
                )
            ]
            for i, r in enumerate(rewards):
                episode_reward[i] += r

            obs_raw = next_obs_raw
            if done:
                break

        return episode_reward

    def run(self):
        """
        主训练循环：按 episode 迭代，使用 tqdm 显示进度，定期评估、保存 checkpoint 与 buffer
        """
        num_episodes = self.config["num_episodes"]
        eval_interval = self.config["eval_interval"]
        save_interval = self.config["save_interval"]
        # self.pbar = tqdm(range(num_episodes), desc="Training episodes")
        for ep in range(num_episodes): #self.pbar:
            ep_reward = self.run_episode()
            # 每 eval_interval 做一次评估
            if (self.episode_count % eval_interval) == 0:
                eval_reward = self.evaluate_episode()
                # 记录 eval 到 tensorboard
                self.writer.add_scalar("eval/avg_reward", float(np.mean(eval_reward)), self.episode_count)

            # 保存 checkpoint
            if (self.episode_count % save_interval) == 0:
                self.save_checkpoint(os.path.join(self.log_dir, f"checkpoint_{self.episode_count}.pth"))
                # 同时保存 buffer
                buf_path = os.path.join(self.log_dir, f"replay_buffer_{self.episode_count}.pkl")
                self.save_buffer(buf_path)

            # 更新 tqdm 后缀：利用 metrics
            if self.pbar:
                self.pbar.set_postfix({
                    "episode": self.episode_count,
                    "avg_reward": float(np.mean(ep_reward)),
                    "replay": self.replay_buffer.size()
                })
        # 训练结束
        self.writer.flush()
        self.writer.close()
        if self.pbar:
            self.pbar.close()

    def build_metrics(self,
                      episode_reward,
                      all_td_errors=None,
                      agent_stats_list=None,
                      actions=None,
                      prior_last_losses=None,
                      prior_last_accs=None,
                      extra=None):
        """
        构造用于 log_metrics(...) 的 metrics 字典。
        参数:
          - episode_reward: list or array, per-agent cumulative reward for current episode
          - all_td_errors: list (len=n_agents) of lists/arrays of td errors returned from updates (可为 None)
          - agent_stats_list: list (len=n_agents) of per-agent stats (dict or tuple) returned by maddpg.update (可为 None)
          - actions: list (len=n_agents) of last actions (numpy array or torch tensor) (可为 None)
          - prior_last_losses: list of prior losses for each prior (可为 None)
          - prior_last_accs: list of prior accuracies for each prior (可为 None)
          - extra: dict of any other scalars/text to include under "others"
        返回:
          metrics dict, 结构符合 log_metrics 的输入。
        """
        import numpy as _np
        metrics = {
            "scalars": {},
            "per_agent": {},
            "prior": {},
            "replay": {},
            "others": {}
        }

        # 基本 scalars
        avg_reward = None
        avg_reward = float(_np.mean(episode_reward)) if episode_reward is not None else 0.0
        metrics["scalars"]["avg_reward"] = avg_reward
        metrics["scalars"]["episode"] = int(getattr(self, "episode_count", 0))
        metrics["scalars"]["step"] = int(getattr(self, "step_count", 0))

        # replay size
        metrics["replay"]["size"] = int(self.replay_buffer.size())


        # 确定 agent 数量
        n_agents = int(self.config.get("n_agents", len(episode_reward) if episode_reward is not None else 0))

        # per-agent entries
        for i in range(n_agents):
            per = {
                "actor_loss": None,
                "critic_loss": None,
                "td_error": None,
                "reward": None,
                "action": None,
                "action_mean": None,
                "action_std": None
            }

            # reward
            per["reward"] = float(episode_reward[i]) if (episode_reward is not None and i < len(episode_reward)) else None

            # td_error mean
            if all_td_errors is not None and i < len(all_td_errors):
                arr = _np.asarray(all_td_errors[i], dtype=_np.float32)
                per["td_error"] = float(arr.mean()) if arr.size else None


            # agent_stats_list: 优先从中读取 actor/critic loss
            if agent_stats_list is not None and i < len(agent_stats_list):
                stats = agent_stats_list[i]
                if isinstance(stats, dict):
                    per["actor_loss"] = stats.get("actor_loss", stats.get("actor_loss_mean", per["actor_loss"]))
                    per["critic_loss"] = stats.get("critic_loss", stats.get("critic_loss_mean", per["critic_loss"]))
                elif isinstance(stats, (list, tuple)):
                    # 常见场景: stats 包含 numeric tuples (actor_loss, critic_loss, ...)
                    if len(stats) >= 1:
                        per["actor_loss"] = float(stats[0])
                    if len(stats) >= 2:
                        per["critic_loss"] = float(stats[1])

            # action info
            if actions is not None and i < len(actions):
                a = actions[i]
                # torch tensor -> numpy
                if isinstance(a, torch.Tensor):
                    a = a.detach().cpu().numpy()

                a = _np.asarray(a, dtype=_np.float32)
                per["action"] = a
                if a.size:
                    per["action_mean"] = float(a.mean())
                    per["action_std"] = float(a.std())

            metrics["per_agent"][i] = per

        # aggregate actor/critic losses（平均）
        actor_losses = [v.get("actor_loss") for v in metrics["per_agent"].values() if v.get("actor_loss") is not None]
        critic_losses = [v.get("critic_loss") for v in metrics["per_agent"].values() if
                         v.get("critic_loss") is not None]
        if actor_losses:
            metrics["scalars"]["mean_actor_loss"] = float(_np.mean(actor_losses))
        if critic_losses:
            metrics["scalars"]["mean_critic_loss"] = float(_np.mean(critic_losses))

        # prior info (if applicable)
        if getattr(self, "use_i2c", False):
            n_priors = len(getattr(self, "prior_buffers", []))
            for i in range(n_priors):
                pl = None
                pa = None
                if prior_last_losses is not None and i < len(prior_last_losses):
                    pl = float(prior_last_losses[i]) if prior_last_losses[i] is not None else None
                if prior_last_accs is not None and i < len(prior_last_accs):
                    pa = float(prior_last_accs[i]) if prior_last_accs[i] is not None else None
                metrics["prior"][i] = {"loss": pl, "accuracy": pa}
        return metrics

    def log_metrics(self, metrics: dict = None, global_step: int = None, mode: str = "marl"):
        if mode == "marl":
            if metrics is None:
                metrics = {}
            step = global_step if (global_step is not None) else (self.step_count if hasattr(self, "step_count") else None)
            # 频率控制：histogram 不需要每 step 写入，减少 IO 压力
            histogram_freq = self.config.get("histogram_freq", 100)  # 每 100 steps 写一次 histogram
            write_hist = (histogram_freq > 0 and step is not None and (step % histogram_freq == 0))
            # 1) 普通 scalars（扁平化记录）
            scalars = metrics.get("scalars", {})
            for k, v in scalars.items():
                self.writer.add_scalar(k, float(v), step)
            # 2) per-agent metrics
            per_agent = metrics.get("per_agent", {})
            for agent_idx, d in per_agent.items():
                prefix = f"train/agent_{agent_idx}"
                # scalar items
                for k, v in d.items():
                    if k == "action":
                        # action 可以是 numpy array / list / tensor：记录 mean/std 和 histogram（可选）
                        a = np.asarray(v)
                        if a.size == 0:
                            continue
                        # 记录均值与 std
                        self.writer.add_scalar(f"{prefix}/action_mean", float(a.mean()), step)
                        self.writer.add_scalar(f"{prefix}/action_std", float(a.std()), step)
                        # 记录 histogram（按频率）
                        if write_hist:
                            self.writer.add_histogram(f"{prefix}/action_hist", a, step)

                    elif k in ("actor_loss", "critic_loss", "td_error", "reward", "entropy"):
                        self.writer.add_scalar(f"{prefix}/{k}", float(v), step)
                    elif k.startswith("grad_") or k.startswith("weight_"):
                        # 梯度/权重范数
                        self.writer.add_scalar(f"{prefix}/{k}", float(v), step)
                    else:
                        # 其他标量一律尝试转 float
                        self.writer.add_scalar(f"{prefix}/{k}", float(v), step)
        if mode == "prior":
            # 3) prior/message nets metrics
            prior = metrics.get("prior", {})
            for agent_idx, d in prior.items():
                prefix = f"prior/agent_{agent_idx}"
                for k, v in d.items():
                    # 支持 loss、accuracy、kl_mean 等
                    self.writer.add_scalar(f"{prefix}/{k}", float(v), step)


        # 4) replay / buffer related
        replay = metrics.get("replay", {})
        for k, v in replay.items():
            self.writer.add_scalar(f"replay/{k}", float(v), step)


        # 5) 其它自定义 scalars
        others = metrics.get("others", {})
        for k, v in others.items():
            self.writer.add_scalar(k, float(v), step)


        # 6) 更新 tqdm postfix（只显示少量关键项）
        if self.pbar is not None:
            postfix_keys = self.config.get("tqdm_postfix_keys", ["avg_reward", "replay_size", "mean_actor_loss", "mean_critic_loss", "mean_td_error", "mean_action_norm", "mean_policy_std"])
            postfix = {}
            # avg_reward 来源优先 metrics.scalars 再 per_agent reward 平均
            if "avg_reward" in scalars:
                postfix["avg_reward"] = float(scalars["avg_reward"])
            else:
                # compute from per_agent rewards if present
                agent_rewards = [d.get("reward") for _, d in per_agent.items() if d.get("reward") is not None]
                postfix["avg_reward"] = float(np.mean(agent_rewards)) if len(agent_rewards) else None

            # replay_size
            if "size" in replay:
                postfix["replay_size"] = int(replay["size"])
            elif "replay_size" in scalars:
                postfix["replay_size"] = int(scalars["replay_size"])

            # mean_actor_loss: average across agents if present
            actor_losses = []
            for _, d in per_agent.items():
                if "actor_loss" in d:
                    actor_losses.append(float(d["actor_loss"]))

            if actor_losses:
                postfix["mean_actor_loss"] = float(np.mean(actor_losses))

            # only keep keys listed in postfix_keys and present
            filtered = {k: v for k, v in postfix.items() if k in postfix_keys and v is not None}
            # attach episode/step
            filtered["ep"] = self.episode_count
            filtered["step"] = self.step_count
            self.pbar.set_postfix(filtered)


        # flush writer occasionally (避免过长延迟)
        self.writer.flush()

    def save_checkpoint(self, path: str):
        """
        保存网络与训练状态到 path（.pth）
        """
        state = {
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "config": self.config,
        }
        # maddpg 保存：尽量保存 agents 的 actor/critic state_dict
        # 如果 MADDPG 提供 state_dict，则直接保存
        if hasattr(self.maddpg, "state_dict"):
            state["maddpg"] = self.maddpg.state_dict()
        else:
            # 否则逐 agent 保存 actor/critic
            agents_state = []
            for ag in self.maddpg.agents:
                ag_state = {}
                if hasattr(ag, "actor"):
                    ag_state["actor"] = ag.actor.state_dict()
                if hasattr(ag, "critic"):
                    ag_state["critic"] = ag.critic.state_dict()
                # 尝试保存 agent optimizer 状态
                if hasattr(ag, "actor_optimizer"):
                    ag_state["actor_optim"] = ag.actor_optimizer.state_dict()

                agents_state.append(ag_state)
            state["agents"] = agents_state


        # 保存 message_nets / prior_nets / optimizers
        if self.use_i2c:
            state["message_nets"] = [mn.state_dict() for mn in self.message_nets]
            state["prior_nets"] = [pn.state_dict() for pn in self.prior_nets]
            state["prior_optim"] = [opt.state_dict() for opt in self.prior_nets_optimizers]

        torch.save(state, path)
        self.writer.add_text("checkpoint", f"Saved checkpoint to {path}", self.episode_count)


    def load_checkpoint(self, path: str):
        """
        加载 checkpoint（.pth）并尝试恢复网络与训练状态
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = torch.load(path, map_location=self.device)
        self.step_count = data.get("step_count", self.step_count)
        self.episode_count = data.get("episode_count", self.episode_count)
        # 恢复 maddpg
        if "maddpg" in data and hasattr(self.maddpg, "load_state_dict"):
            self.maddpg.load_state_dict(data["maddpg"])
        elif "agents" in data:
            for ag, ag_state in zip(self.maddpg.agents, data["agents"]):
                if "actor" in ag_state and hasattr(ag, "actor"):
                    ag.actor.load_state_dict(ag_state["actor"])
                if "critic" in ag_state and hasattr(ag, "critic"):
                    ag.critic.load_state_dict(ag_state["critic"])
                if "actor_optim" in ag_state and hasattr(ag, "actor_optimizer"):
                    ag.actor_optimizer.load_state_dict(ag_state["actor_optim"])


        # 恢复 message/prior
        if self.use_i2c and "message_nets" in data:
            for mn, st in zip(self.message_nets, data["message_nets"]):
                mn.load_state_dict(st)
        if self.use_i2c and "prior_nets" in data:
            for pn, st in zip(self.prior_nets, data["prior_nets"]):
                pn.load_state_dict(st)
        if self.use_i2c and "prior_optim" in data:
            for opt, st in zip(self.prior_nets_optimizers, data["prior_optim"]):
                opt.load_state_dict(st)


    def save_buffer(self, path: str):
        """
        使用 pickle 保存 replay buffer 和 prior_buffers（如存在）。
        保存结构为 dict:
        {
          "replay_buffer": <list-of-transitions> 或 <raw-object fallback>,
          "prior_buffers": [<list-or-raw> ...]   # 每个 prior buffer 的可序列化内容
        }
        """
        data = {}
        # 1) replay buffer 内容优先取 .buffer（deque -> list），否则序列化整个对象
        if hasattr(self.replay_buffer, "buffer"):
            # 将 deque 转为 list 以提高可移植性
            data["replay_buffer"] = list(self.replay_buffer.buffer)
        else:
            # 直接尝试把整个对象转换为可序列化的形式（可能失败）
            data["replay_buffer"] = self.replay_buffer

        # 2) prior_buffers（如果存在）: 对每个 prior buffer 尝试提取内部可序列化字段
        prior_list = []
        if getattr(self, "prior_buffers", None) is not None:
            for pb in self.prior_buffers:
                serialized = None
                # 优先尝试 pb.buffer / pb.data / pb._buffer
                for attr in ("buffer", "data", "_buffer", "_data", "items"):
                    if hasattr(pb, attr):
                        attr_val = getattr(pb, attr)
                        # deque -> list
                        if isinstance(attr_val, (deque, list)):
                            serialized = list(attr_val)
                        else:
                            # 尝试转换为 list，如果失败则直接赋值（pickle 可能仍能处理）
                            serialized = list(attr_val)
                        break
                # fallback: 如果 prior buffer 有 get_state 方法，使用它
                if serialized is None and hasattr(pb, "get_state"):
                    serialized = pb.get_state()

                # 最后 fallback：尽量 pickle 整个对象
                if serialized is None:
                    serialized = pb

                prior_list.append(serialized)
        data["prior_buffers"] = prior_list

        # 写入文件
        with open(path, "wb") as f:
            pickle.dump(data, f)
        self.writer.add_text("buffer", f"Saved replay+prior buffers to {path}", self.episode_count)


    def load_buffer(self, path: str):
        """
        从 pickle 恢复 replay buffer 和 prior_buffers（如果保存了）。
        尝试把保存的数据写回到现有的 self.replay_buffer 和 self.prior_buffers（索引对应）。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        # 1) 恢复 replay_buffer
        saved_replay = data.get("replay_buffer", None)
        if saved_replay is not None:
            # 若当前 replay_buffer 有 .buffer 属性并且 saved_replay 是可迭代的 list，则恢复为 deque
            if hasattr(self.replay_buffer, "buffer") and isinstance(saved_replay, (list, tuple)):
                maxlen = getattr(self.replay_buffer.buffer, "maxlen", None)
                self.replay_buffer.buffer = deque(saved_replay, maxlen=maxlen)
            else:
                # 直接替换对象（当无法将数据写回到现有结构时）
                self.replay_buffer = saved_replay
        self.writer.add_text("buffer", f"Loaded replay buffer from {path}", self.episode_count)

        # 2) 恢复 prior_buffers（若存在）
        saved_priors = data.get("prior_buffers", None)
        if saved_priors is not None and getattr(self, "prior_buffers", None) is not None:
            # 对应索引恢复：如果数量不匹配，按最小长度恢复；多余的保存数据会被忽略
            n_restore = min(len(saved_priors), len(self.prior_buffers))
            for i in range(n_restore):
                pb_saved = saved_priors[i]
                target_pb = self.prior_buffers[i]
                restored = False
                # 若 target_pb 有 buffer 属性并保存的数据是 list，则恢复为 deque
                if hasattr(target_pb, "buffer") and isinstance(pb_saved, (list, tuple)):
                    maxlen = getattr(target_pb.buffer, "maxlen", None)
                    target_pb.buffer = deque(pb_saved, maxlen=maxlen)
                    restored = True

                # 若 target_pb 提供 set_state 接口
                if (not restored) and hasattr(target_pb, "set_state"):
                    target_pb.set_state(pb_saved)
                    restored = True

                # 若 target_pb 有兼容字段 data 或 items，则尝试赋值
                if (not restored):
                    for attr in ("data", "_data", "_buffer", "items"):
                        if hasattr(target_pb, attr):
                            setattr(target_pb, attr, pb_saved)
                            restored = True
                            break
                # 最后 fallback: 如果 pb_saved 看起来是整个对象，则尽量替换该 prior_buffer（注意类型）
                if (not restored) and isinstance(pb_saved, type(target_pb)):
                    self.prior_buffers[i] = pb_saved
                    restored = True
                if not restored:
                    warnings.warn(f"Could not fully restore prior_buffers[{i}] from file; manual intervention may be required.")
            self.writer.add_text("buffer", f"Loaded prior buffers from {path}", self.episode_count)
        else:
            # saved_priors 为 None 或 self.prior_buffers 不存在
            if saved_priors is not None:
                warnings.warn("Saved prior_buffers exist in file but Trainer has no prior_buffers to restore into.")


if __name__ == "__main__":
    print("Trainer configuration preview:")
    my_config = get_config(config)
    for k in sorted(my_config.keys()):
        print(f"  {k}: {my_config[k]}")
    # ---- 创建 Trainer 并运行 ----
    trainer = Trainer(my_config)
    trainer.run()