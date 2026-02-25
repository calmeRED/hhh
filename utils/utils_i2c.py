import pdb
import torch
import torch.nn as nn
import numpy as np
from pprint import pprint
from model.mlp_block import MLPModel

@ torch.no_grad()
def generate_msg_observation(prior_nets, all_obs, device):
    """
    all_obs: list of [B, obs_dim_i] tensor or numpy
    return:
        msg_obs: list of [B, enhanced_dim_i]
        comm_masks  : list of [B, n_agents]
    """
    n_agents = len(all_obs)

    # ---- unify batch ----
    obs_tensors = []
    for obs in all_obs:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs_tensors.append(obs.to(device))

    B = obs_tensors[0].shape[0]

    def make_onehot(idx):
        onehot = torch.zeros(B, n_agents, device=device)
        onehot[:, idx] = 1.0
        return onehot

    msg_obs = []
    comm_masks = []

    for i in range(n_agents):
        obs_i = obs_tensors[i]  # [B, obs_dim_i]

        masked_parts = []
        comm_i = []

        for j in range(n_agents):
            obs_j = obs_tensors[j]
            if i == j:
                # masked_parts.append(torch.zeros_like(obs_j))
                masked_parts.append(torch.zeros_like(obs_i))
                comm_i.append(torch.zeros(B, device=device))
                continue

            onehot_j = make_onehot(j)

            # ---- PRIOR FORWARD (logits -> prob) ----
            logits = prior_nets[i](torch.cat([obs_i, onehot_j], dim=1))     # [B, 2]
            probs = torch.softmax(logits, dim=-1)       # [B, 2]

            communicate = probs[:, 0] > 0.5             # [B]

            selected = torch.where(
                communicate.unsqueeze(-1),
                obs_j,
                torch.zeros_like(obs_j)
            )

            masked_parts.append(selected)
            comm_i.append(communicate.float())

        global_masked = torch.cat(masked_parts, dim=-1)     # [B, sum obs_dim]
        # enhanced = torch.cat([obs_i, global_masked], dim=-1)

        msg_obs.append(global_masked)
        comm_masks.append(torch.stack(comm_i, dim=-1))      # [B, n_agents]
    return msg_obs, comm_masks







if __name__ == "__main__":
    # --- 1. 设置参数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_agents = 3  # 假设有3个智能体
    batch_size = 2  # 批次大小
    obs_dims = [1, 2, 3]  # 每个智能体的观测维度不同

    # --- 2. 构造 dummy_prior_nets ---
    # 创建一个包含 n_agents 个 PriorNet 的列表
    prior_nets = torch.nn.ModuleList([
        MLPModel(input_dim=obs_dims[i] + n_agents,
                 num_outputs=2,
                 num_layers=2,
                 hidden_dim=64).to(device)
        for i in range(n_agents)
    ])
    # --- 3. 构造 dummy_obs ---
    # 模拟观测数据：可以是 numpy 或 tensor
    dummy_obs = []
    for i in range(n_agents):
        # 生成随机数据
        obs_np = np.random.rand(batch_size, obs_dims[i]).astype(np.float32)
        # 这里可以模拟某些 batch 数据为 numpy，某些为 tensor
        if i % 2 == 0:
            dummy_obs.append(obs_np)  # numpy
        else:
            dummy_obs.append(torch.from_numpy(obs_np))  # tensor

    # --- 4. 运行函数 ---

    msg_obs, comm_masks = generate_msg_observation(
        prior_nets=prior_nets,
        all_obs=dummy_obs,
        device=device
    )

    # --- 5. 打印结果 ---
    print("\n--- 运行结果 ---")
    print(f"智能体数量: {n_agents}")
    print(f"批次大小: {batch_size}")

    for i in range(n_agents):
        print(f"\n智能体 {i}:")
        print(f"  原始观测形状: [B, {obs_dims[i]}]")
        print(f"  增强后观测形状: {msg_obs[i].shape}")
        print(f"  通信掩码形状: {comm_masks[i].shape}")
        print(f"  通信掩码示例: {comm_masks[i].cpu().numpy()}")
    pdb.set_trace()
