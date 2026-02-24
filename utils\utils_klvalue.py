import pdb
import torch
import torch.nn as nn
from buffer.kl_buffer import KLBuffer

class DummyCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class DummyAgent:
    def __init__(self, obs_dim, act_dim):
        self.critic = DummyCritic(obs_dim, act_dim)
def merge_action_dict(con_dict, dis_dict):
    return [
        con + dis
        for con, dis in zip(con_dict, dis_dict)
    ]

def sample_action(batch_size, action_names, action_bounds, device):
    acts = []
    for name in action_names:
        bound = action_bounds[name]
        if isinstance(bound[0], bool):  # discrete
            a = torch.randint(0, 2, (batch_size, 1), device=device).float()
        else:  # continuous
            low, high = bound
            a = torch.rand(batch_size, 1, device=device) * (high - low) + low
        acts.append(a)
    return torch.cat(acts, dim=1)

def sample_obs(batch_size, obs_dim, device):
    return torch.randn(batch_size, obs_dim, device=device)

def build_dummy_system(
    obs_dict,
    action_con_str_dict,
    action_dis_str_dict,
    action_bounds,
    batch_size=32,
    device="cpu"
):
    num_agents = len(obs_dict)
    action_merged_dict = merge_action_dict(
        action_con_str_dict, action_dis_str_dict
    )
    obs_dims = [len(obs) for obs in obs_dict]
    act_dims = [len(act) for act in action_merged_dict]
    agents = []
    obs_n = []
    act_n = []
    total_obs_dim = sum(obs_dims)
    total_act_dim = sum(act_dims)
    for i in range(num_agents):
        agents.append(
            DummyAgent(total_obs_dim, total_act_dim)
        )
    for i in range(num_agents):
        obs_n.append(
            sample_obs(batch_size, obs_dims[i], device)
        )
        act_n.append(
            sample_action(
                batch_size,
                action_merged_dict[i],
                action_bounds,
                device
            )
        )
    return agents, obs_n, act_n, action_merged_dict


def build_1d_action_grid(name, action_bounds, action_sep_num, device):
    bound = action_bounds[name]

    if isinstance(bound[0], bool):
        return torch.tensor([0.0, 1.0], device=device)
    else:
        n = action_sep_num[name]
        return torch.linspace(bound[0], bound[1], n+1, device=device)


def get_kl_value(
    agents,
    obs_n,
    act_n,
    agent_i,
    action_bounds,
    action_sep_num,
    action_merged_dict,
    temperature=1.0,
):
    """
    return:
        obs_i_rep:      Tensor [num_agents-1, B, obs_dim_i]
        comm_onehot:   Tensor [num_agents-1, num_agents]
        KL_values:     Tensor [num_agents-1, B]
    """
    device = obs_n[0].device
    B = obs_n[0].shape[0]
    num_agents = len(agents)
    obs_all = torch.cat(obs_n, dim=-1)
    eps = 1e-8
    T = temperature
    obs_i_list = []
    comm_id_list = []
    KL_list = []
    # ===== loop over candidate communication agents j =====
    for agent_j in range(num_agents):
        if agent_j == agent_i:
            continue
        KL_ij = torch.zeros(B, device=device)
        # ----- per action dimension of agent_i -----
        for i_dim, i_act_name in enumerate(action_merged_dict[agent_i]):
            grid_i = build_1d_action_grid(
                i_act_name, action_bounds, action_sep_num, device
            )
            # ===== conditional =====
            Q_cond = []
            for a_i_d in grid_i:
                act_all = []
                for n in range(num_agents):
                    a = act_n[n].clone()
                    if n == agent_i:
                        a[:, i_dim] = a_i_d
                    act_all.append(a)
                Q = agents[agent_i].critic(
                    torch.cat([obs_all,torch.cat(act_all, dim=-1)], dim=-1)
                )  # [B,1]
                Q_cond.append(Q)
            Q_cond = torch.cat(Q_cond, dim=1)
            P_cond = torch.softmax(Q_cond / T, dim=1)
            # ===== marginal over agent_j =====
            Q_marg = []
            for a_i_d in grid_i:
                Q_sum = 0.0
                for j_dim, j_act_name in enumerate(action_merged_dict[agent_j]):
                    grid_j = build_1d_action_grid(
                        j_act_name, action_bounds, action_sep_num, device
                    )
                    # print(j_dim, j_act_name, grid_j)

                    for a_j_d in grid_j:
                        # print(a_j_d)
                        act_all = []
                        for n in range(num_agents):
                            a = act_n[n].clone()
                            if n == agent_i:
                                a[:, i_dim] = a_i_d
                            elif n == agent_j:
                                a[:, j_dim] = a_j_d
                            act_all.append(a)
                        Q = agents[agent_i].critic(
                            torch.cat([obs_all,torch.cat(act_all, dim=-1)], dim=-1)
                        )
                        Q_sum = Q_sum + torch.exp(Q / T)
                Q_marg.append(Q_sum)
            Q_marg = torch.cat(Q_marg, dim=1)
            P_marg = Q_marg / (Q_marg.sum(dim=1, keepdim=True) + eps)
            KL_d = torch.sum(
                P_marg * torch.log((P_marg + eps) / (P_cond + eps)),
                dim=1
            )
            KL_ij = KL_ij + KL_d
        # ===== collect samples =====
        obs_i_list.append(obs_n[agent_i])          # [B, obs_dim_i]
        onehot = torch.zeros(num_agents, device=device)
        onehot[agent_j] = 1.0
        comm_id_list.append(onehot)                # [num_agents]
        KL_list.append(KL_ij)                      # [B]
    obs_i_rep = torch.stack(obs_i_list, dim=0)     # [num_agents-1, B, obs_dim_i]
    comm_onehot = torch.stack(comm_id_list, dim=0) # [num_agents-1, num_agents]
    KL_values = torch.stack(KL_list, dim=0)        # [num_agents-1, B]
    return obs_i_rep, comm_onehot, KL_values


def build_kl_sample(
        obs_i_rep,  # [N, B, obs_dim_i]
        comm_id,  # [N, N]
        KL_vals,  # [N, B]
):
    """
    return:
        obs_inputs        [B_total, obs_dim_i]
        obs_onehot_inputs [B_total, N]
        KL_values         [B_total]
    """
    device = obs_i_rep.device
    N, B, obs_dim = obs_i_rep.shape
    obs_inputs = []
    obs_onehot_inputs = []
    KL_values = []
    for agent_j in range(N):
        # [B, obs_dim]
        obs_inputs.append(obs_i_rep[agent_j])
        # [B, N]
        comm_j = comm_id[agent_j].unsqueeze(0).repeat(B, 1)
        obs_onehot_inputs.append(comm_j)
        # [B]
        KL_values.append(KL_vals[agent_j])
    obs_inputs = torch.cat(obs_inputs, dim=0)
    obs_onehot_inputs = torch.cat(obs_onehot_inputs, dim=0)
    KL_values = torch.cat(KL_values, dim=0)
    return obs_inputs, obs_onehot_inputs, KL_values


if __name__ == "__main__":
    obs_dict = [
        ["T_cabin_set", "cabinVolume.summary.T", "driverPerformance.controlBus.driverBus._acc_pedal_travel",
         "driverPerformance.controlBus.driverBus._brake_pedal_travel",
         "driverPerformance.controlBus.vehicleStatus.vehicle_velocity"],
        ["superHeatingSensor.outPort", "superCoolingSensor.outPort", "battery.controlBus.batteryBus.battery_SOC[1]"],
        ["T_bat_set", "T_motor_set", "battery.Batt_top[1].T", "machine.heatCapacitor.T"]
    ]
    action_con_str_dict = [
        ["RPM_blower"],
        ["RPM_comp"],
        ["RPM_batt", "RPM_motor"]
    ]
    action_dis_str_dict = [
        [],
        [],
        ['V_three', 'V_four']
    ]
    # action_merged_dict = [
    #     con + dis
    #     for con, dis in zip(action_con_str_dict, action_dis_str_dict)
    # ]
    action_bounds = {
        "RPM_blower": [0, 300],
        "RPM_comp": [0, 3000],
        "RPM_batt": [0, 3000],
        "RPM_motor": [0, 3000],
        "V_three": [True, False],
        "V_four": [True, False],
    }
    action_sep_num = {
        "T_epsilon": 6,
        "RPM_blower": 10,
        "RPM_comp": 30,
        "RPM_batt": 30,
        "RPM_motor": 30,
    }

    agents, obs_n, act_n, action_merged_dict = build_dummy_system(
        obs_dict,
        action_con_str_dict,
        action_dis_str_dict,
        action_bounds,
        batch_size=5,
        device='cpu'
    )

    agent_i = 0
    obs_dim = obs_n[agent_i].shape[1]
    obs_onehot_dim = len(obs_n)
    buffer = KLBuffer(
        buffer_size=10,
        obs_dim=obs_dim,
        obs_onehot_dim=obs_onehot_dim,
        percentile=80
    )

    with torch.no_grad():
        obs_i_rep, comm_id, KL_vals = get_kl_value(
            agents=agents,
            obs_n=obs_n,
            act_n=act_n,
            agent_i=agent_i,
            action_bounds=action_bounds,
            action_sep_num=action_sep_num,
            action_merged_dict=action_merged_dict,
            temperature=10.0
        )
    obs_inputs, obs_onehot_inputs, KL_values = build_kl_sample(obs_i_rep, comm_id, KL_vals)
    is_full = buffer.insert(
        obs_inputs=obs_inputs.cpu().numpy(),  # [B, obs_dim_i]
        obs_onehot_inputs=obs_onehot_inputs.numpy(),
        KL_values=KL_values.cpu().numpy()
    )
    if is_full:
        print("Prior buffer full. Hard labels generated.")
        print(buffer.get_samples(4)) # 一次最大取buffer_size*(1-percentile)个正样本，再*2为get_samples最大可取batch_size
        pdb.set_trace()