from pprint import pp

def derive_dimensions(config):
    obs_dict = config["obs_dict"]
    action_con_str_dict = config["action_con_str_dict"]
    action_dis_str_dict = config["action_dis_str_dict"]
    action_merged_dict = [
        con + dis
        for con, dis in zip(action_con_str_dict, action_dis_str_dict)
    ]
    message_feature_dim = config["message_feature_dim"]
    # 每个 agent 的观测维度
    obs_dims = [len(obs) for obs in obs_dict]
    # 连续 / 离散动作维度
    action_con_dims = [len(a) for a in action_con_str_dict]
    action_dis_dims = [len(a) for a in action_dis_str_dict]
    # I2C: 增强观测（local obs + message）
    enhanced_obs_dims = [
        dim + message_feature_dim for dim in obs_dims
    ]
    # Critic 输入：所有 obs + 所有 action
    critic_input_dim = (
        sum(obs_dims)
        + sum(action_con_dims)
        + sum(action_dis_dims)
    )
    n_agents = len(obs_dims)
    return {
        "n_agents": n_agents,
        "obs_dims": obs_dims,
        "state_dims": obs_dims,  # 与原实现保持兼容
        "action_con_dims": action_con_dims,
        "action_dis_dims": action_dis_dims,
        "action_merged_dict": action_merged_dict,
        "enhanced_obs_dims": enhanced_obs_dims,
        "critic_input_dim": critic_input_dim,
        "i2c_hidden_dim": config.get("i2c_hidden_dim", 256),
    }

def get_config(base_config):
    config = base_config.copy()
    derived = derive_dimensions(config)
    config.update(derived)
    return config


if __name__ == "__main__":
    from config.base_config import config
    pp(get_config(config))
