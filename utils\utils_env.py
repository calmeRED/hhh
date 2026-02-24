
def fill_observation(obs_dict, obs):
    filled = []
    for row in obs_dict:
        new_row = []
        for item in row:
            if isinstance(item, str):
                # 如果是字符串路径，尝试从 obs 中取值
                if item in obs:
                    new_row.append(obs[item])
                else:
                    raise KeyError(f"Key not found in obs: {item}")
            else:
                # 非字符串（如 T_cabin_set），保留原值
                new_row.append(item)
        filled.append(new_row)
    return filled

def construct_action_dict(actions, action_con_str_dict, action_dis_str_dict):
    result = {}
    for i in range(len(actions)):
        k, l = 0, 0
        for param in actions[i]:
            if k < len(action_dis_str_dict[i]): # dis先存
                result[action_dis_str_dict[i][k]] = param
                k += 1
            elif l < len(action_con_str_dict[i]):
                result[action_con_str_dict[i][l]] = param
                l += 1
    return result


def scale_actions(action_dict, action_bounds):
    scaled = {}
    for key, val in action_dict.items():
        if key not in action_bounds:
            print(f"警告: {key} 不在 action_bounds 中，跳过。")
            continue
        bounds = action_bounds[key]
        if isinstance(bounds[0], bool) or isinstance(bounds[1], bool):
            scaled[key] = val
        else:
            low, high = bounds
            scaled[key] = ((val + 1) / 2) * (high - low) + low
    return scaled




def fill_list_with_dict(nested_list, data_dict):
    for i in range(len(nested_list)):
        item = nested_list[i]
        if isinstance(item, list):
            fill_list_with_dict(item, data_dict)
        elif isinstance(item, str):
            nested_list[i] = data_dict[item]
    return nested_list