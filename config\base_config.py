import torch
from utils.utils_misc import C_to_K

config = {
    # ===== 环境 =====
    # "fmu_path": "env/MyITMS.fmu",
    "fmu_step_size": 1,
    # ===== 设备 & 训练 =====
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_folder": "runs/",
    "num_episodes": 50,
    "episode_iter": 200,
    "buffer_size": 10000,
    "hidden_dim": 1024,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "gamma": 0.95,
    "tau": 1e-2,
    "batch_size": 20,
    "eval_interval":100, # 每n个episode
    "save_interval":10, # 每n个episode

    # ===== I2C =====
    "use_i2c": True,
    "lambda_temp": 10.0,
    "i2c_hidden_dim": 256,
    "prior_buffer_size": 100,
    "prior_buffer_percentile": 80,
    "message_feature_dim":16,
    "i2c_num_layers": 6,
    "prior_lr": 1e-3,
    "prior_train_iter": 3,
    "prior_train_batch_size": 10,
    "prior_update_frequency": 5,
    # ===== 观测（每个 agent 一个 list）=====
    "obs_dict": [
        [
            "T_cabin_set",
            "cabinVolume.summary.T",
            "driverPerformance.controlBus.driverBus._acc_pedal_travel",
            "driverPerformance.controlBus.driverBus._brake_pedal_travel",
            "driverPerformance.controlBus.vehicleStatus.vehicle_velocity",
        ],
        [
            "superHeatingSensor.outPort",
            "superCoolingSensor.outPort",
            "battery.controlBus.batteryBus.battery_SOC[1]",
        ],
        [
            "T_bat_set",
            "T_motor_set",
            "battery.Batt_top[1].T",
            "machine.heatCapacitor.T",
        ],
    ],
    # ===== 动作 =====
    "action_con_str_dict": [
        ["RPM_blower"],
        ["RPM_comp"],
        ["RPM_batt", "RPM_motor"],
    ],
    "action_dis_str_dict": [
        [],
        [],
        [], # "V_three", "V_four"
    ],
    # ===== 动作约束 =====
    "action_bounds": {
        "RPM_blower": [0, 300],
        "RPM_comp": [0, 3000],
        "RPM_batt": [0, 3000],
        "RPM_motor": [0, 3000],
        "V_three": [True, False],
        "V_four": [True, False],
    },
    "action_sep_num": {
        "T_epsilon": 6,
        "RPM_blower": 10,
        "RPM_comp": 30,
        "RPM_batt": 30,
        "RPM_motor": 30,
    },
    "reward_dict": ["TableDC.Pe", "TableDC1.Pe", "TableDC2.Pe", "TableDC3.Pe"],
    # ===== 物理设定 =====
    "T_cabin_set": C_to_K([20]),
    "T_bat_set": C_to_K([30]),
    "T_motor_set": C_to_K([90]),
    # ===== Replay Buffer =====
    "replay_buffer_agents_sample_weight": [0.4, 0.3, 0.3],
    "use_agent_buffer": False,

    "env_reset_dict": {"MY_socinit": [0.1, 1.0, 0.05],
                       "MY_battT0": [303.15, 303.15, 0],
                        "MY_motorT0": [333.15, 333.15, 0],
                        # "T_Cabin": 0.123,
                        # "T_out": 0.123,
                        }
}