import pdb

import numpy as np
import torch
from collections import deque


class DummyEnv:
    def __init__(self, obs_dict, action_con_str_dict, action_dis_str_dict, action_bounds, step_size=10):
        """
        初始化 DummyEnv

        Args:
            obs_dict: 观测字典结构
            action_con_str_dict: 连续动作字符串列表
            action_dis_str_dict: 离散动作字符串列表
            action_bounds: 动作边界字典
            step_size: 步长
        """
        self.obs_dict = obs_dict
        self.action_con_str_dict = action_con_str_dict
        self.action_dis_str_dict = action_dis_str_dict
        self.action_bounds = action_bounds
        self.step_size = step_size
        # 展平所有观测键
        self.obs_keys = []
        for item in obs_dict:
            if isinstance(item, str):
                self.obs_keys.append(item)

        # 展平所有动作键
        self.action_keys = []
        for con_list in action_con_str_dict:
            self.action_keys.extend(con_list)
        for dis_list in action_dis_str_dict:
            self.action_keys.extend(dis_list)

        # 模拟状态
        self.current_step = 0
        self.max_steps = 1000
        self.state = {}

        # 初始化状态
        self._reset_state()

        # 模拟观测空间
        self.observation_space = {
            'shape': (len(self.obs_keys),),
            'low': np.array([-100.0] * len(self.obs_keys)),
            'high': np.array([100.0] * len(self.obs_keys))
        }

    def _reset_state(self):
        """重置状态"""
        # 为每个观测键生成随机初始值
        for key in self.obs_keys:
            if key == 'cabinVolume.summary.T':
                self.state[key] = 25.0 + np.random.randn() * 50  # 座舱温度
            elif key == 'battery.Batt_top[1].T':
                self.state[key] = 35.0 + np.random.randn() * 50  # 电池温度
            elif key == 'machine.heatCapacitor.T':
                self.state[key] = 85.0 + np.random.randn() * 50  # 电机温度
            elif key == 'battery.controlBus.batteryBus.battery_SOC[1]':
                self.state[key] = 0.8 + np.random.randn() * 1  # 电池SOC
            elif key == 'superHeatingSensor.outPort':
                self.state[key] = 5.0 + np.random.randn() * 20  # 过热传感器
            elif key == 'superCoolingSensor.outPort':
                self.state[key] = 5.0 + np.random.randn() * 20  # 过冷传感器
            elif key == 'driverPerformance.controlBus.driverBus._acc_pedal_travel':
                self.state[key] = 0.5 + np.random.randn() * 2  # 加速踏板
            elif key == 'driverPerformance.controlBus.driverBus._brake_pedal_travel':
                self.state[key] = 0.2 + np.random.randn() * 1  # 制动踏板
            elif key == 'driverPerformance.controlBus.vehicleStatus.vehicle_velocity':
                self.state[key] = 60.0 + np.random.randn() * 100  # 车速
            elif key == 'cabinVolume.summary.T':
                self.state[key] = 20.0 + np.random.randn() * 30  # 座舱温度
            elif key == 'TableDC.Pe':
                self.state[key] = 1000.0 + np.random.randn() * 1000  # 压缩机功率
            elif key == 'TableDC1.Pe':
                self.state[key] = 500.0 + np.random.randn() * 500  # 电池水泵功率
            elif key == 'TableDC2.Pe':
                self.state[key] = 800.0 + np.random.randn() * 800  # 电机水泵功率
            elif key == 'TableDC3.Pe':
                self.state[key] = 300.0 + np.random.randn() * 300  # 风扇功率
            else:
                self.state[key] = np.random.randn() * 100  # 默认随机值

    def reset(self):
        """重置环境"""
        self.current_step = 0
        self._reset_state()
        return self.state.copy()

    def step(self, actions):
        """执行一步"""
        self.current_step += 1

        # 模拟环境动态
        for key in self.state:
            self.state[key] += (np.random.randn() * 2)

        # # 限制状态范围
        # for key in self.state:
        #     if key == 'cabinVolume.summary.T':
        #         self.state[key] = np.clip(self.state[key], 15, 30)
        #     elif key == 'battery.Batt_top[1].T':
        #         self.state[key] = np.clip(self.state[key], 25, 45)
        #     elif key == 'machine.heatCapacitor.T':
        #         self.state[key] = np.clip(self.state[key], 75, 100)
        #     elif key == 'battery.controlBus.batteryBus.battery_SOC[1]':
        #         self.state[key] = np.clip(self.state[key], 0.2, 0.95)


        # 检查是否结束
        trunc = (self.current_step >= self.max_steps)
        term = False
        # 返回观测、奖励、是否结束
        return self.state.copy(), term, trunc

    def render(self):
        """渲染环境（可选）"""
        pass

    def close(self):
        """关闭环境"""
        pass


# 用于测试的示例代码
if __name__ == "__main__":
    # 创建测试参数
    obs_dict = [
        ['cabinVolume.summary.T', 'driverPerformance.controlBus.driverBus._acc_pedal_travel',
         'driverPerformance.controlBus.driverBus._brake_pedal_travel'],
        ['superHeatingSensor.outPort', 'superCoolingSensor.outPort', 'battery.controlBus.batteryBus.battery_SOC[1]'],
        ['battery.Batt_top[1].T', 'machine.heatCapacitor.T']
    ]

    action_con_str_dict = [
        ['RPM_blower'],
        ['RPM_comp'],
        ['RPM_batt', 'RPM_motor']
    ]

    action_dis_str_dict = [
        [],
        [],
        []
    ]

    action_bounds = {
        "RPM_blower": [0, 300],
        "RPM_comp": [0, 3000],
        "RPM_batt": [0, 3000],
        "RPM_motor": [0, 3000],
    }

    # 创建环境
    env = DummyEnv(obs_dict, action_con_str_dict, action_dis_str_dict, action_bounds)

    # 测试重置
    obs = env.reset()

    # 测试动作
    actions = {
        "RPM_blower": 150,
        "RPM_comp": 1500,
        "RPM_batt": 1500,
        "RPM_motor": 1500
    }

    # 测试一步执行
    for i in range(3):
        next_obs, term, trunc = env.step(actions)
        print(next_obs)