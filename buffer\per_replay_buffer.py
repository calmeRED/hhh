import numpy as np
import random


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._n_entries = 0

    def update(self, data_idx, priority):
        """更新叶子节点优先级"""
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # 向上传播
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def add(self, priority):
        """添加新叶子（仅用于初始化，实际由外部控制索引）"""
        if self._n_entries < self.capacity:
            self._n_entries += 1

    def get(self, s):
        """根据值 s 检索样本索引"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def n_entries(self):
        return self._n_entries


class MultiAgentExclusivePER:
    def __init__(self, capacity, n_agents, alpha=0.6, epsilon=0.01):
        self.capacity = capacity
        self.n_agents = n_agents
        self.alpha = alpha
        self.epsilon = epsilon

        # 主存储
        self.data = [None] * capacity
        self.used = np.zeros(capacity, dtype=bool)

        # 每个智能体一个 SumTree
        self.trees = [SumTree(capacity) for _ in range(n_agents)]

        self.write_idx = 0
        self.size = 0

    def add_default_priority(self, sample):
        """使用最大优先级或上次使用的优先级添加新样本"""
        idx = self.write_idx
        self.data[idx] = sample
        self.used[idx] = False

        # 获取当前最大优先级
        max_priority = max(tree.tree[0] for tree in self.trees) if self.size > 0 else 1.0

        # 为所有智能体设置初始优先级
        for i in range(self.n_agents):
            self.trees[i].update(idx, max_priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def add(self, td_errors, sample):
        """兼容旧API，直接调用add_default_priority"""
        self.add_default_priority(sample)
        # 如果提供了td_errors，也更新优先级
        if td_errors is not None:
            idx = (self.write_idx - 1) % self.capacity  # 获取刚添加的索引
            self.update_priorities([idx], [td_errors])

    def sample(self, batch_size, agent_weights):
        if self.size == 0:
            return [], [], []

        # 计算每个智能体采样数量
        agent_samples = [int(batch_size * w) for w in agent_weights]
        remainder = batch_size - sum(agent_samples)
        for i in range(remainder):
            agent_samples[i % self.n_agents] += 1

        batch = []
        agent_assignments = []
        sampled_indices = []
        used_this_round = set()

        for agent_id in range(self.n_agents):
            needed = agent_samples[agent_id]
            collected = 0
            attempts = 0
            max_attempts = needed * 20

            while collected < needed and attempts < max_attempts:
                if self.trees[agent_id].total() == 0:
                    break

                s = random.uniform(0, self.trees[agent_id].total())
                data_idx, priority = self.trees[agent_id].get(s)

                # 检查有效性
                if (0 <= data_idx < self.size and
                        not self.used[data_idx] and
                        data_idx not in used_this_round):
                    batch.append(self.data[data_idx])
                    agent_assignments.append(agent_id)
                    sampled_indices.append(data_idx)
                    used_this_round.add(data_idx)
                    collected += 1

                attempts += 1

        # # 标记为已使用（互斥）
        # for idx in sampled_indices:
        #     self.used[idx] = True

        return batch, agent_assignments, sampled_indices

    def update_priorities(self, indices, td_errors_list):
        for idx, td_errors in zip(indices, td_errors_list):
            if 0 <= idx < self.size:
                for i in range(self.n_agents):
                    priority = (abs(td_errors[i]) + self.epsilon) ** self.alpha
                    self.trees[i].update(idx, priority)