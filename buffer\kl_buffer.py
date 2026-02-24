import numpy as np


class KLBuffer(object):
    def __init__(self, buffer_size, obs_dim, obs_onehot_dim, percentile):
        self.num_steps = buffer_size

        self.obs_inputs = np.zeros((self.num_steps, obs_dim))
        self.obs_onehot_inputs = np.zeros((self.num_steps, obs_onehot_dim))
        self.KL_values = np.zeros((self.num_steps, 1))
        self.labels = np.zeros((self.num_steps, 2))

        self.step = 0
        self.percentile = percentile

        self.pos_idx = []
        self.neg_idx = []
        self.pos_t = 0
        self.neg_t = 0
        """
        持续存储KL散度信息等，buffer满时自动构造监督硬标签
        """

    def insert(self, obs_inputs, obs_onehot_inputs, KL_values):
        """
        obs_inputs: [B, obs_dim]
        obs_onehot_inputs: [B, obs_onehot_dim]
        KL_values: [B]
        """
        size = obs_inputs.shape[0]

        if self.step + size < self.num_steps:
            self.obs_inputs[self.step:self.step + size] = obs_inputs
            self.obs_onehot_inputs[self.step:self.step + size] = obs_onehot_inputs
            self.KL_values[self.step:self.step + size] = KL_values[:, None]
            self.step += size
            return False  # not full
        else:
            remain = self.num_steps - self.step

            self.obs_inputs[self.step:] = obs_inputs[:remain]
            self.obs_onehot_inputs[self.step:] = obs_onehot_inputs[:remain]
            self.KL_values[self.step:] = KL_values[:remain, None]

            self._build_labels()
            self._shuffle_indices()

            self.step = 0
            return True  # buffer full

    def _build_labels(self):
        self.pos_idx = []
        self.neg_idx = []

        threshold = np.percentile(self.KL_values, self.percentile)

        for i in range(self.num_steps):
            if self.KL_values[i] > threshold:
                self.labels[i] = [1, 0]  # communicate
                self.pos_idx.append(i)
            else:
                self.labels[i] = [0, 1]  # no communicate
                self.neg_idx.append(i)

    def _shuffle_indices(self):
        np.random.shuffle(self.pos_idx)
        np.random.shuffle(self.neg_idx)
        self.pos_t = 0
        self.neg_t = 0

    def get_samples(self, batch_size):
        half = batch_size // 2

        pos = self.pos_idx[self.pos_t:self.pos_t + half]
        neg = self.neg_idx[self.neg_t:self.neg_t + half]

        self.pos_t += half
        self.neg_t += half

        if self.pos_t >= len(self.pos_idx):
            self.pos_t = 0
        if self.neg_t >= len(self.neg_idx):
            self.neg_t = 0

        idx = np.array(pos + neg)

        return (
            self.obs_inputs[idx],
            self.obs_onehot_inputs[idx],
            self.labels[idx]
        )