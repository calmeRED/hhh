import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, num_outputs, num_layers=3, hidden_dim=128):
        super(MLPModel, self).__init__()
        self.layers = nn.ModuleList()

        # 构建隐藏层：num_layers-1 层，每层 hidden_dim 单元，使用 ReLU 激活
        in_dim = input_dim
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim

        # 输出层：无激活函数
        self.layers.append(nn.Linear(in_dim, num_outputs))
        # 注意：不加 activation_fn，保持 logits 输出

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x