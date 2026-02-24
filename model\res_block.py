import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.ReLU()               # ← 替换

        if dim != hidden_dim:
            self.proj = nn.Linear(dim, dim, bias=False)
        else:
            self.proj = nn.Identity()

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = self.proj(x)
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out = out + residual
        out = self.activation(self.norm(out))
        return out


class DeepResFC(nn.Module):
    def __init__(self, num_in, num_out, hidden_dim, n_blocks=6):
        super().__init__()
        self.input_proj = nn.Linear(num_in, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, num_out)
        self.act = nn.ReLU()                      # ← 替换
        self.norm = nn.LayerNorm(num_out)

    def forward(self, x):
        out = self.act(self.input_proj(x))
        for blk in self.blocks:
            out = blk(out)
        out = self.output_proj(out)          # (B, num_out) 直接返回
        # out = self.norm(out) # 不能使用norm否则输出又扁平了
        return out