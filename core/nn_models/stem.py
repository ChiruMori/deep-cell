import torch
from torch import nn

class StemNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            # 8个输入节点，2个输出节点，通过2个全连接层，每层4个神经元，使用ReLU激活函数
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        # 确保输入张量的形状正确
        if x.dim() == 1:
            # 如果是一维张量，将其转换为二维张量 (1, 8)
            x = x.unsqueeze(0)
        return self.fc(x)
