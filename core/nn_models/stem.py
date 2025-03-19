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
        return self.fc(x)
