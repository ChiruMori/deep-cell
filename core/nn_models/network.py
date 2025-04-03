import math
from torch import nn
import torch.nn.functional as F
import torch

class CellNetwork(nn.Module):

    VALID_TYPES = ['cancer', 'erythrocyte', 'alveolar', 'stem']

    def __init__(self, cell_type=''):
        super().__init__()
        if cell_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid cell type: {cell_type}. Valid types: {self.VALID_TYPES}")
        
        # 保存细胞类型
        self.cell_type = cell_type
        
        # 扩展输入维度，添加细胞类型、养分、位置等信息
        self.input_layer = nn.Linear(11, 32)  # [9个基本信息 + 2速度]
        self.hidden1 = nn.Linear(32, 32)
        self.hidden2 = nn.Linear(32, 16)
        
        # 不同细胞类型的专用层
        if cell_type == 'cancer':
            self.type_specific = nn.Linear(16, 16)
        elif cell_type == 'erythrocyte':
            self.type_specific = nn.Linear(16, 16)
        elif cell_type == 'alveolar':
            self.type_specific = nn.Linear(16, 16)
        elif cell_type == 'stem':
            self.type_specific = nn.Linear(16, 16)
        else:
            raise ValueError("Invalid cell type" + cell_type)
        
        # 输出层 - 角度、强度和行为
        self.angle_layer = nn.Linear(16, 1)
        self.strength_layer = nn.Linear(16, 1)  
        self.behavior_layer = nn.Linear(16, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用正态分布初始化 + 随机扰动
                nn.init.normal_(m.weight, mean=0.0, std=0.3)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)
        # 输出层特殊初始化
        with torch.no_grad():
            # 角度层初始化到π附近
            self.angle_layer.weight.data.uniform_(-3.14, 3.14)
            # 初始化强度到较大值，增大运动倾向
            self.strength_layer.bias.data.fill_(0.8)

    def forward(self, x):
        # 确保输入张量的形状正确
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 使用Leaky ReLU激活函数
        x = F.leaky_relu(self.input_layer(x), 0.1)
        x = F.leaky_relu(self.hidden1(x), 0.1)
        x = F.leaky_relu(self.hidden2(x), 0.1)
        
        # 细胞类型特定处理
        x = F.leaky_relu(self.type_specific(x), 0.1)
        
        # 角度输出：分解为sin/cos分量 + 相位学习
        angle_feat = self.angle_layer(x)
        # 直接学习相位参数
        angle_sin = torch.sin(angle_feat)
        angle_cos = torch.cos(angle_feat)

        # 强度输出 - 使用sigmoid（增加温度系数）
        strength = torch.sigmoid(self.strength_layer(x) / 0.5) * 1.0

        # 行为参数 - 添加噪声并限制数值范围
        kw_base = torch.sigmoid(self.behavior_layer(x))
        kw_noise = torch.randn_like(x[:, :1]) * 0.1
        kw = (kw_base + kw_noise).clamp(min=0.0, max=1.0)

        return torch.cat([angle_sin, angle_cos, strength, kw], dim=1)
