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
                nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 细胞类型特定的初始化
        with torch.no_grad():
            # 全部参数均初始化为0.5输出
            self.type_specific.bias.data.fill_(0.5)
            self.type_specific.weight.data.fill_(0.5)

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
        
        # 角度输出 - 标准化到 0-2π
        angle = torch.sigmoid(self.angle_layer(x)) * 6.28319

        # 强度输出 - 直接使用sigmoid
        strength = torch.sigmoid(self.strength_layer(x))

        # 行为参数 - 移除复杂调制
        kw = torch.sigmoid(self.behavior_layer(x))

        return torch.cat([angle, strength, kw], dim=1)
