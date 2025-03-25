from torch import nn
import torch.nn.functional as F
import torch

class CellNetwork(nn.Module):
    def __init__(self, cell_type=''):
        super().__init__()
        
        # 保存细胞类型
        self.cell_type = cell_type
        
        # 扩展输入维度，添加细胞类型、养分、位置等信息
        self.input_layer = nn.Linear(10, 32)  # [9个基本信息 + 当前速度]
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
            self.type_specific = nn.Linear(16, 16)
        
        # 输出层 - 角度、强度和行为
        self.angle_layer = nn.Linear(16, 1)
        self.strength_layer = nn.Linear(16, 1)  
        self.behavior_layer = nn.Linear(16, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 细胞类型特定的初始化
        with torch.no_grad():
            # 行为层初始化成多个高斯分布
            self.behavior_layer.bias.data[0] = 0.5  # 中间值
            
            # 强度层初始化，不同细胞类型不同默认值
            if self.cell_type == 'erythrocyte':
                self.strength_layer.bias.data[0] = 0.7  # 红细胞默认更快
            elif self.cell_type == 'alveolar':
                self.strength_layer.bias.data[0] = 0.3  # 表皮细胞相对静止
            else:
                self.strength_layer.bias.data[0] = 0.5  # 其他默认中等速度

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
        
        # 角度输出 - 使用三角函数处理循环性
        angle_raw = self.angle_layer(x)
        angle_radians = torch.atan2(
            torch.sin(angle_raw),
            torch.cos(angle_raw)
        )
        angle_normalized = torch.where(
            angle_radians < 0,
            angle_radians + 2 * torch.pi,
            angle_radians
        )
        
        # 强度输出 - 使用sigmoids和振荡函数
        strength_raw = self.strength_layer(x)
        
        # 正弦波调制的sigmoid，创造多个吸引子点(0.2, 0.5, 0.8)
        modulation = 0.3 * torch.sin(strength_raw * 6) 
        strength = torch.sigmoid(strength_raw) + modulation
        strength = torch.clamp(strength, 0.05, 0.95)  # 避免极端值
        
        # 行为参数 - 多个稳定点
        behavior_raw = self.behavior_layer(x)
        behavior_base = torch.sigmoid(behavior_raw)
        
        # 创建多个吸引子区域(0.2, 0.5, 0.8)
        behavior_mod = 0.25 * torch.sin(behavior_base * 6 * torch.pi)
        behavior = behavior_base + behavior_mod
        behavior = torch.clamp(behavior, 0.05, 0.95)  # 避免极端值
        
        # 构建最终输出
        if x.size(0) == 1:
            return torch.cat([
                angle_normalized.view(1, 1),
                strength.view(1, 1),
                behavior.view(1, 1)
            ], dim=1)
        else:
            return torch.stack([angle_normalized, strength, behavior], dim=1)
