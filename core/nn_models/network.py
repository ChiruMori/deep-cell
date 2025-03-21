from torch import nn
import torch.nn.functional as F
import torch

class CellNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 简化网络结构，提高推理速度
        self.input_layer = nn.Linear(9, 16)  # 减少神经元数量
        self.hidden = nn.Linear(16, 16)      # 只使用一个隐藏层
        self.output_layer = nn.Linear(16, 3) # 输出三个值：角度、强度和kw
        
        # 移除批归一化层，加快推理速度
        # 对于实时应用，批归一化可能会引入额外延迟
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
                # 对输出层特殊处理，增加各维度的初始化差异
                if m is self.output_layer:
                    # 角度维度正常初始化
                    # 强度和kw维度给予不同的初始偏置，避免始终输出相似值
                    if m.bias is not None:
                        m.bias.data[0] = 0.0     # 角度偏置为0
                        m.bias.data[1] = -0.5    # 强度偏置为负，经过sigmoid后为~0.37
                        m.bias.data[2] = 0.5     # kw偏置为正，经过sigmoid后为~0.62
                elif m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 确保输入张量的形状正确
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # 使用ReLU激活函数，计算快速
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden(x))
        
        # 获取原始输出
        raw_output = self.output_layer(x)
        
        # 处理角度输出: 将任意值转换为0-2π范围
        # 方法1: 使用余弦和正弦对角度进行编码，然后转换回角度
        angle_radians = torch.atan2(
            torch.sin(raw_output[:, 0]),
            torch.cos(raw_output[:, 0])
        )
        # 确保角度在0-2π范围内
        angle_radians = torch.where(
            angle_radians < 0,
            angle_radians + 2 * torch.pi,
            angle_radians
        )
        
        # 处理强度和kw (保持在0-1范围)
        strength = torch.sigmoid(raw_output[:, 1])
        kw = torch.sigmoid(raw_output[:, 2])
        
        # 构建最终输出
        if x.size(0) == 1:  # 如果只有一个样本
            return torch.cat([
                angle_radians.view(1, 1),
                strength.view(1, 1),
                kw.view(1, 1)
            ], dim=1)
        else:
            return torch.stack([angle_radians, strength, kw], dim=1)
