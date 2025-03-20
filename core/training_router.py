from fastapi import APIRouter, WebSocket
from .models.cell import TrainingInput, TrainingOutput, TrainingFeedback
from .nn_models.stem import StemNetwork
from typing import Dict, List
import torch
import numpy as np
from collections import deque
import random

router = APIRouter()

# 经验回放缓冲区
class ExperienceBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

# 全局变量
model = StemNetwork()
experience_buffer = ExperienceBuffer()
training_sessions: Dict[str, WebSocket] = {}

def calculate_reward(feedback: TrainingFeedback) -> float:
    """计算奖励值
    - 存活时间越长越好
    - 子代数量越多越好
    - 养料剩余越多越好（说明是自然死亡，且资源充足，而不是因饥饿、遭到攻击等死亡）
    """
    # 标准化参数
    max_life_time = 60000  # 可能的最大存活时间
    max_son_count = 10    # 可能的最大子代数量
    max_hp = 10000       # 可能的最大HP值
    
    # 计算各项奖励
    life_reward = min(feedback.life_time / max_life_time, 1.0) * 0.4  # 40% 权重
    son_reward = min(feedback.son_count / max_son_count, 1.0) * 0.4   # 40% 权重
    hp_reward = max(feedback.hp / max_hp, 0.0) * 0.2                  # 20% 权重
    
    return life_reward + son_reward + hp_reward

async def train_model(batch_size=32):
    """训练模型"""
    if len(experience_buffer) < batch_size:
        return
        
    # 从经验回放缓冲区采样
    batch = experience_buffer.sample(batch_size)
    
    # 准备训练数据
    states = []
    target_directions = []
    target_strengths = []
    
    for exp in batch:
        state = exp['state']
        action = exp['action']
        reward = exp['reward']
        
        # 状态数据
        states.append([
            state['life'],
            state['hp'],
            *state['surround']
        ])
        
        # 根据奖励调整动作
        direction = action.get('angle', -1)
        strength = action.get('strength', -1)
        if direction == -1 or strength == -1:
            raise ValueError(f"Error action: {action}")
        # 如果奖励好，强化这个动作；如果奖励差，减弱这个动作
        target_directions.append(direction)  # 方向保持不变
        target_strengths.append(strength * (1 + reward))  # 根据奖励调整强度
    
    # 转换为张量
    states_tensor = torch.tensor(states, dtype=torch.float32)
    target_directions_tensor = torch.tensor(target_directions, dtype=torch.float32)
    target_strengths_tensor = torch.tensor(target_strengths, dtype=torch.float32)
    
    # 训练模型
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 前向传播
    predictions = model(states_tensor)
    pred_directions = predictions[:, 0]
    pred_strengths = torch.sigmoid(predictions[:, 1])
    
    # 计算损失
    direction_loss = torch.nn.MSELoss()(pred_directions, target_directions_tensor)
    strength_loss = torch.nn.MSELoss()(pred_strengths, target_strengths_tensor)
    total_loss = direction_loss + strength_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    model.eval()

@router.websocket("/training/tick")
async def handle_websocket(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 连接已接受")
    try:
        while True:
            data = await websocket.receive_json()
            if isinstance(data, list):
                responses = []
                # 批量处理输入数据
                input_tensors = []
                input_map = {}
                
                for idx, item in enumerate(data):
                    input = TrainingInput(**item)
                    input_map[input.id] = idx
                    
                    input_tensors.append([
                        input.life,
                        input.hp,
                        *input.surround
                    ])
                
                if input_tensors:
                    batch_tensor = torch.tensor(input_tensors, dtype=torch.float32)
                    with torch.no_grad():
                        predictions = model(batch_tensor)
                    
                    predictions = predictions.squeeze()
                    directions = predictions[:, 0].tolist()
                    strengths = torch.sigmoid(predictions[:, 1]).tolist()
                    
                    for item in data:
                        idx = input_map[item['id']]
                        direction = directions[idx]
                        strength = strengths[idx]
                        
                        responses.append({
                            "id": item['id'],
                            "angle": direction,
                            "strength": strength
                        })
                
                await websocket.send_json(responses)
    except Exception as e:
        print(f"tick webSocket error: {e}")

@router.websocket("/training/apoptosis")
async def on_apoptosis(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message['type']
            data = message['data']

            if message_type == 'realtime':
                # 处理实时反馈
                for feedback in data:
                    state = feedback['state']
                    action = feedback['action']
                    immediate_reward = feedback['immediate_reward']
                    is_terminal = feedback['is_terminal']

                    # 添加到经验回放缓冲区
                    experience_buffer.add({
                        'state': state,
                        'action': action,
                        'reward': immediate_reward,
                        'is_terminal': is_terminal
                    })
            # 定期训练
            if len(experience_buffer) >= 32:
                await train_model()
    except Exception as e:
        print(f"apoptosis webSocket error: {e}")

@router.websocket("/senescence")
async def on_senescence(websocket: WebSocket):
    pass