from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .models.cell import TrainingInput
from .nn_models.network import CellNetwork
import torch
import numpy as np
from collections import deque
import random
import traceback
import asyncio
import os
import time

router = APIRouter()

# 经验回放缓冲区
class ExperienceBuffer:
    def __init__(self, max_size=5000):  # 减小缓冲区大小，提高训练效率
        self.buffer = deque(maxlen=max_size)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

# 训练记录器
class TrainingLogger:
    def __init__(self, log_dir="training_logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_log_{int(time.time())}.txt")
        
    def log(self, message):
        # 仅写入文件，不打印到控制台，减少I/O延迟
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")

# 全局变量
logger = TrainingLogger()
model = CellNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # 学习率
experience_buffer = ExperienceBuffer()
training_iterations = 0
active_connections = set()  # 跟踪活动连接

# 设置模型为评估模式以提高推理速度
model.eval()

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        
    async def connect(self, websocket: WebSocket, client_id: str = "default"):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        return websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    def is_connected(self, client_id: str):
        return client_id in self.active_connections

manager = ConnectionManager()

# 简化的训练函数
async def train_model(batch_size=16, epochs=5):
    """训练模型"""
    global training_iterations, optimizer, model, experience_buffer
    
    if len(experience_buffer) < batch_size:
        return False
        
    try:
        # 从经验回放缓冲区采样
        batch = experience_buffer.sample(batch_size)
        
        # 准备训练数据
        states = []
        actions = []
        rewards = []
        
        for exp in batch:
            try:
                state = exp['state']
                action = exp['action']
                reward = exp['reward']
                
                states.append([
                    state['c_type'],
                    state['life'],
                    state['hp'],
                    *state['surround']
                ])
                
                actions.append([
                    action['angle'],
                    action['strength'],
                    action.get('kw', 0.5)
                ])
                
                rewards.append(reward)
            except KeyError:
                continue
        
        if len(states) == 0:
            return False
        
        # 转换为张量
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        
        # 训练模型
        model.train()  # 设置为训练模式
        
        for _ in range(epochs):
            # 清空梯度
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(states_tensor)
            
            # 确保维度匹配
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(0)
                
            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(0)
            
            # 计算损失 - 添加单独的维度权重
            angle_loss = torch.nn.MSELoss()(predictions[:, 0], actions_tensor[:, 0])
            strength_loss = torch.nn.MSELoss()(predictions[:, 1], actions_tensor[:, 1]) * 2.0  # 增加强度的训练权重
            kw_loss = torch.nn.MSELoss()(predictions[:, 2], actions_tensor[:, 2]) * 2.0      # 增加kw的训练权重
            
            # 组合损失
            loss = angle_loss + strength_loss + kw_loss
            
            # 根据奖励加权损失
            weighted_loss = loss * (1 + rewards_tensor)
            weighted_loss = weighted_loss.mean()
            
            # 反向传播和参数更新
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 减小梯度裁剪范围
            optimizer.step()
        
        # 训练完成后设置回评估模式
        model.eval()
        training_iterations += 1
        
        # 降低保存模型的频率
        if training_iterations % 20 == 0:
            save_path = f"models/cell_model_{training_iterations}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iterations': training_iterations
            }, save_path)
        
        return True
    except Exception as e:
        logger.log(f"训练错误: {e}")
        traceback.print_exc()
        return False

@router.websocket("/training/tick")
async def handle_websocket(websocket: WebSocket):
    client_id = f"tick_{time.time()}"
    await manager.connect(websocket, client_id)
    logger.log(f"Tick WebSocket连接已建立: {client_id}")
    
    try:
        while True:
            try:
                # 设置接收超时
                data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                
                if isinstance(data, list) and len(data) > 0:
                    # 准备批量处理
                    responses = []
                    input_tensors = []
                    input_map = {}
                    
                    for idx, item in enumerate(data):
                        try:
                            # 直接使用字典处理输入数据，避免Pydantic验证开销
                            input_id = item.get('id', '')
                            if not input_id:
                                continue
                                
                            input_map[input_id] = idx
                            
                            # 确保surround数组有正确的长度
                            surround = item.get('surround', [0, 0, 0, 0, 0, 0])
                            if len(surround) < 6:
                                surround = surround + [0] * (6 - len(surround))
                            elif len(surround) > 6:
                                surround = surround[:6]
                            
                            input_tensors.append([
                                item.get('c_type', 0),
                                item.get('life', 0),
                                item.get('hp', 0),
                                *surround
                            ])
                        except Exception as item_e:
                            logger.log(f"处理请求项错误: {item_e}")
                            continue
                    
                    if input_tensors:
                        # 转为tensor并进行推理
                        try:
                            batch_tensor = torch.tensor(input_tensors, dtype=torch.float32)
                            with torch.no_grad():
                                predictions = model(batch_tensor)
                            
                            # 格式化输出
                            if len(input_tensors) == 1:
                                predictions = predictions.view(1, -1)
                            
                            # 提取预测结果 - 直接使用网络输出，网络已经处理好了范围
                            directions = predictions[:, 0].tolist()
                            strengths = predictions[:, 1].tolist()
                            kws = predictions[:, 2].tolist()
                            
                            # 记录几个样本的输出，用于调试
                            if len(input_tensors) > 0:
                                sample_idx = min(len(directions)-1, random.randint(0, len(directions)-1))
                                logger.log(f"Sample output: angle={directions[sample_idx]:.4f}, strength={strengths[sample_idx]:.4f}, kw={kws[sample_idx]:.4f}")
                            
                            # 生成响应
                            for item in data:
                                input_id = item.get('id', '')
                                if input_id in input_map:
                                    idx = input_map[input_id]
                                    if idx < len(directions):
                                        data_to_send = {
                                            "id": input_id,
                                            "angle": directions[idx],
                                            "strength": strengths[idx],
                                            "kw": kws[idx]
                                        }
                                        responses.append(data_to_send)
                        except Exception as pred_e:
                            logger.log(f"模型推理错误: {pred_e}")
                            # 发生错误时，使用随机值作为后备
                            for item in data:
                                responses.append({
                                    "id": item.get('id', ''),
                                    "angle": random.uniform(0, 6.28),
                                    "strength": random.uniform(0, 1),
                                    "kw": random.uniform(0, 1)
                                })
                    
                    # 发送响应
                    if responses:
                        try:
                            await websocket.send_json(responses)
                        except Exception as send_e:
                            logger.log(f"发送响应错误: {send_e}")
                            break
                        
            except asyncio.TimeoutError:
                # 超时检查连接是否仍然有效
                try:
                    # 发送一个ping来确认连接
                    pong = await websocket.receive_text()
                    continue
                except:
                    # 连接可能已断开
                    break
            except WebSocketDisconnect:
                logger.log(f"WebSocket连接断开: {client_id}")
                break
            except Exception as e:
                logger.log(f"Tick处理错误: {e}")
                # 继续循环而不是中断连接
                continue
    finally:
        manager.disconnect(client_id)
        logger.log(f"关闭Tick WebSocket连接: {client_id}")

@router.websocket("/training/apoptosis")
async def on_apoptosis(websocket: WebSocket):
    client_id = f"feedback_{time.time()}"
    await manager.connect(websocket, client_id)
    logger.log(f"反馈WebSocket连接已建立: {client_id}")
    
    try:
        while True:
            try:
                # 设置接收超时
                message = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                
                if not isinstance(message, dict) or 'type' not in message or 'data' not in message:
                    continue
                
                message_type = message['type']
                data = message['data']

                if message_type == 'realtime' and isinstance(data, list):
                    # 处理反馈
                    feedback_count = 0
                    for feedback in data:
                        try:
                            # 简化验证逻辑
                            if 'state' not in feedback or 'action' not in feedback:
                                continue
                                
                            state = feedback.get('state', {})
                            action = feedback.get('action', {})
                            
                            # 确保所有必要字段都存在
                            if not all(k in state for k in ['c_type', 'life', 'hp', 'surround']):
                                continue
                                
                            if not all(k in action for k in ['angle', 'strength']):
                                continue
                            
                            # 计算奖励 (使用传入的奖励或默认值)
                            immediate_reward = feedback.get('immediate_reward', 0) 
                            
                            # 简单处理奖励
                            immediate_reward = max(min(immediate_reward + random.uniform(-0.1, 0.1), 1.0), -1.0)
                            
                            # 添加到经验缓冲区
                            experience_buffer.add({
                                'state': state,
                                'action': action,
                                'reward': immediate_reward,
                                'is_terminal': feedback.get('is_terminal', False)
                            })
                            feedback_count += 1
                        except Exception:
                            continue
                    
                    # 只有当积累足够样本时才训练
                    if len(experience_buffer) >= 32:
                        await train_model(batch_size=16, epochs=3)
                    
            except asyncio.TimeoutError:
                # 超时检查连接
                try:
                    pong = await websocket.receive_text()
                    continue
                except:
                    break
            except WebSocketDisconnect:
                logger.log(f"反馈WebSocket连接断开: {client_id}")
                break
            except Exception as e:
                logger.log(f"处理反馈错误: {e}")
                continue
    finally:
        manager.disconnect(client_id)
        logger.log(f"关闭反馈WebSocket连接: {client_id}")

@router.websocket("/senescence")
async def on_senescence(websocket: WebSocket):
    pass