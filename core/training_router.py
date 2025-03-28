from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from collections import defaultdict

from fastapi.websockets import WebSocketState

from core.models import cell
from core.models.cell import CELL_TYPE_MAP
from .nn_models.network import CellNetwork
import torch
from collections import deque
import random
import traceback
import asyncio
import os
import time
import numpy as np

router = APIRouter()

# 经验回放缓冲区
class ExperienceBuffer:
    def __init__(self, max_size=5000):
        # 缓冲区
        self.buffer = deque(maxlen=max_size)
        self.priority = deque(maxlen=max_size)
        
    def add(self, experience, priority=1.0):
        self.buffer.append(experience)
        self.priority.append(priority)
        
    def sample(self, batch_size):
        # 使用优先级采样
        probabilities = np.array(self.priority) / sum(self.priority)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[i] for i in indices]
    
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
models = {
    'cancer': CellNetwork('cancer'),
    'erythrocyte': CellNetwork('erythrocyte'),
    'alveolar': CellNetwork('alveolar'),
    'stem': CellNetwork('stem')
}
# AdamW优化器
optimizers = {k: torch.optim.AdamW(v.parameters(), lr=0.001) for k, v in models.items()}
experience_buffers = {k: ExperienceBuffer() for k in models.keys()}
training_iterations = defaultdict(int)  # 改为按类型记录训练次数
active_connections = set()  # 跟踪活动连接

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

async def train_model(cell_type, batch_size=32, epochs=5):
    """按细胞类型训练模型"""
    global training_iterations, optimizers, models, experience_buffers
    
    if len(experience_buffers[cell_type]) < batch_size:
        return False
        
    try:
        # 从经验回FFER区采样
        batch = experience_buffers[cell_type].sample(batch_size)
        
        print("T2", cell_type)
        
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
                    *state['surround'],
                    *state['speed']
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
        
        print("T3", cell_type)
        
        # 训练模型
        models[cell_type].train()  # 设置为训练模式
        
        for _ in range(epochs):
            # 清空梯度
            optimizers[cell_type].zero_grad()
            
            # 前向传播
            predictions = models[cell_type](states_tensor)
            
            # 确保维度匹配
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(0)
                
            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(0)
            
            # 损失计算
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(predictions, actions_tensor)  # 统一计算三个维度的损失
            
            # 根据奖励加权损失
            # 在训练循环中修改损失计算和优化策略
            weighted_loss = loss * (1 + 0.5 * rewards_tensor)  # 降低奖励影响系数
            weighted_loss = weighted_loss.mean()
            
            # 增加梯度裁剪力度
            torch.nn.utils.clip_grad_norm_(models[cell_type].parameters(), max_norm=1.0)
            
            # 添加学习率衰减
            if training_iterations[cell_type] % 100 == 0:
                for param_group in optimizers[cell_type].param_groups:
                    param_group['lr'] *= 0.95
            optimizers[cell_type].step()
        
        print("T4", cell_type)
        # 训练完成后设置回评估模式
        models[cell_type].eval()
        # 训练完成后计数器更新
        training_iterations[cell_type] += 1
        # 降低保存模型的频率
        if training_iterations[cell_type] % 20 == 0:
            save_path = f"models/{cell_type}_model_{training_iterations[cell_type]}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': models[cell_type].state_dict(),
                'optimizer_state_dict': optimizers[cell_type].state_dict(),
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
                    # 按细胞类型分组数据
                    grouped_data = defaultdict(list)
                    input_map = defaultdict(dict)
                    
                    for idx, item in enumerate(data):
                        try:
                            cell_type = item.get('c_type', 'cancer')  # 获取细胞类型
                            input_id = item.get('id', '')
                            
                            # 填充分组数据
                            grouped_data[cell_type].append([
                                item.get('c_type', 0),
                                item.get('life', 0),
                                item.get('hp', 0),
                                *item.get('surround', [0]*6)[:6],  # 确保长度6
                                *item.get('speed', [0]*2)[:2],  # 确保长度2
                            ])
                            
                            # 记录输入映射关系
                            if not input_map[cell_type]:
                                input_map[cell_type] = {}
                            input_map[cell_type][input_id] = idx
                        except Exception as item_e:
                            logger.log(f"处理请求项错误: {item_e}")
                            continue
                    
                    responses = []
                    # 按类型批量推理
                    for cell_type, type_data in grouped_data.items():
                        parsed_cell_type = CELL_TYPE_MAP[cell_type]
                        # print(f'处理类型: {parsed_cell_type}')
                        if parsed_cell_type not in models:
                            logger.log(f"未知细胞类型: {cell_type}")
                            continue
                        processed_data = []
                        for sample in type_data:
                            try:
                                # 确保每个元素都是浮点数
                                validated_sample = [float(v) for v in sample]
                                processed_data.append(validated_sample)
                            except (TypeError, ValueError) as ve:
                                logger.log(f"数据格式错误: {sample} | 错误: {ve}")
                                # 添加默认值防止崩溃
                                processed_data.append([0.0] * 11)  # 输入维度为11
                        batch_tensor = torch.tensor(type_data, dtype=torch.float32)
                        with torch.no_grad():
                            predictions = models[parsed_cell_type](batch_tensor)
                        
                        # 处理预测结果
                        directions = predictions[:, 0].tolist()
                        strengths = predictions[:, 1].tolist()
                        kws = predictions[:, 2].tolist()
                        # print('预测完成', len(directions), len(strengths), len(kws), len(input_map[cell_type]))
                        
                        # 构建响应
                        for idx, (input_id, orig_idx) in enumerate(input_map[cell_type].items()):
                            if idx < len(directions):
                                responses.append({
                                    "id": input_id,
                                    "angle": directions[idx],
                                    "strength": strengths[idx],
                                    "kw": kws[idx],
                                    "type": cell_type
                                })
                                
                        # 记录样本输出
                        if type_data:
                            sample_idx = random.randint(0, len(directions)-1)
                            logger.log(f"[{parsed_cell_type}] Sample: angle={directions[sample_idx]:.2f} strength={strengths[sample_idx]:.2f}")
                    
                    # 发送响应
                    if responses:
                        try:
                            # print('发送响应')
                            await websocket.send_json(responses)
                            # print('响应发送完成')
                        except Exception as send_e:
                            logger.log(f"发送响应错误: {send_e}")
                            traceback.print_exc()
                            break
                    else:
                        print('没有响应数据', responses)
            except asyncio.TimeoutError:
                # 超时检查连接是否仍然有效
                try:
                    print('Tick 等待数据')
                    await websocket.receive_text()
                    continue
                except:
                    # 连接可能已断开
                    break
            except WebSocketDisconnect:
                logger.log(f"WebSocket连接断开: {client_id}")
                break
            except Exception as e:
                logger.log(f"Tick处理错误: {e}")
                traceback.print_exc()
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
                if isinstance(message, list):
                    for feedback in message:
                        try:
                            state = feedback.get('state', {})
                            cell_type = state.get('c_type')
                            action = feedback.get('action', {})
                            parsed_cell_type = CELL_TYPE_MAP[cell_type]
                            
                            # 验证必要字段
                            required_fields = ['c_type', 'life', 'hp', 'surround', 'speed']
                            if not all(k in state for k in required_fields):
                                raise ValueError("Missing required fields in state")
                                
                            print('A1.2', cell_type)
                            # 添加到对应类型的经验缓冲区
                            experience = {
                                'state': state,
                                'action': action,
                                'reward': feedback.get('immediate_reward', 0),
                                'is_terminal': feedback.get('is_terminal', False)
                            }
                            experience_buffers[parsed_cell_type].add(experience)
                            
                            # 触发对应类型的训练
                            if len(experience_buffers[parsed_cell_type]) >= 32:
                                await train_model(parsed_cell_type)
                                
                        except Exception as e:
                            logger.log(f"反馈处理错误: {e}")
                            continue
            except asyncio.TimeoutError:
                # 超时继续等待
                print(f"反馈等待数据...")
                continue
            except WebSocketDisconnect:
                logger.log(f"反馈WebSocket客户端连接断开: {client_id}")
                break
            except Exception as e:
                logger.log(f"处理反馈错误: {e}")
                continue
    except WebSocketDisconnect as e:
        logger.log(f"连接异常断开: {e}")
    finally:
        manager.disconnect(client_id)
        logger.log(f"关闭反馈WebSocket连接: {client_id}")
