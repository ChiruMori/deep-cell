import math
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from collections import defaultdict

from fastapi.websockets import WebSocketState

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

# 经验回FFER区
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
        buffer_len = len(self.buffer)
        # 添加长度校验
        if buffer_len < batch_size:
            return []
        
        # 禁止重复采样
        indices = np.random.choice(buffer_len, batch_size, 
                                  p=np.array(self.priority)/sum(self.priority), 
                                  replace=False)
        
        # 转换为集合去重并排序
        sorted_indices = sorted(set(indices), reverse=True)
        
        res = [self.buffer[i] for i in indices]
        
        # 安全删除逻辑
        for idx in sorted_indices:
            del self.buffer[idx]
            del self.priority[idx]
            
        return res
    
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
optimizers = {k: torch.optim.AdamW(v.parameters(), lr=0.001, betas=(0.9, 0.999)) for k, v in models.items()}
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

# 损失计算
def angular_loss(pred, target):
    # 将动作分解为sin/cos分量
    pred_sin = pred[:, 0]
    pred_cos = pred[:, 1]
    target_rad = target[:, 0]  # 原始角度值
    target_sin = torch.sin(target_rad)
    target_cos = torch.cos(target_rad)
    
    # 计算余弦相似度损失
    cos_loss = 1 - (pred_sin * target_sin + pred_cos * target_cos).mean()
    return cos_loss

async def train_model(cell_type, batch_size=32, epochs=5):
    """按细胞类型训练模型"""
    global training_iterations, optimizers, models, experience_buffers
    
    if len(experience_buffers[cell_type]) < batch_size:
        return False
        
    try:
        # 从经验回FFER区采样并从缓冲区中移除
        batch = experience_buffers[cell_type].sample(batch_size)
        
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
        
        # 训练模型
        models[cell_type].train()  # 设置为训练模式
        
        for _ in range(epochs):
            # 清空梯度
            optimizers[cell_type].zero_grad()
            # 输入特征归一化
            states_tensor = (states_tensor - states_tensor.mean(dim=0)) / (states_tensor.std(dim=0) + 1e-8)
            # 添加输入噪声
            noisy_states = states_tensor + torch.randn_like(states_tensor) * 0.1
            # 前向传播
            predictions = models[cell_type](noisy_states)
            # 损失计算
            strength_loss = torch.nn.functional.mse_loss(predictions[:, 2], actions_tensor[:, 1])
            kw_loss = torch.nn.functional.mse_loss(predictions[:, 3], actions_tensor[:, 2])
            # 熵奖励，鼓励探索
            entropy_bonus = -0.1 * (kw_loss * torch.log(kw_loss)).mean()
            total_loss = angular_loss(predictions, actions_tensor) + strength_loss * 0.5 + kw_loss * 0.3 + entropy_bonus

            # 根据奖励加权损失
            # 在训练循环中修改损失计算和优化策略
            reward_coef = torch.sigmoid(rewards_tensor) * 0.2  # 限制奖励影响在0-0.2之间
            weighted_loss = (total_loss * (1 + reward_coef)).mean()
            # 梯度裁剪力度
            torch.nn.utils.clip_grad_norm_(models[cell_type].parameters(), max_norm=0.5)
            
            # 学习率衰减
            if training_iterations[cell_type] % 10 == 0:
                params = [p.data.abs().mean() for p in models[cell_type].parameters()]
                logger.log(f"[{cell_type}] Params mean: {sum(params)/len(params):.4f}")
            
            weighted_loss.backward()
            optimizers[cell_type].step()

            # 在经验回放采样时增加随机探索率
            exploration_rate = max(0.1, 1 - training_iterations[cell_type] / 200)  # 200轮后保持10%探索
            if random.random() < exploration_rate:
                # 添加随机扰动到预测结果
                predictions += torch.randn_like(predictions) * 0.2
        
        # 训练完成后设置回评估模式
        models[cell_type].eval()
        # 训练完成后计数器更新
        training_iterations[cell_type] += 1
        print(f"[{cell_type}] 训练完成, 迭代次数: {training_iterations[cell_type]}")
        # 降低保存模型的频率
        if training_iterations[cell_type] % 20 == 0:
            print(f"[{cell_type}] 保存模型")
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
    print('Tick 连接建立')
    client_id = f"tick_{time.time()}"
    await manager.connect(websocket, client_id)
    logger.log(f"Tick WebSocket连接已建立: {client_id}")
    
    try:
        
        while websocket.client_state == WebSocketState.CONNECTED:
            try:
                # 设置接收超时
                data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                
                if isinstance(data, list) and len(data) > 0:
                    # 按细胞类型分组数据
                    grouped_data = defaultdict(list)
                    input_map = defaultdict(dict)
                    
                    for idx, item in enumerate(data):
                        try:
                            cell_type = item.get('c_type')
                            input_id = item.get('id', '')
                            
                            # 填充分组数据
                            grouped_data[cell_type].append([
                                cell_type,
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
                            print(f"未知细胞类型: {cell_type}")
                            continue
                        processed_data = []
                        for sample in type_data:
                            try:
                                # 确保每个元素都是浮点数
                                validated_sample = [float(v) for v in sample]
                                processed_data.append(validated_sample)
                            except (TypeError, ValueError) as ve:
                                print(f"数据格式错误: {sample} | 错误: {ve}")
                                traceback.print_exc()
                                # 添加默认值防止崩溃
                                processed_data.append([0.0] * 11)  # 输入维度为11
                        batch_tensor = torch.tensor(type_data, dtype=torch.float32)
                        with torch.no_grad():
                            predictions = models[parsed_cell_type](batch_tensor)
                        
                        # 处理预测结果（需要同步修改输出维度）
                        angle_sin = predictions[:, 0].tolist()
                        angle_cos = predictions[:, 1].tolist()
                        strengths = predictions[:, 2].tolist()
                        kws = predictions[:, 3].tolist()  # 注意索引位置变化
                        
                        # 构建响应
                        for idx, (input_id, _) in enumerate(input_map[cell_type].items()):
                            if idx < len(angle_sin):
                                actual_angle = math.atan2(angle_sin[idx], angle_cos[idx]) % (2 * math.pi)
                                responses.append({
                                    "id": input_id,
                                    "angle": actual_angle,  # 使用计算后的实际角度
                                    "strength": strengths[idx],
                                    "kw": kws[idx],
                                    "type": cell_type
                                })
                                
                        # 记录样本输出
                        if type_data:
                            sample_idx = random.randint(0, len(angle_sin)-1)
                            sample_angle = math.atan2(angle_sin[sample_idx], angle_cos[sample_idx]) % (2 * math.pi)
                            logger.log(f"[{parsed_cell_type}] Sample: angle={sample_angle:.2f} strength={strengths[sample_idx]:.2f}")
                    
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
                    traceback.print_exc()
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
                            # 添加到对应类型的经验缓冲区
                            experience = {
                                'state': state,
                                'action': action,
                                'reward': feedback.get('immediate_reward', 0),
                                'is_terminal': feedback.get('is_terminal', False)
                            }
                            experience_buffers[parsed_cell_type].add(experience)
                            # print('add', parsed_cell_type, len(experience_buffers[parsed_cell_type]))
                            
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

print(f"[路由注册] WebSocket端点已加载")