from fastapi import APIRouter, WebSocket
from .models.cell import TrainingInput, TrainingOutput
from .nn_models.stem import StemNetwork
from typing import Dict
import torch

router = APIRouter(prefix="/training", tags=["training"])

training_sessions: Dict[str, WebSocket] = {}

@router.websocket("/tick")
async def handle_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            input = TrainingInput(**data)
            # 初始化神经网络模型
            model = StemNetwork()
            model.eval()
            
            # 转换输入为张量
            input_tensor = torch.tensor([
                input.life,
                input.hp,
                # 展开数组（6个元素），每个元素作为一个输入
                *input.vision_surround
            ], dtype=torch.float32)
            
            # 执行推理
            with torch.no_grad():
                prediction = model(input_tensor)
            
            print(prediction)
            # 解析输出结果
            direction = prediction[0].item()
            strength = torch.sigmoid(prediction[1]).item()
            output = TrainingOutput(direction=direction, strength=strength)
            await websocket.send_json(output.model_dump())
    except Exception as e:
        print(f"WebSocket error: {e}")

@router.websocket("/apoptosis")
async def on_apoptosis(websocket: WebSocket):
    # 反馈到神经网络（实际寿命、子代个数、剩余生命），直接灌给神经网络
    # 神经网络会根据反馈进行学习
    await websocket.accept()
    

@router.websocket("/senescence")
async def on_senescence(websocket: WebSocket):
    pass