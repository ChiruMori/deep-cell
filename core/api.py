# from fastapi import APIRouter
# from .model_service import predict
# from .models.cell import TrainingInput, DecisionResult

# router = APIRouter()

# @router.post("/cell/decision", response_model=DecisionResult)
# async def make_decision(cell_state: TrainingInput):
#     """
#     细胞行为决策接口
    
    
#     输入当前细胞状态，返回决策结果
#     """
#     # 调用Ray远程任务进行分布式计算
#     prediction = await predict.remote(cell_state.model_dump())
#     return {"action": prediction["action"], "probability": prediction["probability"]}