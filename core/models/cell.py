from pydantic import BaseModel, Field
from typing import List

class TrainingInput(BaseModel):
    c_type: int
    life: float
    hp: float
    surround: List[int] = Field(min_length=6, max_length=6)
    id: str

class TrainingOutput(BaseModel):
    direction: float = Field(..., ge=0, le=6.28319)  # 0-2π范围
    strength: float = Field(..., ge=0, le=1)
    kw: float = Field(..., ge=0, le=1)  # 预留的通用字段，在部分细胞类型中使用
    id: str

class DecisionResult(BaseModel):
    action: str
    probability: float

class TrainingFeedback(BaseModel):
    id: str
    # 实际存活时间
    life_time: int
    # 产生的子代数量
    son_count: int
    # 剩余生命值
    life: int
    # 剩余养料
    hp: int
    # 最终的细胞类型
    type: str

# 细胞类型转换
CELL_TYPE_MAP = {
    1: 'stem',
    2: 'cancer',
    3: 'erythrocyte',
    4: 'alveolar',
}
