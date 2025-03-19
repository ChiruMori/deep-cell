from pydantic import BaseModel, Field
from typing import List

class TrainingInput(BaseModel):
    life: float
    hp: float
    vision_surround: List[int] = Field(..., max_length=6)

class TrainingOutput(BaseModel):
    direction: float = Field(..., ge=0, le=6.28319)  # 0-2π范围
    strength: float = Field(..., ge=0, le=1)

class DecisionResult(BaseModel):
    action: str
    probability: float