from ray import remote
from .models.cell import TrainingInput

@remote
def predict(input_data: TrainingInput):
    # 深度学习模型预测远程任务
    # cell_type = input_data.cell_type
    # if cell_type == "stem":
    #     return predict_stem(input_data)
    # else:
    #     raise ValueError(f"Unknown cell type: {cell_type}")
    pass

def predict_stem(input_data: TrainingInput):
    # 干细胞预测逻辑
    hp = input_data.hp
    vision_surround = input_data.vision_surround
    life = input_data.life
