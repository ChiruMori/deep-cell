from fastapi import FastAPI
# from core.api import router
from .training_router import router as training_router
import ray

app = FastAPI()
# app.include_router(router)
app.include_router(training_router)

@ray.remote
def init_ray():
    # Ray初始化配置
    pass

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在应用启动时执行
    ray.init()
    init_ray.remote()
    yield

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)