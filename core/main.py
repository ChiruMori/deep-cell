from fastapi import FastAPI
from .training_router import router as training_router
import ray
from contextlib import asynccontextmanager
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在应用启动时执行
    print("初始化 Ray...")
    ray.init()
    yield
    # 应用关闭时执行清理操作
    print("关闭 Ray...")
    ray.shutdown()

# 将 lifespan 函数应用到 FastAPI 应用
app = FastAPI(lifespan=lifespan)
app.include_router(training_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)