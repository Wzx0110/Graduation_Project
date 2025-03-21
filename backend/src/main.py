from fastapi import FastAPI, File, UploadFile
from backend.src.im2text import getimg2text
from PIL import Image
import torch
from fastapi.middleware.cors import CORSMiddleware  # 導入 CORS 中間件
from ollamaConect import getResponse
import io
from fastapi.responses import JSONResponse
app = FastAPI()# FastAPI 物件
# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源，或者指定具體的來源（例如 ["http://localhost:8000"]）
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有 HTTP 方法
    allow_headers=["*"],  # 允許所有 HTTP 頭
)
@app.post("/upload/")# 裝飾器
async def upload_image(file: UploadFile = File(...)):  # 接收前端上傳的圖片
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        text = getimg2text(image)
        response = getResponse(text)
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
'''
def index():
    image = "london.jpg"
    image = Image.open(image).convert("RGB")
    text = getimg2text(image)# 取得圖片描述
    return getResponse(text)# 回傳字典
我希望從前端獲取image
uvicorn main:app --reload
'''