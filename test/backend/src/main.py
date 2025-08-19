from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from fastapi.middleware.cors import CORSMiddleware  # 導入 CORS 中間件
import io
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from transition import transition

app = FastAPI()
oringin = [
    "*"
    # 如果有前端，可以添加前端的地址           ]
]
# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=oringin,
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头
)
# @app.post("/upload/")# 裝飾器
# async def upload_image(file: UploadFile = File(...)):  # 接收前端上傳的圖片
#     try:
#         image_data = await file.read()
#         image = Image.open(io.BytesIO(image_data)).convert("RGB")
#         text = getimg2text(image)
#         response = getResponse(text)
#         return JSONResponse(content=response)

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=400)   

@app.post("/Process/")
async def Process(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        image_data1 = await image1.read()
        image_data2 = await image2.read()
        oldImage = Image.open(io.BytesIO(image_data1)).convert("RGB")
        newImage = Image.open(io.BytesIO(image_data2)).convert("RGB")

        video_io = await transition(oldImage, newImage)

        return StreamingResponse(video_io, media_type="video/mp4", headers={
            "Content-Disposition": "attachment; filename=transition.mp4"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    

# uvicorn main:app --reload
'''
def index():
    image = "london.jpg"
    image = Image.open(image).convert("RGB")
    text = getimg2text(image)# 取得圖片描述
    return getResponse(text)# 回傳字典
我希望從前端獲取image
'''