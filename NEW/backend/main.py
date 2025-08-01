# backend/main.py

import uvicorn
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import asyncio
import time
import traceback  # 引入 traceback 模塊以打印詳細錯誤

from core.alignment import align_images_pipeline
from core.storyboard import storyboard_pipeline

# ... (FastAPI app, CORSMiddleware, ConnectionManager class 都不變) ...

# 這裡我把之前的 ConnectionManager 也貼上來，確保完整性
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_progress(self, major_step: str, status: str, substep: Dict[str, Any] = None):
        payload = {"major_step": major_step, "status": status}
        if substep:
            payload["substep"] = substep

        for connection in self.active_connections:
            await connection.send_json(payload)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/generate-video")
async def generate_video(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    prompt: str = Form(...),
    strength: float = Form(...),
    num_frames: int = Form(...)
):
    try:
        await manager.send_progress(major_step="任務開始", status="completed")

        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        await manager.send_progress(major_step="影像對齊", status="running")

        # **--- 修正點：使用同步 for 迴圈 ---**
        alignment_generator = align_images_pipeline(img1_bytes, img2_bytes)

        final_result_data = None

        # 使用普通的 for 迴圈
        for progress in alignment_generator:
            if "final_result" in progress:
                # 獲取最終結果並保存，然後跳出循環
                final_result_data = progress["final_result"]
                break

            # 正常發送進度
            await manager.send_progress(
                major_step="影像對齊", status="running", substep=progress
            )
            # 在同步循環中，短暫地將控制權交還給事件循環，避免阻塞
            await asyncio.sleep(0.01)

        if final_result_data is None:
            raise Exception("Alignment pipeline did not yield a final result.")

        aligned_img1, aligned_img2, alignment_was_successful = final_result_data

        if alignment_was_successful:
            await manager.send_progress(major_step="影像對齊", status="completed")
        else:
            await manager.send_progress(major_step="影像對齊", status="failed")

         # 將對齊後的圖像轉回 bytes 以供下一步使用
        _, buffer1 = cv2.imencode('.png', aligned_img1)
        aligned_img1_bytes = buffer1.tobytes()
        _, buffer2 = cv2.imencode('.png', aligned_img2)
        aligned_img2_bytes = buffer2.tobytes()

       # --- [流程點 2] 故事腳本生成 (簡化後的調用) ---
        await manager.send_progress(major_step="故事腳本", status="running")

        # 直接調用 pipeline，傳入兩張圖的 bytes
        storyboard_generator = storyboard_pipeline(
            aligned_img1_bytes,
            aligned_img2_bytes,
            prompt,
            num_frames
        )
        storyboard_json = None
        for progress in storyboard_generator:
            if "final_result" in progress:
                storyboard_json = progress["final_result"]
                break
            # 現在只有一個細項了，所以這個循環其實只會跑兩次（running -> completed）
            await manager.send_progress(major_step="故事腳本", status="running", substep=progress)
            await asyncio.sleep(0.01)

        if storyboard_json is None:
            raise Exception("Storyboard pipeline failed.")
        await manager.send_progress(major_step="故事腳本", status="completed")
        # --- 任務完成 ---
        await manager.send_progress(major_step="任務完成", status="completed")

        # 返回生成的腳本以供除錯
        return JSONResponse(status_code=200, content=storyboard_json)

    except Exception as e:
        # **增加更詳細的錯誤日誌**
        print("--- AN ERROR OCCURRED ---")
        traceback.print_exc()  # 這會打印完整的錯誤堆疊
        print("-------------------------")
        await manager.send_progress(major_step="任務失敗", status="failed", substep={"name": "嚴重錯誤", "status": "failed", "text": str(e)})
        return JSONResponse(status_code=500, content={"message": f"伺服器錯誤: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
