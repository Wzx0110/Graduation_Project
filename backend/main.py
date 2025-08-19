# backend/main.py

import uvicorn
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Literal
import asyncio
import time
import traceback
from PIL import Image
import io
import subprocess
import platform
import os

# 導入 PyTorch 和 diffusers 相關類別
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL, UniPCMultistepScheduler
import gc

from websockets.exceptions import ConnectionClosedError
from uvicorn.protocols.utils import ClientDisconnected
from starlette.websockets import WebSocketState

# 導入我們自己的核心模塊
from core.alignment import align_images_pipeline
from core.storyboard import storyboard_pipeline
from core.generation import generate_latent_previews, generate_frames_from_images
from core.interpolation import interpolate_frames_cli, unload_film_model
from core.synthesis import synthesis_pipeline_moviepy

# --- FastAPI 應用程式初始化 ---
app = FastAPI(title="ChronoWeaver API")

app.mount("/outputs", StaticFiles(directory="temp"), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- WebSocket 管理器 ---


class ConnectionManager:
    def __init__(self): self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket): await websocket.accept(
    ); self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def send_progress(self, major_step: str, status: str, substep: Dict[str, Any] = None):
        payload = {"major_step": major_step, "status": status}
        if substep:
            payload["substep"] = substep
        for connection in self.active_connections[:]:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_json(payload)
            except Exception as e:
                print(f"發送 WebSocket 消息失敗: {type(e).__name__}。正在移除無效連接。")
                self.disconnect(connection)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("客戶端 WebSocket 主動斷開連接。")
        manager.disconnect(websocket)

# --- 資源管理函數 ---
pipeline_instance = None
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"


def load_sd_pipeline_sync():
    global pipeline_instance
    if pipeline_instance is None:
        print("正在載入 SDXL Img2Img 模型...")
        try:
            vae = AutoencoderKL.from_pretrained(
                VAE_MODEL_ID, torch_dtype=torch.float16)
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, safety_checker=None)
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config)
            pipe.to("cuda")
            pipeline_instance = pipe
            print("模型載入完成。")
        except Exception as e:
            print(f"載入模型時發生錯誤: {e}")
            pipeline_instance = None
    return pipeline_instance


def unload_sd_pipeline_sync():
    """同步的卸載函數 (最終修正版)"""
    global pipeline_instance
    if pipeline_instance is not None:
        print("正在卸載 SD Pipeline...")

        # **核心修改點：直接刪除 GPU 上的物件，不再 to("cpu")**
        del pipeline_instance
        pipeline_instance = None

        # 強制垃圾回收
        gc.collect()
        # 強制清理 CUDA VRAM 快取
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("SD Pipeline 已卸載。")


def restart_ollama_service():
    print("正在重啟 Ollama 服務以釋放 VRAM...")
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.Popen(["ollama", "serve"],
                             creationflags=subprocess.CREATE_NO_WINDOW)
        else:  # macOS & Linux
            subprocess.run("pkill ollama", shell=True, check=False)
            subprocess.Popen("ollama serve", shell=True)
        print("Ollama 服務已發送重啟命令。")
        return True
    except Exception as e:
        print(f"重啟 Ollama 失敗: {e}")
        return False


def encode_pil_to_base64(image: Image.Image) -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, "PNG")
        return base64.b64encode(buffer.getvalue()).decode()

# --- 背景任務 ---


async def video_generation_task(session_id, img1_bytes, img2_bytes, prompt, strength, num_frames, strategy):
    loop = asyncio.get_running_loop()
    pipe = None
    generated_frames_for_synthesis = []
    all_frames_for_video = []

    try:
        await manager.send_progress("任務開始", "completed", {"name": "任務ID", "status": "completed", "text": session_id})

        # [流程點 1] 影像對齊
        await manager.send_progress("影像對齊", "running")
        alignment_results = await loop.run_in_executor(None, list, align_images_pipeline(img1_bytes, img2_bytes))
        final_alignment_result = next(
            (p["final_result"] for p in alignment_results if "final_result" in p), None)
        if final_alignment_result is None:
            raise Exception("影像對齊流程未能返回最終結果。")
        for progress in alignment_results:
            if "final_result" not in progress:
                await manager.send_progress("影像對齊", "running", substep=progress)
        aligned_img1_np, aligned_img2_np, _ = final_alignment_result
        await manager.send_progress("影像對齊", "completed")
        _, b1 = cv2.imencode('.png', aligned_img1_np)
        img1_bytes = b1.tobytes()
        _, b2 = cv2.imencode('.png', aligned_img2_np)
        img2_bytes = b2.tobytes()

        # [流程點 2] 故事腳本生成
        await manager.send_progress("故事腳本", "running")
        storyboard_results = await loop.run_in_executor(None, list, storyboard_pipeline(img1_bytes, img2_bytes, prompt, num_frames))
        storyboard_json = next(
            (p["final_result"] for p in storyboard_results if "final_result" in p), None)
        if storyboard_json is None:
            raise Exception("故事腳本生成流程未能返回最終結果。")
        for progress in storyboard_results:
            if "final_result" not in progress:
                await manager.send_progress("故事腳本", "running", substep=progress)
        await manager.send_progress("故事腳本", "completed")

        # [流程點 3] 資源清理
        await manager.send_progress("資源清理", "running", {"name": "釋放腳本模型資源", "status": "running"})
        await loop.run_in_executor(None, restart_ollama_service)
        await asyncio.sleep(5)
        await manager.send_progress("資源清理", "completed")

        # [流程點 4] 模型載入
        await manager.send_progress("模型載入", "running", {"name": "載入圖像生成模型", "status": "running"})
        pipe = await loop.run_in_executor(None, load_sd_pipeline_sync)
        if pipe is None:
            raise RuntimeError("模型載入失敗。")
        await manager.send_progress("模型載入", "completed")

        img1_pil = Image.open(io.BytesIO(img1_bytes))
        img2_pil = Image.open(io.BytesIO(img2_bytes))

        if strategy == "basic":
            # [流程點 5] 潛在空間插值預覽
            await manager.send_progress("潛在空間插值", "running")
            latent_results = await loop.run_in_executor(None, list, generate_latent_previews(pipe, img1_pil, img2_pil, num_frames))
            interpolated_images = next(
                (p["final_result"] for p in latent_results if "final_result" in p), None)
            if interpolated_images is None:
                raise Exception("潛在空間插值失敗。")
            for progress in latent_results:
                if "final_result" not in progress:
                    bIO = io.BytesIO()
                    progress["image"].save(bIO, "PNG")
                    b64 = base64.b64encode(bIO.getvalue()).decode()
                    substep = {"name": f"插值預覽 ({progress['frame_index']+1}/{progress['total_frames']})",
                               "status": "completed", "previews": [f"data:image/png;base64,{b64}"]}
                    await manager.send_progress("潛在空間插值", "running", substep=substep)
            await manager.send_progress("潛在空間插值", "completed")

            # [流程點 6] 圖像生成
            await manager.send_progress("圖像生成", "running")
            frame_results = await loop.run_in_executor(None, list, generate_frames_from_images(pipe, interpolated_images, storyboard_json, strength))
            generated_frames_for_synthesis = next(
                (p["final_result"] for p in frame_results if "final_result" in p), None)
            if generated_frames_for_synthesis is None:
                raise Exception("圖像生成失敗。")
            for progress in frame_results:
                if "final_result" not in progress:
                    bIO = io.BytesIO()
                    progress["image"].save(bIO, "PNG")
                    b64 = base64.b64encode(bIO.getvalue()).decode()
                    substep = {"name": f"生成第 {progress['frame_index']+1}/{progress['total_frames']} 幀",
                               "status": "completed", "previews": [f"data:image/png;base64,{b64}"]}
                    await manager.send_progress("圖像生成", "running", substep=substep)
            await manager.send_progress("圖像生成", "completed")

            keyframes_for_interpolation = [
                img1_pil.resize((1024, 1024), Image.LANCZOS)] + generated_frames_for_synthesis + [img2_pil.resize((1024, 1024), Image.LANCZOS)]

        # **[流程點 7] 影片插幀**
        await manager.send_progress("影片插幀", "running", {"name": "準備插幀文件...", "status": "running"})
        interpolation_results = await loop.run_in_executor(None, list, interpolate_frames_cli(keyframes_for_interpolation, 2))
        all_frames_for_video = next(
            (p["final_result"] for p in interpolation_results if "final_result" in p), None)
        if all_frames_for_video is None:
            raise Exception("影片插幀失敗。")
        substep = {
            "name": f"插幀完成，共 {len(all_frames_for_video)} 幀", "status": "completed"}
        await manager.send_progress("影片插幀", "completed", substep=substep)

        # [流程點 8] 影片合成
        await manager.send_progress("影片合成", "running")

        synthesis_generator = synthesis_pipeline_moviepy(
            all_frames_for_video,
            storyboard_json,
            session_id
        )

        final_video_path = None
        # 使用 run_in_executor 來運行耗時的 MoviePy 寫入操作
        synthesis_results = await loop.run_in_executor(None, list, synthesis_generator)
        final_video_path = next(
            (p["final_result"] for p in synthesis_results if "final_result" in p), None)

        # 回報中間進度
        for progress in synthesis_results:
            if "final_result" not in progress:
                await manager.send_progress("影片合成", "running", substep=progress)

        if final_video_path is None:
            raise Exception("影片合成失敗。")

        # 將本地文件路徑轉換為前端可以訪問的 URL
        # e.g., temp\session_123\synthesis\final_video_no_audio.mp4
        # os.path.relpath 會計算相對路徑
        relative_video_path = os.path.relpath(final_video_path, "temp")
        # 確保在 URL 中使用正斜線 '/'
        video_url = f"/outputs/{relative_video_path.replace(os.path.sep, '/')}"
        full_video_url = f"http://localhost:8000{video_url}"

        print(f"影片合成完畢。準備發送完整 URL 給前端: {full_video_url}")
        substep = {"name": "合成完成", "status": "completed",
                   "video_url": full_video_url}
        await manager.send_progress("影片合成", "completed", substep=substep)

        # 在完成時也發送一次URL
        await manager.send_progress("任務完成", "completed", {"video_url": full_video_url})
        print(f"--- [{session_id}] 任務流程成功結束 ---")

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"--- [{session_id}] 背景任務發生嚴重錯誤 ---")
        print(error_details)
        await manager.send_progress("任務失敗", "failed", {"name": "嚴重錯誤", "status": "failed", "text": str(e)})
    finally:
        # **確保所有模型都被卸載**
        if pipeline_instance is not None:
            print("任務結束或出錯，卸載 SD 模型...")
            await loop.run_in_executor(None, unload_sd_pipeline_sync)
        # 因為 FILM 是外部進程，我們不需要從主程式卸載它，但可以保留函數以備未來使用
        # unload_film_model()

# --- API 路由定義 ---
GenerationStrategy = Literal["basic", "controlnet", "lora", "controlnet_lora"]


@app.post("/generate-video", status_code=202)
async def generate_video(background_tasks: BackgroundTasks, image1: UploadFile = File(...), image2: UploadFile = File(...), prompt: str = Form(...), strength: float = Form(...), num_frames: int = Form(...), strategy: GenerationStrategy = Form(...)):
    session_id = f"session_{int(time.time())}"
    background_tasks.add_task(video_generation_task, session_id=session_id, img1_bytes=await image1.read(), img2_bytes=await image2.read(), prompt=prompt, strength=strength, num_frames=num_frames, strategy=strategy)
    return {"message": "Accepted: Video generation started in the background.", "session_id": session_id}

# --- 啟動伺服器 ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
