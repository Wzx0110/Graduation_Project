# backend/core/synthesis.py

import os
import shutil
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Generator
from moviepy import ImageSequenceClip, AudioFileClip, CompositeVideoClip, TextClip

# TTS 相關的載入/卸載函數 (保持不變)
# ...


def synthesis_pipeline_moviepy(
    all_frames: List[Image.Image],
    storyboard: Dict[str, Any],
    session_id: str,
    output_fps: int = 6
) -> Generator[Dict[str, Any], None, None]:

    if not all_frames:
        yield {"final_result": None}
        return

    # --- 步驟 A: 準備工作目錄 ---
    WORK_DIR = os.path.join("temp", session_id, "synthesis")
    os.makedirs(WORK_DIR, exist_ok=True)
    FINAL_VIDEO_PATH = os.path.join(WORK_DIR, "final_video.mp4")

    try:
        # --- 步驟 B: 旁白生成 (暫時先不要，我們先合成無聲影片) ---
        # (未來的音訊和字幕邏輯可以放在這裡)
        yield {"name": "旁白與字幕 (跳過)", "status": "completed"}

        # --- 步驟 C: 使用 MoviePy 將 PIL Image 序列轉換為影片剪輯 ---
        yield {"name": "準備圖像序列", "status": "running"}

        # MoviePy 需要 numpy 陣列列表
        frames_np = [np.array(frame.convert("RGB")) for frame in all_frames]

        # 創建 ImageSequenceClip
        video_clip = ImageSequenceClip(frames_np, fps=output_fps)

        yield {"name": "圖像序列轉換", "status": "completed", "text": f"已將 {len(frames_np)} 幀載入 MoviePy。"}

        # --- 步驟 D: 寫入影片檔案 ---
        yield {"name": "影片編碼", "status": "running", "text": f"正在使用 MoviePy (ffmpeg) 編碼影片..."}

        # .write_videofile() 是一個阻塞操作，所以我們在 main.py 中會用 executor
        video_clip.write_videofile(
            FINAL_VIDEO_PATH,
            codec='libx264',
            audio=False,  # **目前沒有音訊**
            threads=os.cpu_count(),  # 使用所有 CPU 核心
            logger='bar'  # 顯示進度條
        )

        yield {"name": "影片編碼", "status": "completed", "text": "無聲影片已生成。"}

        yield {"final_result": FINAL_VIDEO_PATH}

    finally:
        # 清理 TTS 模型 (如果載入了的話)
        # unload_tts_model()
        pass
