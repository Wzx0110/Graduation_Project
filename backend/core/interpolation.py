# backend/core/interpolation.py

import subprocess
import os
import shutil
from PIL import Image
from typing import List, Generator, Dict, Any
import glob
import sys
import platform

# --- 1. 動態計算路徑 ---
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CORE_DIR)

FRAME_INTERPOLATION_PROJECT_PATH = os.path.join(
    BACKEND_DIR, "frame-interpolation")

if sys.platform == "win32":
    PYTHON_FOR_FILM_PATH = os.path.join(
        FRAME_INTERPOLATION_PROJECT_PATH, "venv_tf/Scripts/python.exe")
else:
    PYTHON_FOR_FILM_PATH = os.path.join(
        FRAME_INTERPOLATION_PROJECT_PATH, "venv_tf/bin/python")

# **重要：請將此路徑修改為你存放 `saved_model` 的實際位置**
FILM_MODEL_PATH = "C:/Users/user/pretrained_models/film_net/Style/saved_model"

TEMP_BASE_DIR = os.path.join(BACKEND_DIR, "temp")
TEMP_INPUT_DIR = os.path.join(TEMP_BASE_DIR, "interpolation_input")

# --- 2. 核心插幀函數 ---


def interpolate_frames_cli(
    keyframes: List[Image.Image],
    inter_frames: int
) -> Generator[Dict[str, Any], None, None]:

    if not keyframes:
        yield {"final_result": []}
        return

    # 準備臨時資料夾
    if os.path.exists(TEMP_INPUT_DIR):
        shutil.rmtree(TEMP_INPUT_DIR)
    os.makedirs(TEMP_INPUT_DIR, exist_ok=True)

    print(f"正在將 {len(keyframes)} 張關鍵幀保存到臨時資料夾...")
    for i, frame in enumerate(keyframes):
        frame.save(os.path.join(TEMP_INPUT_DIR, f"keyframe_{i:04d}.png"))

    times_to_interpolate = inter_frames + 1
    command = [
        PYTHON_FOR_FILM_PATH,
        "-m", "eval.interpolator_cli",
        "--pattern", TEMP_INPUT_DIR,
        "--model_path", FILM_MODEL_PATH,
        "--times_to_interpolate", str(times_to_interpolate),
    ]

    print(f"正在執行 FILM 插幀命令: {' '.join(command)}")

    process = subprocess.run(
        command,
        cwd=FRAME_INTERPOLATION_PROJECT_PATH,
        capture_output=True,
        text=True,
        encoding='latin-1'
    )

    if process.returncode != 0:
        print("--- FILM CLI 錯誤輸出 ---\n", process.stderr)
        shutil.rmtree(TEMP_INPUT_DIR)
        raise RuntimeError(f"影片插幀失敗，CLI 返回錯誤碼: {process.returncode}")

    print("--- FILM CLI 標準輸出 ---\n", process.stdout)

    result_dir = os.path.join(
        TEMP_INPUT_DIR, f"interpolated_frames")
    result_pattern = os.path.join(result_dir, "*.png")
    result_files = sorted(glob.glob(result_pattern))

    if not result_files:
        raise FileNotFoundError(f"在 '{result_dir}' 中找不到任何插幀後的圖片。")

    print(f"成功找到 {len(result_files)} 張插幀後的圖片。")
    output_frames = []
    for f_path in result_files:
        with Image.open(f_path) as img:
            # 標準化為 RGB 格式並複製到記憶體
            standardized_img = img.convert("RGB")
            output_frames.append(standardized_img.copy())

    print(f"讀取完成，總共 {len(output_frames)} 幀。")

    # 清理臨時資料夾
    try:
        shutil.rmtree(TEMP_INPUT_DIR)
        print("臨時資料夾已清理。")
    except OSError as e:
        print(f"清理臨時資料夾失敗: {e}")

    yield {"final_result": output_frames}


def unload_film_model():
    """
    因為 FILM 是通過外部進程調用的，主程式沒有載入模型到記憶體，
    所以這個函數理論上不需要做任何事。保留它是為了 API 的一致性。
    """
    print("FILM 是外部進程，無需在主程式中卸載。")
    pass
