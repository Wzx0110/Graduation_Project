import time
import cv2
import numpy as np
from PIL import Image
import gc
import torch
import io

from alignment import align_images, crop_images
from description import initialize_image_description_model, generate_image_description
from prompting import generate_transition_prompts
from generation import initialize_sd_pipeline, generate_image_sequence, interpolate_frames
from sub2 import create_video2_file
from subtitle_generate import generate_subtitle
from sub_audio import create_video_with_subtitles

INPUT_IMG1_PATH = "1.png"
INPUT_IMG2_PATH = "2.png"

# generation.py 的參數
GENERATION_TARGET_SIZE = (512, 512)  # 生成圖像的目標尺寸，建議符合 SD 模型的原生尺寸
GENERATION_STRENGTH = 0.3          # Img2Img 強度 (0.0 到 1.0，較低值表示允許更多變化)
GENERATION_GUIDANCE_SCALE = 12     # CFG 引導比例 (Classifier-Free Guidance)
GENERATION_NUM_INFERENCE_STEPS = 100  # Stable Diffusion 的去噪步數 (較少步數可加速生成)
GENERATION_NEGATIVE_PROMPT = "blurry, low quality, deformed, distorted, unreal, unscale, words"  # 負面提示詞，避免生成特定內容

NUM_INTERPOLATION_FRAMES = 100

# --- 裝置設定 (Device Setting) ---
# 自動偵測是否有可用的 CUDA GPU，否則使用 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置: {DEVICE}")

# --- 輔助函式 (Helper Functions) ---


def clear_gpu_memory():
    """釋放 GPU 記憶體"""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()  # 清空 PyTorch 的 CUDA 快取
    gc.collect()  # 執行 Python 的垃圾回收


def np_bgr_to_pil_rgb(np_image: np.ndarray) -> Image.Image:
    """將 NumPy BGR 格式影像轉換為 PIL RGB 格式影像"""
    if np_image is None:
        return None
    # 檢查影像是否為灰階 (2 維)
    if np_image.ndim == 2:
        # 先將灰階轉換為 BGR
        np_image_bgr = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
    elif np_image.ndim == 3 and np_image.shape[2] == 3:
        # 已經是 BGR 格式
        np_image_bgr = np_image
    else:
        # 處理非預期格式，例如可能是 BGRA
        print(f"警告：非預期的影像形狀 {np_image.shape}。嘗試轉換...")
        try:
            # 嘗試從常見的 BGRA 轉換
            np_image_bgr = cv2.cvtColor(np_image, cv2.COLOR_BGRA2BGR)
        except cv2.error:
            print("錯誤：無法將影像轉換為 BGR 格式。")
            return None  # 轉換失敗

    # 將 BGR 轉換為 RGB
    np_image_rgb = cv2.cvtColor(np_image_bgr, cv2.COLOR_BGR2RGB)
    # 從 NumPy 陣列創建 PIL Image 物件
    return Image.fromarray(np_image_rgb)


def pil_rgb_to_np_bgr(pil_image: Image.Image) -> np.ndarray:
    """將 PIL RGB 格式影像轉換為 NumPy BGR 格式影像 (OpenCV 常用格式)"""
    if pil_image is None:
        return None
    # 確保輸入是 RGB 格式
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    # 將 PIL Image 轉換為 NumPy 陣列 (RGB)
    np_image_rgb = np.array(pil_image)
    # 將 RGB 轉換為 BGR
    np_image_bgr = cv2.cvtColor(np_image_rgb, cv2.COLOR_RGB2BGR)
    return np_image_bgr


async def transtion(oldImage: Image.Image, newImage: Image.Image):
    main_start_time = time.time()

    # 轉成 bytes 格式，模擬原本從檔案讀取的方式
    with io.BytesIO() as output1, io.BytesIO() as output2:
        oldImage.save(output1, format="PNG")
        newImage.save(output2, format="PNG")
        img1_bytes = output1.getvalue()
        img2_bytes = output2.getvalue()

    print("成功取得上傳圖片資料")

    # 影像對齊
    align_start_time = time.time()
    aligned_img1_np, ref_img2_np = align_images(img1_bytes, img2_bytes)
    align_end_time = time.time()

    if aligned_img1_np is None or ref_img2_np is None:
        print("影像對齊失敗")
        exit()
    print(f"影像對齊，耗時 {align_end_time - align_start_time:.2f} 秒。")

    # 裁剪對齊後的圖片
    crop_start_time = time.time()
    cropped_aligned_np, cropped_ref_np = crop_images(
        aligned_img1_np, ref_img2_np)
    crop_end_time = time.time()

    if cropped_aligned_np is None or cropped_ref_np is None:
        print("裁剪圖片失敗")
        exit()
    print(f"圖片裁剪，耗時 {crop_end_time - crop_start_time:.2f} 秒。")

    # 將裁剪後的圖片轉換為 PIL RGB 格式
    start_image_pil = np_bgr_to_pil_rgb(cropped_aligned_np)
    end_image_pil = np_bgr_to_pil_rgb(cropped_ref_np)

    if start_image_pil is None or end_image_pil is None:
        print("將裁剪圖片轉換為 PIL 格式時失敗")
        exit()
    print("圖片成功轉換為 PIL 格式。")

    # 生成圖像描述
    desc_start_time = time.time()
    try:
        initialize_image_description_model()
        desc1 = generate_image_description(
            start_image_pil)
        clear_gpu_memory()
        desc2 = generate_image_description(
            end_image_pil)
        clear_gpu_memory()
    except Exception as e:
        print(f"在描述模型初始化或生成過程中發生錯誤: {e}")
        desc1, desc2 = None, None

    desc_end_time = time.time()

    if not desc1 or not desc2:
        print("描述生成失敗")
        exit()
    print(f"生成描述 1:\n{desc1}")
    print(f"生成描述 2:\n{desc2}")
    print(f"生成描述，耗時 {desc_end_time - desc_start_time:.2f} 秒。")

    # 生成過渡提示詞
    prompt_start_time = time.time()
    transition_steps_list = generate_transition_prompts(
        text1=desc1,
        text2=desc2,
        num_steps=10,
    )
    prompt_end_time = time.time()  #

    if not transition_steps_list:
        print("生成過渡提示詞失敗")
        exit()
    print(f"成功生成 {len(transition_steps_list)} 個過渡步驟。")
    print("生成的步驟:", transition_steps_list)
    print(f"提示詞生成，耗時 {prompt_end_time - prompt_start_time:.2f} 秒。")

    subtitle_start_time = time.time()
    subtitle_list = generate_subtitle(
        text1=desc1,
        text2=desc2,
    )
    subtitle_end_time = time.time()

    if not subtitle_list:
        print("生成字幕失敗")
        exit()
    print(f"成功生成字幕: {subtitle_list}")
    print(f"字幕生成，耗時 {subtitle_end_time - subtitle_start_time:.2f} 秒。")

    # 生成關鍵影格圖片
    gen_start_time = time.time()
    try:
        initialize_sd_pipeline()

        # 將 PIL 圖片調整為 Stable Diffusion 需要的目標尺寸
        start_image_sd_pil = start_image_pil.resize(
            GENERATION_TARGET_SIZE, Image.LANCZOS)
        end_image_sd_pil = end_image_pil.resize(
            GENERATION_TARGET_SIZE, Image.LANCZOS)

        keyframes_pil = generate_image_sequence(
            start_image=start_image_sd_pil,
            target_image=end_image_sd_pil,
            steps_list=transition_steps_list,
            target_size=GENERATION_TARGET_SIZE,
            strength=GENERATION_STRENGTH,
            guidance_scale=GENERATION_GUIDANCE_SCALE,
            num_inference_steps=GENERATION_NUM_INFERENCE_STEPS,
            negative_prompt=GENERATION_NEGATIVE_PROMPT
        )
    except Exception as e:
        # 捕捉 Stable Diffusion 初始化或生成過程中的任何錯誤
        print(f"在 Stable Diffusion 管線初始化或生成過程中發生錯誤: {e}")
        keyframes_pil = None  # 標記生成失敗
    finally:
        # 無論成功或失敗，都嘗試清理 GPU 記憶體
        clear_gpu_memory()

    gen_end_time = time.time()  # 記錄圖像生成結束時間

    if not keyframes_pil:
        print("未能生成關鍵影格圖片")
        exit()

    print(f"成功生成 {len(keyframes_pil)} 個關鍵影格 (包含起始與結束影格)。")
    print(f"生成關鍵影格，耗時 {gen_end_time - gen_start_time:.2f} 秒。")

    # 在生成的關鍵影格之間進行線性內插，以產生更平滑的過渡
    final_frames_pil = interpolate_frames(
        key_frames=keyframes_pil,
        num_interpolations=NUM_INTERPOLATION_FRAMES
    )
    print(f"內插後共得到 {len(final_frames_pil)} 個最終影格。")
    # create_video2_file(final_frames_pil, 20, subtitle_list)
    
    video_audio_path = await create_video_with_subtitles(final_frames_pil, subtitles=subtitle_list, fps=20)
    
    main_end_time = time.time()
    print(f"總執行時間: {main_end_time - main_start_time:.2f} 秒。")

    video_bytes_io = io.BytesIO()
    with open(video_audio_path, "rb") as f:
        video_bytes_io.write(f.read())
    video_bytes_io.seek(0)
    
    return video_bytes_io
