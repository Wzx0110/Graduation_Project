from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
import torch
from PIL import Image
import time
import numpy as np
from typing import List, Dict, Any, Optional
import os  # 用於處理檔案路徑和目錄
import gc  # 垃圾回收模組

# 全域設定與模型 ID
SD_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
VAE_ID = "stabilityai/sd-vae-ft-mse"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "../../assets/keyframes"  # 輸出資料夾

# 全域變數保存 pipeline 和 VAE
SD_PIPELINE = None
SD_VAE = None

# 可選的效能/記憶體優化設定
# 在支援的 GPU 上使用半精度 (float16)，加速並節省記憶體
USE_FLOAT16 = True if DEVICE == "cuda" else False
# 如果安裝了 xformers，嘗試啟用以優化記憶體和速度
ENABLE_XFORMERS = True


def initialize_sd_pipeline():
    """初始化 Stable Diffusion VAE 和 Img2Img pipeline"""
    global SD_PIPELINE, SD_VAE
    dtype = torch.float16 if USE_FLOAT16 else torch.float32

    # 載入 VAE
    SD_VAE = AutoencoderKL.from_pretrained(
        VAE_ID, torch_dtype=dtype).to(DEVICE)
    pipe_args = {"torch_dtype": dtype}  # 設定 pipeline 參數
    if SD_VAE:  # 如果 VAE 成功載入，將其加入參數
        pipe_args["vae"] = SD_VAE
    else:
        print("未成功載入 VAE,使用預設 VAE")

    # 載入 Stable Diffusion Img2Img pipeline
    SD_PIPELINE = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL_ID, **pipe_args).to(DEVICE)

    # 如果設定了啟用 xformers
    if ENABLE_XFORMERS:
        try:
            # 嘗試啟用 xformers 的記憶體效率注意力機制
            SD_PIPELINE.enable_xformers_memory_efficient_attention()
            print("已啟用 xformers 記憶體優化。")
        except ImportError:
            print("未安裝 xformers 或版本不相容，未啟用記憶體優化。")
        except Exception as e:
            print(f"啟用 xformers 時發生錯誤: {e}")


def generate_image_sequence(
    start_image: Image.Image,
    target_image: Image.Image,
    steps_list: List[Dict[str, Any]],
    target_size: tuple = (512, 512),        # SD 2.1 Base 的原生尺寸
    strength: float = 0.35,                 # Img2Img 強度
    guidance_scale: float = 9.0,            # CFG 引導比例
    num_inference_steps: int = 50,          # 去噪步數 (降低以加速，原為 100)
    negative_prompt: str = "blurry, low quality, deformed, distorted, unrealistic, bad anatomy, disfigured",  # 負面提示詞
) -> Optional[List[Image.Image]]:
    """
    根據 steps_list 使用 Stable Diffusion 2.1 Img2Img (搭配 VAE 和前/後向策略)
    生成從 start_image 過渡到 target_image 的圖像序列

    Args:
        start_image: 起始點的 PIL Image 物件
        target_image: 結束點的 PIL Image 物件
        steps_list: 來自 prompting 模組的字典列表，包含 'step', 'title', 'description'
        target_size: 輸出圖像大小的元組 (寬, 高)
        strength: Img2Img 強度 (0.0 到 1.0)
        guidance_scale: CFG 比例
        num_inference_steps: SD 的去噪步數
        negative_prompt: 用於避免生成特定內容的負面提示詞

    Returns:
        包含所有關鍵影格 (含起始、結束、前後向生成) 的 PIL Image 物件列表
    """

    num_key_steps = len(steps_list)  # 實際步驟數量
    print(
        f"參數:target_size={target_size}, strength={strength}, guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}")

    try:
        # 準備初始圖片，轉換為 RGB 並調整大小
        image1_resized = start_image.convert(
            "RGB").resize(target_size, Image.LANCZOS)
        image2_resized = target_image.convert(
            "RGB").resize(target_size, Image.LANCZOS)

        midpoint_index = num_key_steps // 2  # 中間點的索引
        forward_keyframes = []               # 前半段正向生成的關鍵影格
        backward_key_frames = []             # 後半段反向生成的關鍵影格

        # 從起始圖片向後生成
        current_image = image1_resized
        for i in range(midpoint_index):  # 遍歷前半部分的描述步驟 (0 到 midpoint-1)
            step_info = steps_list[i]
            prompt = step_info["description"]
            print(
                f"處理步驟 {step_info['step']}/{num_key_steps} : {step_info['title']}")

            # 使用 pipeline 生成圖片
            with torch.no_grad():  # 節省記憶體
                result = SD_PIPELINE(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=current_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]

            generated_image = result.resize(target_size)
            forward_keyframes.append(generated_image)
            current_image = generated_image
            # 儲存keyframes
            generated_image.save(os.path.join(
                OUTPUT_DIR, f"keyframe_{step_info['step']:03d}.png"))

            # 清理 CUDA 快取
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # 從結束圖片向前生成
        current_image = image2_resized
        for i in range(num_key_steps - 1, midpoint_index - 1, -1):
            step_info = steps_list[i]
            prompt = step_info["description"]
            print(
                f"處理步驟 {step_info['step']}/{num_key_steps} : {step_info['title']}")

            with torch.no_grad():
                result = SD_PIPELINE(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=current_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]

            generated_image = result.resize(target_size)
            backward_key_frames.insert(0, generated_image)
            current_image = generated_image

            generated_image.save(os.path.join(
                OUTPUT_DIR, f"keyframe_{step_info['step']:03d}.png"))

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # 合併影格
        final_keyframe_sequence = [
            image1_resized] + forward_keyframes + backward_key_frames + [image2_resized]

        # 清理中間列表的記憶體
        del forward_keyframes
        del backward_key_frames
        gc.collect()

        return final_keyframe_sequence
    except Exception as e:
        print(f"圖像序列生成過程中發生錯誤：{e}")
    finally:
        # 函數結束時清理記憶體
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def interpolate_frames(key_frames: List[Image.Image], num_interpolations: int = 5) -> List[Image.Image]:
    """
    關鍵影格之間線性內插

    Args:
        key_frames: PIL Image 關鍵影格列表
        num_interpolations: 每對關鍵影格之間要插入的內插影格數量

    Returns:
        包含原始關鍵影格和內插影格的 PIL Image 列表
    """

    interpolated_list = []

    for i in range(len(key_frames) - 1):
        img_start = key_frames[i]
        img_end = key_frames[i + 1]
        interpolated_list.append(img_start)

        # num=num_interpolations + 2 確保包含端點，[1:-1] 去除端點
        alphas = np.linspace(0, 1, num=num_interpolations + 2)[1:-1]
        for alpha in alphas:
            interpolated_image = Image.blend(
                img_start, img_end, alpha=float(alpha))
            interpolated_list.append(interpolated_image)

    interpolated_list.append(key_frames[-1])  # 加入最後一個關鍵影格

    return interpolated_list


# 測試
if __name__ == '__main__':
    start_time = time.time()
    initialize_sd_pipeline()
    img1 = Image.open("1.png").convert("RGB").resize((512, 512), Image.LANCZOS)
    img2 = Image.open("2.png").convert("RGB").resize((512, 512), Image.LANCZOS)

    steps_list = [
        {"step": 1, "title": "Tarp Reconfiguration", "description": "The large blue tarp covering the entire wall of the building begins to shrink, retracting upwards as if being rolled up. The scaffolding holding it in place gradually lifts and retracts along with the tarp, revealing the underlying structure of the wall beneath. This process reveals portions of a white facade, matching the color seen on the finished apartment building."},
        {"step": 2, "title": "Wall Development", "description": "As more of the blue tarp is retracted, the walls beneath it start to gain definition and height. The exposed sections reveal bricks that begin to seamlessly transform into a smooth, light beige-colored surface typical of the completed apartment building's facade."},
        {"step": 3, "title": "Facade Formation", "description": "The wall continues to rise as more tarp is retracted and scaffolding disappears. The color transition completes, with the entire exposed section now uniformly light beige. Windows start to form along the wall in a grid-like pattern, aligning with those visible on the finished apartment building."},
        {"step": 4, "title": "Balcony Addition", "description": "As the facade becomes more defined, protruding platforms begin to emerge from the side of each window set. These are the beginnings of balconies, initially appearing as raw structures that gradually become more defined with metal railings and matching the grid-like pattern observed on the completed building."},
        {"step": 5, "title": "Palm Trees and Fence Adjustment", "description": "The palm trees in front of the construction site begin to shift positions, aligning themselves into a more organized formation. The white fence surrounding the building starts transforming, its sections expanding vertically to match the height of the newly formed apartment building."},
        {"step": 6, "title": "Completion of Building Height", "description": "The entire facade and balconies continue their development until the building reaches its full height as seen in the second image. The wall color is now consistent, windows and balcony railings are fully formed, and the structure matches the appearance of a completed apartment building."},
        {"step": 7, "title": "Sky and Background", "description": "The sky remains clear and blue throughout, with no significant changes. However, other buildings in the background slightly adjust their positions to reflect a more cohesive skyline consistent with the final scene of an established neighborhood."},
        {"step": 8, "title": "Street Lamp Installation", "description": "A street lamp appears on the right side of the image, gradually materializing as if being installed. Its base becomes visible first, followed by the pole and finally the light fixture at the top, completing its form to match that in the final scene."},
        {"step": 9, "title": "Final Touches", "description": "The last adjustments occur with minor details: the fence's lattice pattern becomes more pronounced, aligning perfectly with the one seen on the completed building. Any remaining construction elements are removed or blended into the background."},
        {"step": 10, "title": "Final Scene", "description": "The entire image now fully resembles the second description: an apartment building with multiple floors and balconies, painted in a light beige color with white facades. The sky is clear blue, palm trees are neatly arranged, and the lattice-patterned white fence encloses the scene, completed by the presence of the street lamp."}
    ]

    keyframes = generate_image_sequence(
        start_image=img1,
        target_image=img2,
        steps_list=steps_list,
        target_size=(512, 512),
        strength=0.35,
        guidance_scale=9.0,
        num_inference_steps=20,
    )

    interpolated = interpolate_frames(
        keyframes, num_interpolations=3)

    # 清理 GPU 記憶體
    del SD_PIPELINE
    SD_PIPELINE = None
    del SD_VAE
    SD_VAE = None

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"執行時間：{execution_time:.2f} 秒")
