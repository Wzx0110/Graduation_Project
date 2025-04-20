from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import time
import numpy as np
import IPython.display as display
import cv2

start_time = time.time()

# --- 1. 初始化 Stable Diffusion Img2Img 模型 ---
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5").to("cuda")

# --- 2. 載入並準備輸入圖片 ---
image_path1 = "1.png"
image_path2 = "2.png"
target_size = (512, 512)  # 設定目標圖片大小
try:
    image1 = Image.open(image_path1).convert("RGB").resize(target_size)
    image2 = Image.open(image_path2).convert("RGB").resize(target_size)
    print("圖片載入並調整大小完成。")
except FileNotFoundError:
    print(f"錯誤：找不到圖片檔案 '{image_path1}' 或 '{image_path2}'。請確保檔案在正確的路徑。")
    exit()
except Exception as e:
    print(f"載入或處理圖片時發生錯誤: {e}")
    exit()

# --- 3. 定義轉換步驟的文字描述 ---
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

# --- 4. 設定生成參數 ---
num_interpolations = 50  # 在每個關鍵步驟之間插入多少張內插圖片
num_steps = len(steps_list)  # 總共有多少個關鍵步驟
midpoint = num_steps // 2  # 找到步驟列表的中間點
strength_value = 0.35  # 控制每次轉換時保留多少原始圖片結構 (0.0 到 1.0 之間)
# 較低的值更依賴提示，變化可能更大；較高的值更保留原圖，變化更細微。

# --- 5. 生成關鍵影格 ---
image_list = []  # 儲存生成的關鍵影格

# **前半部分**：從 image1 開始，逐步向前變換
print("--- 處理前半部分 (從 Image 1 開始) ---")
image_a = image1
for i in range(midpoint):
    step = steps_list[i]
    print(f"處理步驟 {step['step']}/{num_steps}: {step['title']}")
    # 使用 Img2Img 模型生成下一步的圖片
    # prompt 是文字描述，image 是目前的圖片，strength 控制變化程度
    image_a = pipe(prompt=step['description'], image=image_a,
                   strength=strength_value, guidance_scale=7.5).images[0]
    image_list.append(image_a)
    # (可選) 儲存每個關鍵影格以供調試
    # image_a.save(f"keyframe_{step['step']}.png")

# **後半部分**：從 image2 開始，逐步向後（倒序描述）變換
print("\n--- 處理後半部分 (從 Image 2 倒推) ---")
image_b = image2
reverse_images = []  # 暫存後半部分的圖片，之後會反轉順序
# 從最後一個步驟的描述開始，倒著處理到中間點之後的那個步驟
for i in range(num_steps - 1, midpoint - 1, -1):
    step = steps_list[i]
    print(f"處理步驟 {step['step']}/{num_steps} (倒序): {step['title']}")
    # 使用 Img2Img 模型生成上一步的圖片（基於目標圖片和當前步驟描述）
    image_b = pipe(prompt=step['description'], image=image_b,
                   strength=strength_value, guidance_scale=7.5).images[0]
    reverse_images.insert(0, image_b)  # 將生成的圖片插入到列表的最前面，以保持正確的時序
    # (可選) 儲存每個關鍵影格以供調試
    # image_b.save(f"keyframe_{step['step']}_reverse.png")

# **合併前後部分**：將前半段和後半段（已反轉順序）的關鍵影格合併
image_list.extend(reverse_images)

# --- 6. 線性內插生成過渡影格 ---
final_image_list = []

# 在每兩個關鍵影格之間進行混合 (blend)
for i in range(len(image_list) - 1):
    final_image_list.append(image_list[i])  # 加入當前的關鍵影格
    # 使用 numpy 的 linspace 在 0 和 1 之間生成內插係數
    # +2 是為了包含頭尾，[1:-1] 去掉頭尾避免重複
    for j in np.linspace(0, 1, num=num_interpolations + 2)[1:-1]:
        # Image.blend 進行線性混合 alpha=j
        interpolated_image = Image.blend(
            image_list[i], image_list[i + 1], alpha=float(j))
        final_image_list.append(interpolated_image)

final_image_list.append(image_list[-1])  # 加入最後一個關鍵影格

# --- 7. (可選) 合併所有影格成一張大圖顯示 ---
cols = 10  # 每行顯示多少張圖片
if len(final_image_list) > 0:
    rows = (len(final_image_list) + cols - 1) // cols
    merged_width = cols * target_size[0]
    merged_height = rows * target_size[1]
    merged_image = Image.new("RGB", (merged_width, merged_height))

    for idx, img in enumerate(final_image_list):
        x_offset = (idx % cols) * target_size[0]
        y_offset = (idx // cols) * target_size[1]
        merged_image.paste(img.resize(target_size),
                           (x_offset, y_offset))  # 確保貼上的也是目標大小

    # 顯示合併後的圖片 (在 IPython 環境如 Jupyter Notebook 中)
    display.display(merged_image)


else:
    print("沒有影格可以合併成預覽圖。")


# --- 8. 將所有影格轉換成影片 ---
output_video_path = "animation_video.mp4"
fps = 15  # 設定影片的幀率 (Frames Per Second)
frame_width, frame_height = target_size  # 影片尺寸應與圖片大小一致

if len(final_image_list) > 0:
    # 設定影片編碼器 (mp4v 對應 .mp4 格式)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 創建 VideoWriter 物件
    out_video = cv2.VideoWriter(
        output_video_path, fourcc, fps, (frame_width, frame_height))

    # 遍歷所有影格圖片
    for img in final_image_list:
        # 將 PIL Image 物件轉換成 NumPy 陣列
        frame = np.array(img)
        # OpenCV 使用 BGR 色彩空間，而 PIL 是 RGB，需要轉換
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 將這一幀寫入影片檔
        out_video.write(frame)

    # 釋放 VideoWriter 資源
    out_video.release()
    print(f"--- 影片已成功儲存為 {output_video_path} ---")
else:
    print("沒有影格可以轉換成影片。")


# --- 9. 清理資源 ---
del pipe  # 刪除 Stable Diffusion 模型物件
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 釋放 GPU 記憶體
    print("GPU 記憶體已清理。")
else:
    print("未使用 CUDA，無需清理 GPU 記憶體。")

# --- 10. 輸出總執行時間 ---
end_time = time.time()
print(f"總執行時間: {end_time - start_time:.2f} 秒")
