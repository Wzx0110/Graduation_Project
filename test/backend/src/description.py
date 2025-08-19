from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import time
from typing import Optional

# 用於模型和處理器的全域變數
IMG_DESC_MODEL = None
IMG_DESC_PROCESSOR = None
IMG_DESC_MODEL_ID = 'microsoft/Florence-2-large'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_image_description_model():
    """初始化 Florence-2 圖片描述模型和處理器"""
    global IMG_DESC_MODEL, IMG_DESC_PROCESSOR
    if IMG_DESC_MODEL is None or IMG_DESC_PROCESSOR is None:  # 只有當模型未載入時才執行
        try:
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            IMG_DESC_MODEL = AutoModelForCausalLM.from_pretrained(
                IMG_DESC_MODEL_ID,
                trust_remote_code=True,  # 信任遠端程式碼
                torch_dtype=dtype  # 在 GPU 上使用 float16 以加速推理
            ).eval().to(DEVICE)  # 設定為評估模式並移到指定設備
            IMG_DESC_PROCESSOR = AutoProcessor.from_pretrained(
                IMG_DESC_MODEL_ID,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"模型初始化失敗：{e}")
            IMG_DESC_MODEL = None
            IMG_DESC_PROCESSOR = None


def generate_image_description(image: Image.Image, task_prompt: str = '<MORE_DETAILED_CAPTION>') -> Optional[str]:
    """
    使用模型為給定的 PIL 圖片生成描述。

    Args:
        image: PIL Image 物件。
        task_prompt: 圖片描述任務的提示詞 (例如 '<MORE_DETAILED_CAPTION>', '<CAPTION>', 等)。

    Returns:
        生成的描述字串，如果發生錯誤則回傳 None。
    """
    global IMG_DESC_MODEL, IMG_DESC_PROCESSOR

    if image is None:
        print("錯誤：輸入圖片為 None。")
        return None

    try:
        # 確保圖片是 RGB 格式
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 使用 processor 處理文字提示和圖片
        inputs = IMG_DESC_PROCESSOR(
            text=task_prompt,
            images=image,
            return_tensors="pt"  # 回傳 PyTorch 張量
        ).to(DEVICE)

        if DEVICE == "cuda":
            inputs = inputs.to(torch.float16)

        # 生成描述
        # 在不計算梯度的情況下執行，以節省記憶體
        with torch.no_grad():
            generated_ids = IMG_DESC_MODEL.generate(
                input_ids=inputs["input_ids"],       # 文字輸入的 token IDs
                pixel_values=inputs["pixel_values"],  # 圖片輸入的像素值
                max_new_tokens=1024,                 # 限制最大生成 token 數量
                early_stopping=False,                # 是否提前停止生成
                do_sample=False,                     # 不使用取樣，確保結果一致性
                num_beams=3,                         # 使用 Beam Search 提高品質
            )

        # 解碼生成的 token IDs 為文字
        generated_text = IMG_DESC_PROCESSOR.batch_decode(
            generated_ids, skip_special_tokens=False)[0]  # 不跳過特殊 token，以便後處理

        # 後處理生成結果
        parsed_output = IMG_DESC_PROCESSOR.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        # 結果是一個字典，{'<MORE_DETAILED_CAPTION>': '描述文字'}
        # 提取描述字串
        description = parsed_output.get(task_prompt)
        print(f"生成的描述：{description}")
        return description

    except Exception as e:
        print(f"生成圖片描述時發生錯誤：{e}")
        return None
    finally:
        # 清除 GPU 記憶體以釋放資源
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# 測試
if __name__ == '__main__':
    start_time = time.time()
    initialize_image_description_model()

    image_path = "../../../assets/test_images/1.png"
    img1 = Image.open(image_path).convert("RGB")
    image_path = "../../../assets/test_images/2.png"
    img2 = Image.open(image_path).convert("RGB")

    detailed_description_1 = generate_image_description(
        img1, task_prompt='<MORE_DETAILED_CAPTION>')
    detailed_description_2 = generate_image_description(
        img2, task_prompt='<MORE_DETAILED_CAPTION>')
    print(f"img1詳細描述:{detailed_description_1}")
    print(f"img2詳細描述:{detailed_description_2}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"執行時間：{execution_time:.2f} 秒")
