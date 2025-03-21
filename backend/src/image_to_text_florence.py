from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image


def init_model():
    try:
        model_id = 'microsoft/Florence-2-large'
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto"
        ).eval().to("cuda")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        return model, processor
    except Exception as e:
        print(f"模型初始化失敗: {e}")
        return None, None


def load_image(file_path):
    try:
        return Image.open(file_path).convert("RGB")
    except Exception as e:
        print(f"載入圖片失敗: {e}")
        return None


def run_example(model, processor, image, task_prompt, text_input=None):
    try:
        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to('cuda', torch.float16)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"].cuda(),  # 文字輸入
            pixel_values=inputs["pixel_values"].cuda(),  # 圖片輸入
            max_new_tokens=1024,  # 限制最大可生成的 token 數量
            early_stopping=False,  # 是否提前結束
            do_sample=False,  # 是否使用隨機取樣來產生不同的結果
            num_beams=3,
        )

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]
        return processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
    except Exception as e:
        print(f"執行過程發生錯誤: {e}")
        return None


def main():
    # 初始化模型和處理器
    model, processor = init_model()
    if model is None or processor is None:
        return

    # 載入本地圖片
    file_path = "C:\\Users\\user\\Desktop\\Jupyter\\1.png"

    image = load_image(file_path)

    image.show()
    if image is None:
        return

    # 提示詞
    task_prompt = '<MORE_DETAILED_CAPTION>'

    # 執行模型
    result = run_example(model, processor, image, task_prompt)
    if result:
        print(f"結果: {result}")

    # 釋放資源
    del model, processor  # 刪除變數
    torch.cuda.empty_cache()  # 清空 CUDA 緩存


if __name__ == '__main__':
    main()
