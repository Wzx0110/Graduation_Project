# backend/core/storyboard.py

import ollama
import json
import base64
from typing import Dict, Any, Generator

# --- 1. 核心 Prompt Engineering ---


def build_story_prompt(num_frames: int) -> str:
    """
    構建指導 Gemma 模型生成 "Context + Change" 格式腳本的 Prompt。
    """
    prompt = f"""
    You are an exceptionally meticulous visual analyst and a master prompt engineer.
    Your new, more advanced task is to create a visual story by describing the **incremental changes** between frames.

    **The "Context + Change" Prompting Philosophy:**
    Instead of describing the entire scene for each frame, you will break down the prompt into three parts:
    1.  `"context"`: Describe the static, unchanging elements of the scene. This part should remain nearly identical across all frames to ensure consistency.
    2.  `"change"`: Describe ONLY the specific, dynamic evolution for THIS frame. This is the core of the story.
    3.  `"quality"`: A fixed string for quality-boosting keywords.

    **Your Core Task:**
    1.  First, analyze the start and end images to identify the primary transformation (e.g., building construction, seasonal change, etc.).
    2.  Then, for each of the {num_frames} keyframes, generate a JSON object with the following structure:
        *   `"narration"`: A short story beat in **Traditional Chinese (繁體中文)**.
        *   `"prompt"`: An object containing the three keys: `"context"`, `"change"`, and `"quality"`.

    **Example of the structure for ONE keyframe (for a building construction scenario):**
    {{
        "narration": "鷹架開始被拆除，露出了建築的新外牆。",
        "prompt": {{
            "context": "A wide, static shot of a street with a red brick building on the left and a large apartment block in the center, under a partly cloudy sky.",
            "change": "The blue scaffolding and tarps on the central apartment block are now 50% removed, revealing the new tan-colored facade and windows underneath.",
            "quality": "(masterpiece, best quality), photorealistic, 8k, highly detailed"
        }}
    }}

    **Key Instructions:**
    *   The `"context"` MUST be consistent.
    *   The `"change"` MUST be gradual and logical from one frame to the next.
    *   The camera view MUST remain static.
    *   The final output MUST be a single, valid JSON object with a "keyframes" array.

    Now, apply this advanced "Context + Change" philosophy to the provided images and generate the complete JSON output.
    """
    return prompt


# --- 2. 主流程生成器 ---
def storyboard_pipeline(
    img1_bytes: bytes,
    img2_bytes: bytes,
    user_prompt: str,
    num_frames: int,
    model_name: str = "gemma3:12b"
) -> Generator[Dict[str, Any], None, None]:

    try:
        client = ollama.Client()

        # --- 細項 1: 直接生成分鏡腳本 ---
        yield {"name": "分析圖像並生成腳本", "status": "running", "text": f"使用 {model_name} 模型直接處理，請稍候..."}

        final_prompt = build_story_prompt(
            num_frames) + f"\n\nPlease also consider this story hint from the user: '{user_prompt}'"

        response = client.generate(
            model=model_name,
            prompt=final_prompt,
            images=[img1_bytes, img2_bytes],
            format='json',
            stream=False,
        )

        storyboard_str = response['response']

        # 為了除錯，打印從 Ollama 收到的原始字符串
        print("--- RAW RESPONSE FROM OLLAMA ---")
        print(storyboard_str)
        print("---------------------------------")

        storyboard_json = json.loads(storyboard_str)

        if "keyframes" not in storyboard_json or not isinstance(storyboard_json["keyframes"], list):
            raise ValueError("模型返回的JSON格式不正確，缺少 'keyframes' 陣列。")

        # 準備預覽和完整數據
        full_text = json.dumps(storyboard_json, indent=2, ensure_ascii=False)
        preview_text = full_text[:1000] + \
            ("\n..." if len(full_text) > 1000 else "")

        yield {
            "name": "分析圖像並生成腳本",
            "status": "completed",
            "text": preview_text,     # 用於 UI 顯示的預覽文本
            "full_data": full_text   # 用於下載的完整文本
        }

        # 在結束前，yield 最終結果
        yield {"final_result": storyboard_json}

    except Exception as e:
        error_message = f"在 storyboard_pipeline 中發生錯誤: {e}"
        print(error_message)
        yield {"name": "腳本生成失敗", "status": "failed", "text": error_message}
        yield {"final_result": None}
