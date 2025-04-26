import time
import json
import re
import ollama
from typing import List, Dict, Any, Optional


def ask_ollama(model_name: str, question: str) -> Optional[str]:
    """將問題發送到指定的 Ollama 模型"""
    try:
        # 使用 ollama.chat 函數與模型互動
        response = ollama.chat(model=model_name,
                               messages=[{'role': 'user', 'content': question}])
        answer = response['message']['content']
        print(answer)
        print("-------------------------\n")
        return answer
    except ollama.ResponseError as e:
        # 處理連接錯誤或模型未找到的錯誤
        print(f"連接 Ollama 或找不到模型 '{model_name}' 時發生錯誤：{e.error}")
        return None
    except Exception as e:
        # 處理其他可能的錯誤
        print(f"Ollama 請求期間發生錯誤：{e}")
        return None


def clean_json_response(answer: str) -> str:
    """提取 JSON 陣列內容，移除前後多餘的文字"""
    # 尋找 JSON 陣列 '[...]'，允許前後有空白字符，且跨越多行
    match = re.search(r"^\s*(\[.*?\])\s*$", answer,
                      re.DOTALL)  # 使用 DOTALL 讓 . 匹配換行符
    if match:
        json_str = match.group(1)
        # 基本驗證：檢查是否以 [ 開頭並以 ] 結尾
        if json_str.startswith('[') and json_str.endswith(']'):
            print("找到 JSON 陣列結構")
            return json_str
    else:
        # 嘗試在 markdown 中尋找 JSON
        match_md = re.search(r"```json\s*(\[.*?\])\s*```", answer, re.DOTALL)
        if match_md:
            json_str = match_md.group(1)
            if json_str.startswith('[') and json_str.endswith(']'):
                print("在 markdown 中找到 JSON 陣列結構")
                return json_str

    print("無法可靠地提取 JSON 陣列")
    # 如果找不到明確的 JSON 陣列，回傳清理過的原始字串
    # 移除可能存在的 ```json 和 ``` 標記
    answer = re.sub(r"^\s*```json\s*", "", answer).strip()
    answer = re.sub(r"\s*```\s*$", "", answer).strip()
    return answer


def parse_steps_json(json_string: str) -> Optional[List[Dict[str, Any]]]:
    """解析預期為步驟字典列表的 JSON 字串"""
    try:
        steps = json.loads(json_string)  # 解析 JSON
        if isinstance(steps, list) and all(isinstance(s, dict) for s in steps):
            # 結構驗證：檢查每個字典是否包含 'step', 'title', 'description'
            if all('step' in s and 'title' in s and 'description' in s for s in steps):
                print("JSON 解析成功且基本結構有效。")
                return steps
            else:
                print("JSON 已解析，但結構可能不正確")
                return steps
        else:
            print(f"解析的 JSON 不是字典列表：{type(steps)}")
            return None
    except json.JSONDecodeError as e:
        # 處理 JSON 解析錯誤
        print(f"JSON 解碼失敗 {e}")
        return None


def parse_steps_markdown(answer: str) -> List[Dict[str, Any]]:
    """解析 Markdown 格式的步驟"""
    # 正則表達式尋找 "### Step X: Title \n Description" 模式
    # ^\s*#+\s*: 匹配行首的 # (標題標記)，允許空白
    # Step\s+(\d+)\s*[:\-]?\s*: 匹配 "Step" 後跟數字允許冒號或破折號
    # (.*?)\s*\n: 匹配標題
    # (.*?): 匹配描述
    # (?=\n\s*#+\s*Step\s+\d+|\Z): 正向前瞻，確保匹配到下一個 Step 標題或字串結尾
    steps = re.findall(
        r"^\s*#+\s*Step\s+(\d+)\s*[:\-]?\s*(.*?)\s*\n(.*?)(?=\n\s*#+\s*Step\s+\d+|\Z)",
        answer,
        re.MULTILINE | re.DOTALL  # MULTILINE 讓 ^ 匹配每行開頭，DOTALL 讓 . 匹配換行符
    )
    parsed_steps = []
    if steps:
        for num, title, desc in steps:
            parsed_steps.append({
                "step": int(num),           # 步驟編號 (整數)
                "title": title.strip(),     # 標題 (去除前後空白)
                "description": desc.strip()  # 描述 (去除前後空白)
            })
        print(f"Markdown 解析找到 {len(parsed_steps)} 個步驟")
    else:
        print("Markdown 解析未能在預期格式中找到步驟")
    return parsed_steps


def generate_transition_prompts(text1: str, text2: str, num_steps: int = 10, model_name: str = 'phi4') -> Optional[List[Dict[str, Any]]]:
    """
    使用 Ollama 模型生成逐步過渡的提示詞

    Args:
        text1: 起始圖片的描述
        text2: 結束圖片的描述
        num_steps: 期望的過渡步驟數量
        model_name: 要使用的 Ollama 模型

    Returns:
        一個字典列表，每個字典代表一個步驟，包含 'step', 'title', 'description'，
        如果生成或解析失敗則回傳 None
    """

    question = f"""
Given two image descriptions:

First image: {text1}  
Second image: {text2}  

Describe a **gradual, step-by-step transformation of the entire scene** from the first image to the second image.  

- Each step should describe **the whole image**, not just specific objects or parts.  
- The transition should be **smooth and visually realistic**, ensuring that each step logically follows the previous one.  
- Focus only on **visible changes** in the image, avoiding construction techniques or non-visual details.  
- Provide **{num_steps} sequential steps**, each showing the full scene at that moment.  



IMPORTANT: Please output the response as a **valid JSON list** with the following structure:
[
    {{"step": 1, "title": "Step Title", "description": "Detailed explanation of the step."}},
    {{"step": 2, "title": "Step Title", "description": "Detailed explanation of the step."}},
    ...
]

DO NOT include extra text outside of this JSON format.
"""

    answer = ask_ollama(model_name, question)  # 呼叫 ask_ollama 獲取回應

    if answer is None:  # 如果 Ollama 請求失敗
        return None

    # 清理回應以分離 JSON
    cleaned_answer = clean_json_response(answer)

    # 將清理後的回應解析為 JSON
    steps_list = parse_steps_json(cleaned_answer)

    # 如果 JSON 解析失敗，嘗試使用 Markdown 解析作為回傳
    if steps_list is None:
        print("JSON 解析失敗。使用 Markdown回傳")
        steps_list = parse_steps_markdown(answer)  # 解析原始回應以尋找 Markdown

    print(f"成功解析出 {len(steps_list)} 個過渡步驟")
    return steps_list


# 測試
if __name__ == "__main__":
    start_time = time.time()
    test_text1 = "The image shows a construction site with a large blue tarp covering a building under construction. The tarp is covering the entire wall of the building and is secured with metal scaffolding. There are several palm trees in front of the tarp, and a white fence surrounding the building. The sky is clear and blue, and there are other buildings visible in the background. The image appears to be taken from a low angle, looking up at the construction site."
    test_text2 = "The image shows an apartment building with multiple floors and balconies. The building is painted in a light beige color and has a white facade. The balconies are arranged in a grid-like pattern and are attached to the building with metal railings. There are palm trees in front of the building and a white fence with a lattice pattern. The sky is blue and there is a street lamp on the right side of the image. The image appears to be taken from a low angle, looking up at the building."
    generate_transition_prompts(
        test_text1, test_text2, num_steps=5, model_name='phi4')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"執行時間：{execution_time:.2f} 秒")
