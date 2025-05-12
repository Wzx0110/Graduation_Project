import time
import json
import re
import ollama
from typing import List, Dict, Any, Optional


def ask_ollama(model_name: str, question: str) -> Optional[str]:
    """將問題發送到指定的 Ollama 模型"""
    try:
        response = ollama.chat(model=model_name,
                               messages=[{'role': 'user', 'content': question}])
        answer = response['message']['content']
        print(answer)
        print("-------------------------\n")
        return answer
    except ollama.ResponseError as e:
        print(f"連接 Ollama 或找不到模型 '{model_name}' 時發生錯誤：{e.error}")
        return None
    except Exception as e:
        print(f"Ollama 請求期間發生錯誤：{e}")
        return None


def clean_json_response(answer: str) -> str:
    """提取 JSON 陣列內容，移除前後多餘的文字"""
    match = re.search(r"^\s*(\[.*?\])\s*$", answer, re.DOTALL)
    if match:
        json_str = match.group(1)
        if json_str.startswith('[') and json_str.endswith(']'):
            return json_str
    else:
        match_md = re.search(r"```json\s*(\[.*?\])\s*```", answer, re.DOTALL)
        if match_md:
            json_str = match_md.group(1)
            if json_str.startswith('[') and json_str.endswith(']'):
                return json_str
    answer = re.sub(r"^\s*```json\s*", "", answer).strip()
    answer = re.sub(r"\s*```\s*$", "", answer).strip()
    return answer

def parse_steps_json(json_string: str) -> Optional[List[Dict[str, str]]]:
    """解析預期為步驟字典列表的 JSON 字串，並只返回每個步驟的 'line' 欄位"""
    try:
        steps = json.loads(json_string)
        if isinstance(steps, list) and all(isinstance(s, dict) for s in steps):
            # 只返回每個步驟的 'line' 欄位
            steps_list = [{"line": s["line"]} for s in steps if "line" in s]
            return steps_list
        else:
            return None
    except json.JSONDecodeError as e:
        print(f"JSON 解碼失敗 {e}")
        return None


def generate_subtitle(text1: str, text2: str, model_name: str = 'phi4') -> Optional[List[Dict[str, str]]]:
    """
    Generate a 300–500 character Traditional Chinese story based on two image descriptions,
    then split it into subtitles of 15-20 Traditional Chinese characters and return in JSON format.

    Args:
        text1: The first scene description
        text2: The second scene description
        model_name: The Ollama model to use

    Returns:
        A list of dicts with "line" keys, each containing one subtitle line
    """

    prompt = f"""Based on the following two scene descriptions, write a complete Traditional Chinese story between 300 to 500 characters in length. 
    The story should have a coherent narrative progression, describing visual changes over time, character actions, and environmental atmosphere.

    Scene 1:
    {text1}

    Scene 2:
    {text2}

    After completing the story, divide it into subtitle segments.
    Each segment must strictly be between 15 and 20 Traditional Chinese characters per line.
    If any sentence exceeds the limit, break it into a new line.
    Do not allow any line to contain more than 20 characters, even by one character.
    Do not combine multiple sentences into one line if it would exceed the limit.
    Output only a pure JSON array in the format:

    [
    {{
    "line": "First subtitle line(may only 15-20 characters)"
    }},
    {{
    "line": "Second subtitle line(may only 15-20 characters)"
    }},
    ...
    ]

Do not include any titles, step numbers, markdown formatting, or additional notes—output only the clean JSON array as described.
"""

    answer = ask_ollama(model_name, prompt)

    if answer is None:
        return None

    cleaned_answer = clean_json_response(answer)
    steps_list = parse_steps_json(cleaned_answer)

    if steps_list is None:
        print("JSON parsing failed. Trying fallback.")
        steps_list = parse_steps_markdown(answer)

    formatted_steps = [[step["line"]] for step in steps_list if "line" in step]

    print(f"Successfully parsed {len(formatted_steps)} subtitle lines.")
    return formatted_steps


def parse_steps_markdown(answer: str) -> List[Dict[str, str]]:
    """解析 Markdown 格式的步驟"""
    steps = re.findall(
        r"^\s*#+\s*Step\s+(\d+)\s*[:\-]?\s*(.*?)\s*\n(.*?)(?=\n\s*#+\s*Step\s+\d+|\Z)",
        answer,
        re.MULTILINE | re.DOTALL
    )
    parsed_steps = []
    if steps:
        for num, title, desc in steps:
            parsed_steps.append({
                "line": desc.strip()
            })
    return parsed_steps


# 測試
if __name__ == "__main__":
    start_time = time.time()
    test_text1 = "The image shows a construction site with a large blue tarp covering a building under construction. The tarp is covering the entire wall of the building and is secured with metal scaffolding. There are several palm trees in front of the tarp, and a white fence surrounding the building. The sky is clear and blue, and there are other buildings visible in the background. The image appears to be taken from a low angle, looking up at the construction site."
    test_text2 = "The image shows an apartment building with multiple floors and balconies. The building is painted in a light beige color and has a white facade. The balconies are arranged in a grid-like pattern and are attached to the building with metal railings. There are palm trees in front of the building and a white fence with a lattice pattern. The sky is blue and there is a street lamp on the right side of the image. The image appears to be taken from a low angle, looking up at the building."
    subtitles = generate_subtitle(
        test_text1, test_text2, model_name='phi4')
    print(json.dumps(subtitles, ensure_ascii=False, indent=2))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"執行時間：{execution_time:.2f} 秒")
