import time
import json
import re
import ollama


def ask_phi4(question: str) -> str:
    response = ollama.chat(model='phi4',
                           messages=[{'role': 'user', 'content': question}]
                           )
    answer = response['message']['content']

    print("\n==== phi4 回應的原始內容 ====")
    print(answer)  # 確保 phi4 有回應內容
    print("================================\n")

    return answer


def clean_json_response(answer: str):
    """
    嘗試擷取 JSON 內容，去除前後多餘的文字。
    """
    match = re.search(r"(\[.*\])", answer, re.S)
    if match:
        return match.group(1)  # 只取 JSON 內容
    return answer  # 若找不到，返回原本的字串（可能是錯誤格式）


def parse_steps_json(answer: str):
    """
    解析 phi4 JSON 格式的輸出，並在 JSON 解析錯誤時提供備用方案。
    """
    clean_answer = clean_json_response(answer)  # 先清理 JSON
    try:
        steps = json.loads(clean_answer)
        if isinstance(steps, list):
            return steps
    except json.JSONDecodeError:
        print("JSON 解析失敗，嘗試手動轉換格式")

    # **如果 JSON 解析仍然失敗，改用 Markdown 解析**
    return parse_steps_markdown(answer)


def parse_steps_markdown(answer: str):
    """
    解析 Markdown 格式的 phi4 輸出。
    """
    steps = re.findall(
        r"### Step (\d+): (.*?)\n(.*?)(?=\n### Step \d+:|\Z)", answer, re.S)
    return [{"step": int(num), "title": title.strip(), "description": desc.strip()} for num, title, desc in steps]


if __name__ == "__main__":
    text1 = "The image shows a construction site with a large blue tarp covering a building under construction. The tarp is covering the entire wall of the building and is secured with metal scaffolding. There are several palm trees in front of the tarp, and a white fence surrounding the building. The sky is clear and blue, and there are other buildings visible in the background. The image appears to be taken from a low angle, looking up at the construction site."
    text2 = "The image shows an apartment building with multiple floors and balconies. The building is painted in a light beige color and has a white facade. The balconies are arranged in a grid-like pattern and are attached to the building with metal railings. There are palm trees in front of the building and a white fence with a lattice pattern. The sky is blue and there is a street lamp on the right side of the image. The image appears to be taken from a low angle, looking up at the building."

    question = f"""
Given two image descriptions:

First image: {text1}  
Second image: {text2}  

Describe a **gradual, step-by-step transformation of the entire scene** from the first image to the second image.  

- Each step should describe **the whole image**, not just specific objects or parts.  
- The transition should be **smooth and visually realistic**, ensuring that each step logically follows the previous one.  
- Focus only on **visible changes** in the image, avoiding construction techniques or non-visual details.  
- Provide **10 sequential steps**, each showing the full scene at that moment.  



IMPORTANT: Please output the response as a **valid JSON list** with the following structure:
[
    {{"step": 1, "title": "Step Title", "description": "Detailed explanation of the step."}},
    {{"step": 2, "title": "Step Title", "description": "Detailed explanation of the step."}},
    ...
]

DO NOT include extra text outside of this JSON format.
"""

    start_time = time.time()
    answer = ask_phi4(question)
    end_time = time.time()

    execution_time = end_time - start_time
    steps_list = parse_steps_json(answer)

    print(f"執行時間: {execution_time:.2f} 秒")

    if steps_list:
        print("分解後的步驟:")
        for step in steps_list:
            print(
                f"Step {step['step']}: {step['title']}\n  - {step['description']}\n")
    else:
        print("無法解析出步驟，請檢查 `phi4` 的回應格式！")
