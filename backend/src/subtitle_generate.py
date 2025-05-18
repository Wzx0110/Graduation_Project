import time
import json
import re
import ollama
from typing import List, Dict, Any, Optional

data = [
    {'step': 1, 'title': 'Day 1', 'description': 'On the first day of transformation, construction workers are actively moving equipment around on the street outside the red-brick building. A delivery truck parks briefly near the mural, unloading supplies as a few pedestrians pass by, admiring the vibrant artwork. The sky remains clear and blue, reflecting an upbeat atmosphere despite the bustling activity.'},
    {'step': 2, 'title': 'Day 3', 'description': 'By Day 3, additional scaffolding is erected around the mural on the red-brick building, casting dynamic shadows as workers apply fresh layers of paint. A street vendor sets up a small cart nearby, selling refreshments to passersby who stop to observe the evolving artwork and construction. The atmosphere buzzes with anticipation, underpinned by bright sunlight.'},
    {'step': 3, 'title': 'Day 7', 'description': "On Day 7, workers have begun covering part of the blue sky mural with a new design featuring a large bird mid-flight. A woman in a business suit walks her dog along the palm-lined street, stopping to snap photos of the changing mural. Clouds start to gather overhead, adding an air of mystery as the city's transformation continues."},
    {'step': 4, 'title': 'Day 10', 'description': 'By Day 10, a new blue bird with a yellow beak is prominently emerging in the mural, catching the eye of children playing nearby. A street musician begins to play softly on his guitar, adding an acoustic layer to the scene as workers take a break and chat under the increasingly cloudy sky.'},
    {'step': 5, 'title': 'Week 2', 'description': "In Week 2, construction of the beige apartment building has started nearby, with scaffolding reaching its mid-section. A young couple walks hand-in-hand along the sidewalk, pausing to watch both murals as they converse about the area's changes. The city breathes a tranquil energy despite the visible transformation."},
    {'step': 6, 'title': 'Week 3', 'description': 'By Week 3, construction progress on the beige building is evident with multiple floors now framed by scaffolding. The delivery truck returns to unload more materials while a cyclist whizzes past, glancing at both murals as he rides along. A gentle breeze stirs through palm leaves as clouds hover densely over the cityscape.'},
    {'step': 7, 'title': 'Week 4', 'description': "On Week 4, scaffolding envelops much of the red building's facade while the mural nears completion with intricate details on the blue bird. A street artist begins sketching nearby, capturing the evolving cityscape and colorful murals in his drawing pad. The atmosphere turns contemplative under a gray sky."},
    {'step': 8, 'title': 'Week 6', 'description': 'By Week 6, scaffolding is largely removed from the red building as workers apply finishing touches to both mural and construction materials stored neatly away. An elderly man feeding pigeons on the street remarks to a neighbor about how vibrant the area has become. Overhead, rain clouds begin to break, letting sunlight peek through.'},
    {'step': 9, 'title': 'Week 8', 'description': 'In Week 8, final touches are added to the mural and both buildings stand proud with completed facades. The street vendor returns, this time attracting a small crowd of curious onlookers eager to see the finished murals. Laughter rings out as children chase each other past, under a calming blue sky that promises clear skies ahead.'},
    {'step': 10, 'title': 'Week 10', 'description': 'By Week 10, the transformation is complete with both buildings fully integrated into their urban environment. The street musician plays as pedestrians stroll leisurely by, admiring the new mural and newly constructed apartment building. A quiet peace envelops the scene under a gentle sunlit sky, marking the culmination of this vibrant cityscape.'}
]

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


def generate_subtitle(descriptions: List, model_name: str = 'phi4') -> Optional[List[Dict[str, str]]]:
    """
    Generate a 300 character Traditional Chinese story based on the transtion descriptions,
    then split the story into subtitle lines, each containing at most 15 Traditional Chinese characters,
    and return them in JSON format.

    Args:
        descriptions: The transtion description (string)
        model_name: The name of the Ollama model to use

    Returns:
        A list of dictionaries with "line" keys, each representing one subtitle line
    """

    prompt = f"""You are given a list of scene transition descriptions. Use these to write a short, coherent story in Traditional Chinese, between **200 and 300 Chinese characters** in total.

    Story Requirements:
    - Describe changes in the scenes and people involved
    - Reflect the passage of time and transitions
    - Maintain a consistent tone and narrative

    Subtitle Formatting Instructions:
    - After writing the story, split it into subtitle lines
    - Each line should be in Traditional Chinese
    - Each line must contain **no more than 10 Chinese characters**
    - Avoid punctuation at the end of each line
    - Do not break words or phrases unnaturally

    Output JSON Format:
    Return **only** a clean JSON array of the subtitle lines. No explanation, no extra text, no markdown formatting.

    Example output:
    [
        {{
            "line": "第一周工地空無一物"
        }},
        {{
            "line": "第二周鋼骨架出現在地面"
        }},
        {{
            "line": "第三周工人忙碌地搭建"
        }}
    ]

    Now here are the scene descriptions:
    {descriptions}

    """

    answer = ask_ollama(model_name, prompt)

    if answer is None:
        return None
    print(answer)
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
    # test_text1 = "The image shows a construction site with a large blue tarp covering a building under construction. The tarp is covering the entire wall of the building and is secured with metal scaffolding. There are several palm trees in front of the tarp, and a white fence surrounding the building. The sky is clear and blue, and there are other buildings visible in the background. The image appears to be taken from a low angle, looking up at the construction site."
    # test_text2 = "The image shows an apartment building with multiple floors and balconies. The building is painted in a light beige color and has a white facade. The balconies are arranged in a grid-like pattern and are attached to the building with metal railings. There are palm trees in front of the building and a white fence with a lattice pattern. The sky is blue and there is a street lamp on the right side of the image. The image appears to be taken from a low angle, looking up at the building."
    descriptions = [entry['description'] for entry in data]
    subtitles = generate_subtitle(descriptions)
    print(json.dumps(subtitles, ensure_ascii=False, indent=2))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"執行時間：{execution_time:.2f} 秒")
