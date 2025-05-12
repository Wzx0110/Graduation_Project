import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List
import os
import edge_tts
import asyncio
import shutil
from mutagen.mp3 import MP3
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips

OUTPUT_DIR = "../../assets/video"
AUDIO_DIR = "../../assets/audio"
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"

# subtitles = [
#     ['城市中心，一幢現代建築映入眼帘。'], 
#     ['紅磚牆面上，大型抽象壁畫引人注目。'], 
#     ['天空藍淨，白雲飄浮，氣氛明媚溫暖。'], 
#     ['建築旁邊，一棟藍色建築正在興建。'], 
#     ['鋼索和工具凌亂堆放，宛如未來的青葱希望。'], 
#     ['街道旁種植着棕櫚樹，小巷映出鮮活生命。'],
#     ['陽光照耀下，停車場上的車輛如沙漏般靜止。'], 
#     ['日子一天天地過去，城市面貌開始改變。'], 
#     ['那棟正在興建的藍色大樓竟在一夜之間完成。'], 
#     ['紅磚牆上壁畫也換上了新的顏色和主題。'], 
#     ['現在是一隻藍鳥，嘴巴黃澄澄地，在天空飛翔。'],
#     ['城市街道依舊宜人，但氣氛似乎變得更加寧靜。'], 
#     ['一個平靜的下午，風吹著棕櫚樹叶子低语。'], 
#     ['云朵漂浮在天空中，顏色變得深邃黯淡。'], 
#     ['幾個行人穿過大街小巷，慢 步而行。'], 
#     ['紅色建築旁的壁畫映射在雲層上，如水彩般柔和融入天際。'],
#     ['這片城市總是變化而不失其本質，寧靜並充滿生命力。']
# ]

async def generate_audio(subtitles: List[List[str]]) -> None:
    # 清空資料夾
    if os.path.exists(AUDIO_DIR):
        shutil.rmtree(AUDIO_DIR)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    for idx, text in enumerate(subtitles):
        filename = f"{AUDIO_DIR}/subtitle_{idx+1}.mp3"
        communicate = edge_tts.Communicate(text[0], voice="zh-TW-YunJheNeural")
        await communicate.save(filename)  # 需要加 await
        print(f"已儲存: {filename}")

def get_audio_durations(audio_dir: str, subtitle_count: int) -> List[float]:
    durations = []
    for i in range(subtitle_count):
        path = os.path.join(audio_dir, f"subtitle_{i+1}.mp3")
        if os.path.exists(path):
            audio = MP3(path)
            durations.append(audio.info.length)
        else:
            durations.append(0.0)
    return durations

def assign_frames_per_subtitle(durations: List[float], fps: int) -> List[int]:
    return [int(d * fps) for d in durations]

def overlay_subtitle(frame, lines, width, height):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, 24)

    line_spacing = 30
    margin_bottom = 30

    for idx, line in enumerate(lines):
        bbox = font.getbbox(line)
        text_width = bbox[2] - bbox[0]
        x = int((width - text_width) / 2)
        y = height - margin_bottom - (len(lines) - 1 - idx) * line_spacing

        draw.text((x + 1, y + 1), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

async def create_video_with_subtitles(image_list: List[Image.Image], subtitles: List[List[str]], fps: int = 15) -> bool:
    video_subtitle_filename = "video_with_sub.mp4"
    video_audio_filename = "video_with_audio.mp4"
    if not image_list:
        print("未提供圖片")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_subtitle_path = os.path.join(OUTPUT_DIR, video_subtitle_filename)
    video_audio_path = os.path.join(OUTPUT_DIR, video_audio_filename)
    print(f"創建影片: {video_subtitle_path}")

    try:
        await generate_audio(subtitles)
        durations = get_audio_durations(AUDIO_DIR, len(subtitles))
        frames_per_subtitle = assign_frames_per_subtitle(durations, fps)
        subtitle_frame_map = []
        for idx, count in enumerate(frames_per_subtitle):
            subtitle_frame_map += [subtitles[idx]] * count

        subtitle_frame_map = subtitle_frame_map[:len(image_list)]

        first_image = image_list[0]
        width, height = first_image.size
        out_video = cv2.VideoWriter(video_subtitle_path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (width, height))

        for i, img in enumerate(image_list):
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            lines = subtitle_frame_map[i] if i < len(subtitle_frame_map) else []
            frame_bgr = overlay_subtitle(frame_bgr, lines, width, height)
            out_video.write(frame_bgr)

        out_video.release()
        print(f"影片成功寫入 {video_subtitle_path}")
        
        # 取得所有音檔，按順序排序
        audio_files = sorted(
            [f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")],
            key=lambda x: int(x.split("_")[1].split(".")[0])  # 依照 subtitle_#.mp3 中的數字排序
        )

        # 合併音訊
        audio_clips = [AudioFileClip(os.path.join(AUDIO_DIR, f)) for f in audio_files]
        final_audio = concatenate_audioclips(audio_clips)

        # 讀取影片並設置音訊
        video_clip = VideoFileClip(video_subtitle_path)
        video_with_audio = video_clip.with_audio(final_audio)

        # 輸出新影片
        video_with_audio.write_videofile(video_audio_path, codec="libx264", audio_codec="aac")
        return video_audio_path

    except Exception as e:
        print(f"錯誤: {e}")
        return False
