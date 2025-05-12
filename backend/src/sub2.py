import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List
import os

OUTPUT_DIR = "../../assets/video"
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"  # 改成你系統的字型路徑


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

        draw.text((x + 1, y + 1), line, font=font, fill=(0, 0, 0))  # 陰影
        draw.text((x, y), line, font=font, fill=(255, 255, 255))    # 白字

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def create_video2_file(image_list: List[Image.Image], fps: int = 15, subtitles: List[List[str]] = None) -> bool:
    output_filename = "video_Test_subtitle.mp4"
    if not image_list:
        print("未提供用於創建影片的圖片")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    print(f"創建影片檔案: {output_path} ({len(image_list)} 個影格, {fps} FPS)")

    try:
        first_image = image_list[0]
        width, height = first_image.size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

        if not out_video.isOpened():
            print(f"無法創建影片檔案 {output_path}")
            return False

        total_frames = len(image_list)
        subtitle_count = len(subtitles) if subtitles else 0
        frames_per_sub = total_frames // subtitle_count if subtitle_count > 0 else total_frames

        for i, img in enumerate(image_list):
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if subtitles:
                sub_index = min(i // frames_per_sub, subtitle_count - 1)
                lines = subtitles[sub_index]
                frame_bgr = overlay_subtitle(frame_bgr, lines, width, height)

            out_video.write(frame_bgr)

        out_video.release()
        print(f"影片成功寫入 {output_path}")
        return True

    except Exception as e:
        print(f"錯誤：{e}")
        return False