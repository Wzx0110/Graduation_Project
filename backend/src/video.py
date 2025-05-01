import cv2
import numpy as np
from PIL import Image
from typing import List
import time
import os  # 用於處理文件路徑和目錄


OUTPUT_DIR = "../../assets/video"  # 輸出資料夾


def create_video_file(image_list: List[Image.Image], fps: int = 15) -> bool:
    """
    從 PIL 圖片列表創建 MP4 影片檔案

    Args:
        image_list: PIL Image 物件列表
        fps: 輸出影片的幀率 (每秒影格數)

    Returns:
        如果影片成功創建並儲存，回傳 True
    """
    output_filename = "video_output.mp4"
    if not image_list:
        print("未提供用於創建影片的圖片")
        return False

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    print(f"創建影片檔案: {output_path} ({len(image_list)} 個影格, {fps} FPS)")

    out_video = None  # 初始化 VideoWriter 變數

    try:
        # 從第一張圖片獲取影格大小
        first_image = image_list[0]
        width, height = first_image.size
        print(f"影片尺寸：{width}x{height}")

        # 定義編解碼器並創建 VideoWriter 物件，直接寫入目標檔案
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out_video = cv2.VideoWriter(
            output_path, fourcc, float(fps), (width, height))

        if not out_video.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(
                output_path, fourcc, float(fps), (width, height))
            if not out_video.isOpened():
                print(f"編解碼器 'mp4v' 失敗。無法創建影片檔案 {output_path}。")
                return False

        # 逐一寫入影格
        for i, img in enumerate(image_list):
            # 將 PIL Image 轉換為 NumPy 陣列 (BGR)
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 將影格寫入影片檔案
            out_video.write(frame_bgr)

        # 釋放 VideoWriter 資源
        out_video.release()
        out_video = None  # 標記為已釋放
        print(f"VideoWriter 已釋放。影片已寫入 {output_path}")
        return True  # 成功創建並儲存

    except Exception as e:
        print(f"創建影片檔案 {output_path} 發生錯誤：{e}")
        return False  # 創建失敗
