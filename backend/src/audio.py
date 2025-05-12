from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips
import os

VIDEO_PATH = "../../assets/video/video_with_sub.mp4"
AUDIO_DIR = "../../assets/audio"
OUTPUT_PATH = "../../assets/video/video_with_audio.mp4"

# 取得所有音檔，按順序排序
audio_files = sorted(
    [f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")],
    key=lambda x: int(x.split("_")[1].split(".")[0])  # 依照 subtitle_#.mp3 中的數字排序
)

# 合併音訊
audio_clips = [AudioFileClip(os.path.join(AUDIO_DIR, f)) for f in audio_files]
final_audio = concatenate_audioclips(audio_clips)

# 讀取影片並設置音訊
video_clip = VideoFileClip(VIDEO_PATH)
video_with_audio = video_clip.with_audio(final_audio)

# 輸出新影片
video_with_audio.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac")
