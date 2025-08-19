import torch
from PIL import Image
import numpy as np
from typing import List, Optional, Tuple
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import time
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticModelLoader:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.processor: Optional[OneFormerProcessor] = None
        self.model: Optional[OneFormerForUniversalSegmentation] = None
        self._load_model()

    def _load_model(self):
        self.processor = OneFormerProcessor.from_pretrained(
            self.model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def cleanup(self):
        del self.model
        self.model = None
        del self.processor
        self.processor = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("語意分割模型已清理")


def get_semantic_map(
    image_pil: Optional[Image.Image],
    processor: Optional[OneFormerProcessor],
    model: Optional[OneFormerForUniversalSegmentation],
    device: str,
) -> Optional[np.ndarray]:
    """
    使用 OneFormer 生成圖像的語義分割圖
    """
    task_inputs = ["semantic"]
    try:
        inputs = processor(
            images=image_pil, task_inputs=task_inputs, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 語義分割後處理函數
        semantic_map_tensor = processor.post_process_semantic_segmentation(
            # target_sizes(height, width) PIL size(width, height)
            outputs, target_sizes=[image_pil.size[::-1]]
        )[0]  # 只處理單張

        semantic_map_np = semantic_map_tensor.cpu().numpy().astype(np.uint8)

        return semantic_map_np

    except Exception as e:
        print(f"直接生成語義圖時發生錯誤: {e}")
        return None


def visualize_map(
    segmentation_map_np: Optional[np.ndarray],
    palette: List[List[int]],
    target_size: Optional[Tuple[int, int]] = None
) -> Optional[Image.Image]:
    """
    將numpy語義分割圖轉換為可視化圖像
    """
    colored_array = np.zeros(
        segmentation_map_np.shape + (3,), dtype=np.uint8)  # (H, W, 3)
    unique_labels = np.unique(segmentation_map_np)

    for label_id in unique_labels:
        if 0 <= label_id < len(palette):  # 確保標籤在調色板範圍內
            colored_array[segmentation_map_np == label_id] = palette[label_id]

    pil_image = Image.fromarray(colored_array)

    if target_size and pil_image.size != target_size:
        # 每個像素值代表一個離散類別，使用最近鄰插值保留類別邊界，避免在類別之間產生模糊的過渡顏色或不存在的類別 ID
        pil_image = pil_image.resize(target_size, Image.NEAREST)

    return pil_image


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
    """
    從路徑載入圖像
    """
    try:
        image = Image.open(image_path).convert("RGB")
        if target_size and image.size != target_size:
            image = image.resize(target_size, Image.LANCZOS)
        return image
    except Exception as e:
        print(f"載入圖片 {image_path} 失敗: {e}")
        return None


# 測試
if __name__ == "__main__":
    start_time = time.time()
    from configs import MainConfig
    cfg = MainConfig()
    semantic_loader = None
    try:
        semantic_loader = SemanticModelLoader(cfg.SEMANTIC_MODEL_NAME, DEVICE)
        img_pil = load_image(cfg.IMAGE_A_PATH, cfg.COMMON_TARGET_SIZE)
        map_np = get_semantic_map(
            img_pil,
            semantic_loader.processor,
            semantic_loader.model,
            DEVICE,
        )

        vis_map_pil = visualize_map(
            map_np, cfg.CITYSCAPES_PALETTE, cfg.COMMON_TARGET_SIZE)

        vis_map_pil.save("../../../assets/semantic_map/semantic_map.png")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"執行時間：{execution_time:.2f} 秒")
    except Exception as e:
        print(f"語義分割測試過程發生錯誤: {e}")
    finally:
        if semantic_loader:
            semantic_loader.cleanup()
