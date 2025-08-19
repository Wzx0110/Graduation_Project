# backend/core/generation.py

import torch
from PIL import Image
# 導入 Pipeline 的類型提示，以便函數簽名使用
from diffusers import StableDiffusionXLImg2ImgPipeline
from typing import Dict, Any, List, Generator

# --- 輔助函數 (保持不變) ---


def image_to_latent(image: Image.Image, pipe: StableDiffusionXLImg2ImgPipeline) -> torch.Tensor:
    """將 PIL Image 編碼成 VAE 的潛在向量"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (1024, 1024):
        image = image.resize((1024, 1024), Image.LANCZOS)
    with torch.no_grad():
        image_tensor = pipe.image_processor.preprocess(
            image).to(pipe.device, pipe.dtype)
        latent_dist = pipe.vae.encode(image_tensor).latent_dist
        latent = latent_dist.sample() * pipe.vae.config.scaling_factor
    return latent


def decode_latents_to_image(latents: torch.Tensor, pipe: StableDiffusionXLImg2ImgPipeline) -> Image.Image:
    """將潛在向量解碼回 PIL Image"""
    with torch.no_grad():
        latents = 1 / pipe.vae.config.scaling_factor * latents
        image_tensor = pipe.vae.decode(latents).sample
    image = pipe.image_processor.postprocess(
        image_tensor, output_type='pil', do_denormalize=[True])[0]
    return image


def interpolate_latents_lerp(lat_a: torch.Tensor, lat_b: torch.Tensor, steps: int) -> List[torch.Tensor]:
    """對潛在向量進行線性插值 (LERP)"""
    interpolated_latents = []
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0.5
        interp_latent = torch.lerp(lat_a, lat_b, t)
        interpolated_latents.append(interp_latent)
    return interpolated_latents

# --- 分步生成流程 (現在接收 pipeline 作為參數) ---


def generate_latent_previews(
    pipe: StableDiffusionXLImg2ImgPipeline,
    img1: Image.Image,
    img2: Image.Image,
    num_frames: int
) -> Generator[Dict[str, Any], None, None]:
    """
    第一階段：插值 latent 並解碼成預覽圖。
    """
    if pipe is None:
        raise ValueError("Pipeline 物件不能為 None。")

    print("正在將輸入圖片編碼到潛在空間...")
    latent_a = image_to_latent(img1, pipe)
    latent_b = image_to_latent(img2, pipe)

    print(f"正在進行 {num_frames} 幀的潛在空間插值...")
    interpolated_latents = interpolate_latents_lerp(
        latent_a, latent_b, num_frames)

    for i, current_latent in enumerate(interpolated_latents):
        latent_preview_img = decode_latents_to_image(current_latent, pipe)
        yield {
            "type": "latent_preview",
            "frame_index": i,
            "total_frames": num_frames,
            "image": latent_preview_img
        }

    # 產出最終結果，供 main.py 傳遞給下一個階段
    yield {"final_result": interpolated_latents}


def generate_frames_from_images(
    pipe: StableDiffusionXLImg2ImgPipeline,
    init_images: List[Image.Image],
    storyboard: Dict[str, Any],
    strength: float,
    seed: int = 42,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25
) -> Generator[Dict[str, Any], None, None]:
    """
    第二階段：接收插值好的 PIL Image，逐幀進行 Img2Img 生成。
    """
    if pipe is None:
        raise ValueError("Pipeline 物件不能為 None。")

    num_frames = len(init_images)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    generated_frames = []
    for i, init_image in enumerate(init_images):
        frame_data = storyboard['keyframes'][i]

        prompt_obj = frame_data.get('prompt', {})
        prompt_1 = f"{prompt_obj.get('change', '')}, {prompt_obj.get('quality', '')}"
        prompt_2 = prompt_obj.get('context', '')
        default_negative_prompt = "(low quality, worst quality, blurry:1.2), text, watermark, signature, ugly"
        negative_prompt = prompt_obj.get(
            'negative_prompt', default_negative_prompt)

        # 調用 Img2Img Pipeline
        image = pipe(
            prompt=prompt_1,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        generated_frames.append(image)
        yield {
            "type": "generated_frame",
            "frame_index": i,
            "total_frames": num_frames,
            "image": image
        }

    yield {"final_result": generated_frames}
