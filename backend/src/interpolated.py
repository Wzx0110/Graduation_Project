from PIL import Image, ImageOps
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from io import BytesIO

def getImgInterpolated(oldImage, newImage):
    # 將 PIL 圖片轉換為 numpy 數組
    image1 = np.array(oldImage)
    image2 = np.array(newImage)

    # 調整圖像大小為 512x512
    image1 = cv2.resize(image1, (512, 512))
    image2 = cv2.resize(image2, (512, 512))

    # 將圖像轉換為 PyTorch 張量並歸一化
    def preprocess_image(image):
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # 調整維度並添加批次維度
        return image

    image1_tensor = preprocess_image(image1).to("cuda", torch.float16)
    image2_tensor = preprocess_image(image2).to("cuda", torch.float16)

    # 加載預訓練模型
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    with torch.no_grad():
        latent1 = pipe.vae.encode(image1_tensor).latent_dist.sample()
        latent2 = pipe.vae.encode(image2_tensor).latent_dist.sample()

    print("潛在空間形狀:", latent1.shape)  # 應該是 (1, 4, 64, 64)

    def interpolate(latent1, latent2, alpha):
        """
        在潛在空間中進行線性插值。
        :param latent1: 第一張圖像的潛在表示
        :param latent2: 第二張圖像的潛在表示
        :param alpha: 插值權重 (0 到 1 之間)
        :return: 插值後的潛在表示
        """
        return latent1 * (1 - alpha) + latent2 * alpha

    # 定義插值權重
    alpha = 0.5  # 0 表示完全使用 latent1，1 表示完全使用 latent2
    interpolated_latent = interpolate(latent1, latent2, alpha)

    def convert(interpolated_latent):
        with torch.no_grad():
            interpolated_image = pipe.vae.decode(interpolated_latent).sample
            interpolated_image = (interpolated_image / 2 + 0.5).clamp(0, 1)  # 歸一化到 [0, 1]
            interpolated_image = interpolated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            interpolated_image = (interpolated_image * 255).astype("uint8")
        return interpolated_image

    images = []
    alapha = 0
    for i in range(20):
        alapha += 0.05
        latent = interpolate(latent1, latent2, alapha)
        pics = convert(latent)
        img = Image.fromarray(pics)
        img = ImageOps.equalize(img)
        images.append(img)

    # 計算合併後圖片的總寬度和高度
    total_width = images[0].width * len(images)//2
    max_height = images[0].height * 2

    # 創建一個新的空白圖片，用於合併所有生成的圖片
    merged_image = Image.new('RGB', (total_width, max_height))

    # 將所有圖片合併到新圖片中
    x_offset = 0
    y_offset = 0
    for img in images:
        merged_image.paste(img, (x_offset, y_offset))
        x_offset += img.width
        if x_offset == total_width:
            x_offset = 0
            y_offset += img.height

    img_bytes = BytesIO()  # 建立一個 BytesIO 物件
    merged_image.save(img_bytes, format="JPEG")  # 把圖片存進 BytesIO
    img_bytes.seek(0)  # 將指標移到開頭
    return img_bytes  # 回傳 BytesIO 物件
    # 顯示合併後的圖片
    # merged_image.show()