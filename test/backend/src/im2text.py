import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
# image = "london.jpg"
# image = Image.open(image).convert("RGB")
def getimg2text(image):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=4096,
    num_beams=3,
    do_sample=False
)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    output = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))
    return output['<MORE_DETAILED_CAPTION>']
if __name__ == "__main__":
    print(getimg2text(image))
