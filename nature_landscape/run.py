from diffusers import StableDiffusionXLPipeline
import torch
import gc
import os
import time

out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"output")

p_prompt = "realistic nature landscape"
n_prompt = "ugly, blurry, poor quality, text, watermark"

os.makedirs(out_dir, exist_ok=True)

meta = {
    "file": "image_i.png",
    "prompt": p_prompt,
    "negative_prompt": n_prompt,
    "seed": "i"
}

pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
#pipe.enable_attention_slicing()

batch_size=1024

start=time.time()
for i in range(batch_size):
    gen = torch.Generator("cuda").manual_seed(i)
    image = pipe(prompt=p_prompt, negative_prompt=n_prompt, generator=gen, height=512, width=512).images[0]

    out_path = os.path.join(out_dir, f'image_{i}.png')
    image.save(out_path)
    print(i)

    #free some vram for low vram gpus
    del image
    gc.collect()
    torch.cuda.empty_cache()

print(f'Saved {batch_size} images, time: {time.time()-start} s')
