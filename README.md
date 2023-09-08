# Stable Diffusion XL Model

[![Static Badge](https://img.shields.io/badge/Python_v3-gray)](https://github.com/ayush-thakur02/stable-diffusion-xl)
[![Stable Diffuison](https://img.shields.io/badge/File_Version-v1.0-blue)](https://github.com/ayush-thakur02/stable-diffusion-xl)
[![Stable Diffuison](https://img.shields.io/badge/Stable_Diffusion-XL_Base_1.0-blue)](https://github.com/ayush-thakur02/stable-diffusion-xl)

This Git Repo contains python notebook that is designed for running the Stable Diffusion XL Model on Google Colab with a T4 GPU. This open-source model is entirely free for you to use as much as you'd like, enabling you to generate an unlimited number of high-quality AI images surpassing those from the mid-journey. Don't forget to share this resource with your friends, and happy generating! 😃

---
#### Make Sure to follow me: 

[![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ayush-thakur02)
[![BioLink](https://img.shields.io/badge/bio.link-000000%7D?style=for-the-badge&logo=biolink&logoColor=white)](https://bio.link/ayush_thakur02)

# Setup

## Installing Packages
```python
%pip install --quiet --upgrade diffusers transformers accelerate invisible_watermark mediapy
use_refiner = False
```

## Downloading Model and Setting up Configuration
```python
import mediapy as media
import random
import sys
import torch

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    )

if use_refiner:
  refiner = DiffusionPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-refiner-1.0",
      text_encoder_2=pipe.text_encoder_2,
      vae=pipe.vae,
      torch_dtype=torch.float16,
      use_safetensors=True,
      variant="fp16",
  )

  refiner = refiner.to("cuda")

  pipe.enable_model_cpu_offload()
else:
  pipe = pipe.to("cuda")
```

## Run Prompt
```python
prompt = "a photo protrait of girl, wearing sunglasses, red blue hair, white colorful background, realistic, high resolution, HD quality"
seed = random.randint(0, sys.maxsize)

images = pipe(
    prompt = prompt,
    output_type = "latent" if use_refiner else "pil",
    generator = torch.Generator("cuda").manual_seed(seed),
    ).images

if use_refiner:
  images = refiner(
      prompt = prompt,
      image = images,
      ).images

print(f"Prompt:\t{prompt}\nSeed:\t{seed}")
media.show_images(images)
images[0].save("output.jpg")
```

# Output 1024x1024 (Default)
<img src="https://i.ibb.co/HNVjc1x/download.png" alt="download" border="0">
<img src="https://i.ibb.co/m5r4FgB/download.png" alt="download" border="0">
<img src="https://i.ibb.co/bNVcFbf/download.png" alt="download" border="0">
<img src="https://i.ibb.co/5vPXB0X/download.png" alt="download" border="0">

# Contribute
If you come across any errors or bugs, please don't hesitate to reach out via email, or feel free to create an issue ticket. Also, if you have any ideas or suggestions for enhancements, we'd love to hear from you. Your feedback and contributions are invaluable in improving this resource. Thank you for your collaboration in making this even better! 😊
