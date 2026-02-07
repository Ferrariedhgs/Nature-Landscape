Synthetic nature landscape dataset made using the [SSD-1B](https://huggingface.co/segmind/SSD-1B) text-to-image model

## Prompt used for generation
Pozitive prompt: realistic nature landscape

Negative prompt: ugly, blurry, poor quality, text, watermark

## Hardware
Ran on an RTX 2060 Super (8GB VRAM) for ~1.5h

## Dataset Description
- **Type**: Synthetic image dataset
- **Images**: PNG
- **Generation**: Text-to-image diffusion model
- **Resolution**: 512x512

## Generation Details
1024 images, the seed for each one is the id. Manually removed the junk images (some were generated as paintings on walls or multiple pictures in one) and generated more to fit the target count

## Model Attribution
Images were generated using:

- **Model**: [SSD-1B](https://huggingface.co/segmind/SSD-1B)
- **License**: Apache-2.0
- **Source**: Hugging Face

## License
Released under Apache 2.0 License, same as the model.
