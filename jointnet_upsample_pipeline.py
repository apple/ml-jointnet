#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import numpy as np
import torch
from PIL import Image as PILImage
import cv2

from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler

from jointnetpano_inpaint_pipeline import StableDiffusionJointNetPanoInpaintPipeline
from io_utils import read_depth_map, read_image, to_tensor, get_dummy_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False, type=str, default=None)
    parser.add_argument('--base_model', required=False, type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--modalities', type=str, default='depth', choices=['depth', 'normal'])
    parser.add_argument('--prompt', required=False, type=str, default='A beach with palm trees')
    parser.add_argument('--image', required=False, type=str, default='test_pano_image.png')
    parser.add_argument('--joint_input', required=False, type=str, default='test_pano_depth.png')
    parser.add_argument('--preset', type=str, default='both',
                        choices=['both', 'image', 'joint'])
    parser.add_argument('--negative_prompt', required=False, type=str, default=None)
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=7.5)
    parser.add_argument('--denoising_strength', type=float, default=0.4)
    parser.add_argument('--H', type=int, default=1024)
    parser.add_argument('--W', type=int, default=4096)
    parser.add_argument('--platform', type=str, choices=['mps', 'cuda'], default='cuda')
    parser.add_argument('--out_prefix', type=str, default='test_gen')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--normalize', action='store_true', default=False)
    args = parser.parse_args()

    image_mask_fill = int(args.preset in ['both', 'image'])
    joint_mask_fill = int(args.preset in ['both', 'joint'])

    image_input = read_image(args.image)
    if args.preset in ['both', 'image']:
        image_input = cv2.resize(image_input, (args.W, args.H), interpolation=cv2.INTER_LANCZOS4)
    image_mask = get_dummy_mask(image_input, image_mask_fill)
    joint_input = read_depth_map(args.joint_input)
    if args.preset in ['both', 'joint']:
        joint_input = cv2.resize(joint_input, (args.W, args.H), interpolation=cv2.INTER_NEAREST)
    joint_mask = get_dummy_mask(joint_input, joint_mask_fill)

    image_input = to_tensor(image_input, rescale=True)
    image_mask = to_tensor(image_mask)
    joint_input = to_tensor(joint_input, rescale=True)
    joint_mask = to_tensor(joint_mask)

    scheduler = DDIMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    if args.platform == 'mps':
        pipe = StableDiffusionJointNetPanoInpaintPipeline.from_pretrained_w_jointnet(args.base_model, args.model, scheduler=scheduler)
        pipe = pipe.to('mps')
        pipe.enable_attention_slicing()
        # pipe.lpips_fn.to('mps')
    else:
        pipe = StableDiffusionJointNetPanoInpaintPipeline.from_pretrained_w_jointnet(args.base_model, args.model, scheduler=scheduler, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        # pipe.lpips_fn.to("cuda")
    # pipe.enable_vae_tiling()

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.platform).manual_seed(args.seed)

    pipe_output = pipe(args.prompt, negative_prompt=args.negative_prompt, height=args.H, width=args.W,
                       image=image_input, mask_image=image_mask, joint_input=joint_input, mask_joint=joint_mask,
                       num_inference_steps=args.sample_steps, guidance_scale=args.cfg, strength=args.denoising_strength, generator=generator, output_type='np',
                       view_window_size=64, view_batch_size=8, view_stride=32, circular_padding=False, random_offset=False, include_full_pano=False,
                       boundary_type={'type':'linear', 'size':8, 'x': True, 'y': True})
    image = pipe_output.images[0]
    image = PILImage.fromarray((image * 255).round().astype("uint8"))
    joint_out = pipe_output.joint_outputs[0]
    match args.modalities:
        case 'depth':
            joint_out = joint_out[:,:,0]
            if args.normalize:
                joint_min, joint_max = np.quantile(joint_out, [0.01,0.99])
                joint_out = ((joint_out - joint_min) / (joint_max - joint_min)).clip(0,1)
            joint_out = PILImage.fromarray((joint_out * 65535).round().astype("uint16"))
        case 'normal':
            joint_out = PILImage.fromarray((joint_out * 255).round().astype("uint8"))

    out_prefix = args.out_prefix.replace(' ', '_')
    image.save(f'{out_prefix}_image.png')
    joint_out.save(f'{out_prefix}_{args.modalities}.png')
