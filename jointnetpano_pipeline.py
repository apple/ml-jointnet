#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import copy
import inspect
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from packaging import version
# import lpips
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler
from diffusers.utils import (
    deprecate,
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_accelerate_version,
    is_compiled_module,
    logging,
    randn_tensor,
    replace_example_docstring,
    BaseOutput,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from jointnet_pipeline import StableDiffusionJointNetPipelineOutput
from jointnet import JointNetModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def denoiseV2(jointnet, unet, latent_model_input, t, prompt_embeds, cross_attention_kwargs):
    num_channels_latents = latent_model_input.shape[1] // 2
    rgb_latent_input = latent_model_input[:, :num_channels_latents]
    joint_latent_input = latent_model_input[:, num_channels_latents:]

    rgb_down_block_res_samples, rgb_mid_block_res_sample, rgb_context = unet.forward_down_mid(
        rgb_latent_input, t,
        encoder_hidden_states=prompt_embeds,
        joint_input=joint_latent_input,
        cross_attention_kwargs=cross_attention_kwargs,
    )
    jnt_down_block_res_samples, jnt_mid_block_res_sample, jnt_context = jointnet.forward_down_mid(
        joint_latent_input, t,
        encoder_hidden_states=prompt_embeds,
        joint_input=rgb_latent_input,
        cross_attention_kwargs=cross_attention_kwargs,
    )
    rgb_noise_pred = unet.forward_up(
        down_block_additional_residuals=jnt_down_block_res_samples,
        mid_block_additional_residual=jnt_mid_block_res_sample,
        **rgb_context
    )
    jnt_noise_pred = jointnet.forward_up(
        down_block_additional_residuals=rgb_down_block_res_samples,
        mid_block_additional_residual=rgb_mid_block_res_sample,
        **jnt_context
    )

    noise_pred = torch.cat([rgb_noise_pred, jnt_noise_pred], dim=1)
    return noise_pred

denoise = denoiseV2


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class StableDiffusionJointNetPanoPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        jointnet: JointNetModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        # if isinstance(controlnet, (list, tuple)):
        #     controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            jointnet=jointnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)
        self.depth_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        # self.lpips_fn = lpips.LPIPS(net='vgg')

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae, controlnet, and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.jointnet]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offload all models to CPU to reduce memory usage with a low impact on performance. Moves one whole model at a
        time to the GPU when its `forward` method is called, and the model remains in GPU until the next model runs.
        Memory savings are lower than using `enable_sequential_cpu_offload`, but performance is much better due to the
        iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            # the safety checker can offload the vae again
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        cpu_offload_with_hook(self.jointnet, device)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents, padding=8):
        # Add padding to latents for circular inference
        # padding is the number of latents to add on each side
        # it would slightly increase the memory usage, but remove the boundary artifacts
        latents = 1 / self.vae.config.scaling_factor * latents
        if padding > 0:
            latents_left = latents[..., :padding]
            latents_right = latents[..., -padding:]
            latents = torch.cat((latents_right, latents, latents_left), axis=-1)
        image = self.vae.decode(latents, return_dict=False)[0]
        if padding > 0:
            padding_pix = self.vae_scale_factor * padding
            image = image[..., padding_pix:-padding_pix]
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        # if isinstance(self.controlnet, MultiControlNetModel):
        #     if isinstance(prompt, list):
        #         logger.warning(
        #             f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
        #             " prompts. The conditionings will be fixed across the prompts."
        #         )

    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # override DiffusionPipeline
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
        variant: Optional[str] = None,
    ):
        if isinstance(self.jointnet, JointNetModel):
            super().save_pretrained(save_directory, safe_serialization, variant)
        else:
            raise NotImplementedError("Currently, the `save_pretrained()` is not implemented for Multi-ControlNet.")

    def get_views(self, panorama_height, panorama_width, window_size=64, stride=8, w_offset=0, circular_padding=False, add_full_image=False):
        assert 0 <= w_offset < stride
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        # if panorama's height/width < window_size, num_blocks of height/width should return 1
        panorama_height //= self.vae_scale_factor
        panorama_width //= self.vae_scale_factor
        num_blocks_height = (panorama_height - window_size) // stride + 1 if panorama_height > window_size else 1
        if circular_padding:
            num_blocks_width = panorama_width // stride if panorama_width > window_size else 1
        else:
            num_blocks_width = (panorama_width - window_size) // stride + 1 if panorama_width > window_size else 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = (i // num_blocks_width) * stride
            h_end = h_start + window_size
            w_start = (i % num_blocks_width) * stride + w_offset
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))
        if add_full_image:
            views.append((0, panorama_height, 0, panorama_width))
        return views

    def calc_lpips_grad(self, latents_for_view, images_original_batch, lpips_anchor):
        loss = self.lpips_fn(images_original_batch, lpips_anchor).squeeze()
        # print(loss)
        loss = loss.sum()
        d_out = torch.ones_like(loss, requires_grad=False)
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=latents_for_view,
            grad_outputs=d_out,
            allow_unused=True,
        )[0]
        return grad

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 2048,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        view_batch_size: int = 1,
        view_stride: int = 8,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        circular_padding: bool = False,
        random_offset: bool = False,
        include_full_pano: Union[bool, float] = False,
        full_pano_strength: int = 1,
        boundary_type = {'type': 'constant'},
        lpips_weight: Union[float, dict] = 0.0,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # jointnet = self.jointnet._orig_mod if is_compiled_module(self.jointnet) else self.jointnet
        # if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        #     controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds_patch = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # prompt_embeds_full_image = self._encode_prompt(
        #     'Panorama, ' + prompt,
        #     device,
        #     num_images_per_prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     lora_scale=text_encoder_lora_scale,
        # )
        prompt_embeds_full_image = prompt_embeds_patch
        prompt_embeds = prompt_embeds_patch

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels * 2
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latent_height, latent_width = latents.shape[-2:]

        # 6. Define panorama grid and initialize views for synthesis.
        if isinstance(lpips_weight, float):
            lpips_weight = {
                'init': lpips_weight,
                'decay': 1,
                'skip_overlap': False,
            }
        lpips_skip_overlap = lpips_weight['skip_overlap']
        lpips_decay = lpips_weight['decay']
        lpips_weight = lpips_weight['init']

        # prepare batch grid
        include_full_pano_proportion = float(include_full_pano)  # boolean or actual value
        include_full_pano = include_full_pano_proportion > 0.0
        include_full_pano_step = int(num_inference_steps * include_full_pano_proportion)

        views = self.get_views(height, width, window_size=latent_height, stride=view_stride, circular_padding=circular_padding, add_full_image=include_full_pano)
        batchable_views = views[(1 if lpips_weight > 0.0 else 0) : (len(views)-1 if include_full_pano else len(views))]
        views_batch = [batchable_views[i : i + view_batch_size] for i in range(0, len(batchable_views), view_batch_size)]
        if lpips_weight > 0.0:
            views_batch = [views[:1]] + views_batch
        if include_full_pano:
            views_batch = views_batch + [views[-1:]]
        views_scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(views_batch)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                count.zero_()
                value.zero_()

                # random offset
                w_offset = np.random.randint(view_stride) if random_offset else 0

                # generate views
                # Here, we iterate through different spatial crops of the latents and denoise them. These
                # denoised (latent) crops are then averaged to produce the final latent
                # for the current timestep via MultiDiffusion. Please see Sec. 4.1 in the
                # MultiDiffusion paper for more details: https://arxiv.org/abs/2302.08113
                # Batch views denoise
                for j, batch_view in enumerate(views_batch):
                    vb_size = len(batch_view)

                    full_image_batch = (batch_view[-1][3] - batch_view[-1][2]) == latent_width  # NOTE hard code: full image must be the last one
                    assert (not full_image_batch) or vb_size == 1  # full_image_batch -> (vb_size == 1), full image cannot be batched with other patches
                    if full_image_batch and i > include_full_pano_step: continue

                    lpips_anchor_batch = lpips_weight > 0.0 and j == 0
                    assert (not lpips_anchor_batch) or vb_size == 1

                    # get the latents corresponding to the current view coordinates
                    latents_for_view = []
                    for h_start, h_end, w_start, w_end in batch_view:
                        w_start += w_offset
                        w_end += w_offset
                        if circular_padding and w_end > latents.shape[3]:
                            # Add circular horizontal padding
                            latent_view = torch.cat(
                                (
                                    latents[:, :, h_start:h_end, w_start:],
                                    latents[:, :, h_start:h_end, : w_end - latents.shape[3]],
                                ),
                                axis=-1,
                            )
                        else:
                            latent_view = latents[:, :, h_start:h_end, w_start:w_end]
                        latents_for_view.append(latent_view)
                    latents_for_view = torch.cat(latents_for_view)

                    # rematch block's scheduler status
                    self.scheduler.__dict__.update(views_scheduler_status[j])

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        latents_for_view.repeat_interleave(2, dim=0)
                        if do_classifier_free_guidance
                        else latents_for_view
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # repeat prompt_embeds for batch
                    prompt_embeds_input = torch.cat([prompt_embeds_full_image if full_image_batch else prompt_embeds] * vb_size)

                    noise_pred = denoise(self.jointnet, self.unet, latent_model_input, t, prompt_embeds_input, cross_attention_kwargs)

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred[::2], noise_pred[1::2]
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    # lpips
                    if lpips_weight > 0.0 and not full_image_batch:
                        if not lpips_anchor_batch:
                            grad_ctx = torch.enable_grad()
                            grad_ctx.__enter__()
                            latents_for_view.requires_grad_(True)
                        if lpips_skip_overlap and not lpips_anchor_batch:
                            non_overlapped = [i for i, (_,_,x_start,_) in enumerate(batch_view) if x_start % latent_height == 0]
                            latents_for_view_selected = latents_for_view[non_overlapped]
                            noise_pred_selected = noise_pred[non_overlapped]
                        elif not lpips_skip_overlap and not lpips_anchor_batch or lpips_anchor_batch:
                            latents_for_view_selected = latents_for_view
                            noise_pred_selected = noise_pred
                        latents_original_batch = self.scheduler.step(
                            noise_pred_selected, t, latents_for_view_selected, **extra_step_kwargs
                        ).pred_original_sample
                        images_original_batch = self.decode_latents(latents_original_batch[:,:num_channels_latents//2], padding=0)
                        if lpips_anchor_batch:
                            lpips_anchor = images_original_batch
                        else:
                            lpips_grad = self.calc_lpips_grad(latents_for_view, images_original_batch, lpips_anchor)
                            latents_for_view.requires_grad_(False)
                            grad_ctx.__exit__(None, None, None)
                            latents_for_view = latents_for_view - lpips_weight * lpips_grad
                            lpips_weight *= lpips_decay

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_denoised_batch = self.scheduler.step(
                        noise_pred, t, latents_for_view, **extra_step_kwargs
                    ).prev_sample

                    # save views scheduler status after sample
                    views_scheduler_status[j] = copy.deepcopy(self.scheduler.__dict__)

                    # extract value from batch
                    for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                        latents_denoised_batch.chunk(vb_size), batch_view
                    ):
                        w_start += w_offset
                        w_end += w_offset
                        if full_image_batch:
                            increment_weight = torch.full_like(latents_view_denoised, full_pano_strength)
                        else:
                            increment_pos_x = torch.arange(latent_height, device=latents_view_denoised.device, dtype=latents_view_denoised.dtype) + 0.5  # NOTE assume patch size == latent height
                            increment_pos_x = increment_pos_x.unsqueeze(0).repeat(latent_height, 1)  # HW
                            if boundary_type['type'] == 'constant':
                                increment_weight = torch.ones_like(latents_view_denoised)  # NCHW
                            elif boundary_type['type'] == 'linear':
                                boundary_size = boundary_type['size']
                                increment_weight = (latent_height/2 - (increment_pos_x - latent_height/2).abs()).clamp(0, boundary_size) / boundary_size
                                increment_weight = increment_weight[None, None].repeat(*latents_view_denoised.shape[:2], 1, 1)
                            elif boundary_type['type'] == 'gaussian':
                                raise NotImplementedError
                        latents_view_denoised = latents_view_denoised * increment_weight
                        if circular_padding and w_end > latents.shape[3]:
                            # Case for circular padding
                            value[:, :, h_start:h_end, w_start:] += latents_view_denoised[
                                :, :, h_start:h_end, : latents.shape[3] - w_start
                            ]
                            value[:, :, h_start:h_end, : w_end - latents.shape[3]] += latents_view_denoised[
                                :, :, h_start:h_end, latents.shape[3] - w_start :
                            ]
                            count[:, :, h_start:h_end, w_start:] += increment_weight[
                                :, :, h_start:h_end, : latents.shape[3] - w_start
                            ]
                            count[:, :, h_start:h_end, : w_end - latents.shape[3]] += increment_weight[
                                :, :, h_start:h_end, latents.shape[3] - w_start :
                            ]
                        else:
                            value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                            count[:, :, h_start:h_end, w_start:w_end] += increment_weight

                # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
                latents = torch.where(count > 0, value / count, value)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.jointnet.to("cpu")
            torch.cuda.empty_cache()

        image_latents = latents[:,:num_channels_latents//2]
        joint_latents = latents[:,num_channels_latents//2:]
        if not output_type == "latent":
            image = self.decode_latents(image_latents, padding=8 if circular_padding else 0)
            joint = self.decode_latents(joint_latents, padding=8 if circular_padding else 0)
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = image_latents
            joint = joint_latents
        has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        joint = self.image_processor.postprocess(joint, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, joint, has_nsfw_concept)

        return StableDiffusionJointNetPipelineOutput(images=image, joint_outputs=joint, nsfw_content_detected=has_nsfw_concept)

    @classmethod
    def from_pretrained_w_jointnet(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        jointnet_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs
    ):
        unet = JointNetModel.from_pretrained(jointnet_model_name_or_path, subfolder="unet", **kwargs)
        jointnet = JointNetModel.from_pretrained(jointnet_model_name_or_path, subfolder="jointnet", **kwargs)
        pipe = cls.from_pretrained(pretrained_model_name_or_path, unet=unet, jointnet=jointnet, **kwargs)
        return pipe


if __name__ == '__main__':
    from PIL import Image as PILImage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False, type=str, default=None)
    parser.add_argument('--base_model', required=False, type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--modalities', type=str, default='depth', choices=['depth', 'normal'])
    parser.add_argument('--prompt', required=False, type=str, default='A beach with palm trees')
    parser.add_argument('--negative_prompt', required=False, type=str, default=None)
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=7.5)
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=2048)
    parser.add_argument('--platform', type=str, choices=['mps', 'cuda'], default='cuda')
    parser.add_argument('--out_prefix', type=str, default='test_gen')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--normalize', action='store_true', default=False)
    args = parser.parse_args()

    scheduler = DDIMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    if args.platform == 'mps':
        pipe = StableDiffusionJointNetPanoPipeline.from_pretrained_w_jointnet(args.base_model, args.model, scheduler=scheduler)
        pipe = pipe.to('mps')
        pipe.enable_attention_slicing()
        # pipe.lpips_fn.to('mps')
    else:
        pipe = StableDiffusionJointNetPanoPipeline.from_pretrained_w_jointnet(args.base_model, args.model, scheduler=scheduler, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        # pipe.lpips_fn.to("cuda")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.platform).manual_seed(args.seed)

    pipe_output = pipe(args.prompt, negative_prompt=args.negative_prompt, height=args.H, width=args.W,
                       num_inference_steps=args.sample_steps, guidance_scale=args.cfg, generator=generator, output_type='np',
                       view_batch_size=8, view_stride=32, circular_padding=True, random_offset=True, include_full_pano=0.4, full_pano_strength=5,
                       boundary_type={'type':'linear', 'size':8}, lpips_weight={'init':0.0,'decay':0.95,'skip_overlap':False}
                    #    view_batch_size=8, view_stride=16, circular_padding=True, random_offset=False, include_full_pano=False, full_pano_strength=5,
                    #    boundary_type={'type':'constant', 'size':8}, lpips_weight={'init':20.0,'decay':0.95,'skip_overlap':False}
                    #    view_batch_size=8, view_stride=8, circular_padding=True, random_offset=False, include_full_pano=False, full_pano_strength=5,
                    #    boundary_type={'type':'constant', 'size':8}, lpips_weight={'init':0.0,'decay':0.95,'skip_overlap':False}
                       )
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
