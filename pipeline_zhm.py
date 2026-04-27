from typing import Any, Callable, Dict, List, Optional, Union
import PIL
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.controlnet import StableDiffusionControlNetImg2ImgPipeline
from diffusers.utils import is_compiled_module
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel


@torch.no_grad()
def func_call_stable_diffusion_pipeline(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
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
    unet = None,
):
    # Dreamstyler: get DreamStyler kwargs
    # passing values through `cross_attention_kwargs` is the simplest way..
    num_stages = cross_attention_kwargs["num_stages"]
    use_sc_guidance = cross_attention_kwargs["use_sc_guidance"]
    sty_gamma = cross_attention_kwargs["sty_gamma"]
    con_gamma = cross_attention_kwargs["con_gamma"]
    neg_gamma = cross_attention_kwargs["neg_gamma"]

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
    # FIXME: Dreamstyler does not support single batch inference for now
    batch_size = num_images_per_prompt

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )
    # DreamStyler: encode timestep-varying prompts
    _prompt_embeds = []
    for prompt_t in prompt:
        _prompt_embeds_t = self._encode_prompt(
            prompt_t,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        _prompt_embeds.append(_prompt_embeds_t)
    prompt_embeds = _prompt_embeds

    # DreamStyler: encoder null style and context prompts as well
    if use_sc_guidance:
        _prompt_nc_embeds = []
        for prompt_nc_t in cross_attention_kwargs["prompt_null_context"]:
            _prompt_nc_embeds_t = self._encode_prompt(
                prompt_nc_t,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance=False,
                negative_prompt=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=text_encoder_lora_scale,
            )
            _prompt_nc_embeds.append(_prompt_nc_embeds_t)
        prompt_nc_embeds = _prompt_nc_embeds

        prompt_ns_t = cross_attention_kwargs["prompt_null_style"]
        prompt_ns_embeds = self._encode_prompt(
            prompt_ns_t,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
       )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds[0].dtype, # since in DreamStyler, `prompt_embeds` is list
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs.
    # TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Dreamstyler: handle both style and context guidance prompts in a batch
            if use_sc_guidance:
                latent_guided_model_input = torch.cat([latents] * 2)
                latent_guided_model_input = self.scheduler.scale_model_input(
                    latent_guided_model_input,
                    t,
                )

            # DreamStyler: prepare multi-stage TI
            max_timesteps = self.scheduler.config.num_train_timesteps
            index_stage = (t / max_timesteps * num_stages).long().view(1)
            result_list = []
            for i in range(0, len(prompt_embeds), 7):
                # 取出每7个张量，并堆叠成一个新的(7, 2, 77, 768)张量
                combined_tensor = torch.stack(prompt_embeds[i:i + 7], dim=0)
                result_list.append(combined_tensor)
            pm0 = result_list[index_stage][:, 0, :, :].unsqueeze(0)
            pm1 = result_list[index_stage][:, 1, :, :].unsqueeze(0)
            pm = torch.cat([pm0,pm1],dim=0)
            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=pm,
                return_dict=False,
            )[0]

            # DreamStyler: predict noise residual for guidance prompts
            if use_sc_guidance:
                prompt_guided_embeds = torch.cat(
                    [prompt_ns_embeds, prompt_nc_embeds[index_stage]],
                    dim=0,
                )

                noise_pred_guided = self.unet(
                    latent_guided_model_input,
                    t,
                    encoder_hidden_states=prompt_guided_embeds,
                    return_dict=False,
                ).sample
                noise_pred_ns, noise_pred_nc = noise_pred_guided.chunk(2)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                if use_sc_guidance:
                    noise_pred = noise_pred_uncond + \
                        0.5 * con_gamma * (noise_pred_ns - noise_pred_uncond) + \
                        0.5 * sty_gamma * (noise_pred_text - noise_pred_ns) + \
                        0.5 * sty_gamma * (noise_pred_nc - noise_pred_uncond) + \
                        0.5 * con_gamma * (noise_pred_text - noise_pred_nc) + \
                        neg_gamma * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_uncond + \
                        guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or \
                ((i + 1) > num_warmup_steps and \
                (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if output_type != "latent":
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        image, has_nsfw_concept = self.run_safety_checker(
            image,
            device,
            prompt_embeds[0].dtype,
        )
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(
        image,
        output_type=output_type,
        do_denormalize=do_denormalize,
    )

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(
        images=image,
        nsfw_content_detected=has_nsfw_concept,
    )

def func_call_controlnet_img2img_pipeline(
    self,
    prompt: Union[str, List[str]] = None, # 生成图像的文本提示    Union表示接受的参数的数据类型    Optional表示可选
    image: Union[
        torch.FloatTensor,
        PIL.Image.Image,
        np.ndarray,
        List[torch.FloatTensor],
        List[PIL.Image.Image],
        List[np.ndarray],
    ] = None, # 图生图的图
    control_image: Union[
        torch.FloatTensor,
        PIL.Image.Image,
        np.ndarray,
        List[torch.FloatTensor],
        List[PIL.Image.Image],
        List[np.ndarray],
    ] = None, # 控制条件的图片
    height: Optional[int] = None,
    width: Optional[int] = None, # 生成图像的高度和宽度
    strength: float = 0.8, # 控制图像和输入图像之间的混合强度
    num_inference_steps: int = 50, # 生成图像时的推理步骤数
    guidance_scale: float = 7.5, # 用于控制生成图像的引导强度
    negative_prompt: Optional[Union[str, List[str]]] = None, # 负提示
    num_images_per_prompt: Optional[int] = 1, # 每个提示生成的图像数量
    eta: float = 0.0, # 调节生成过程的参数
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, # 用于生成随机数的生成器
    latents: Optional[torch.FloatTensor] = None, # 预定义的潜在变量（可选）
    prompt_embeds: Optional[torch.FloatTensor] = None, # 文本提示的嵌入
    negative_prompt_embeds: Optional[torch.FloatTensor] = None, # 负面提示的嵌入
    output_type: Optional[str] = "pil", # 输出图像的类型，默认"pil"
    return_dict: bool = True, # 是否以字典形式返回结果
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None, # 回调函数
    callback_steps: int = 1, # 回调函数执行步骤数（可选）
    cross_attention_kwargs: Optional[Dict[str, Any]] = None, # 交叉注意力的额外参数
    controlnet_conditioning_scale: Union[float, List[float]] = 0.8, # ControlNet控制条件的指导系数
    guess_mode: bool = False, # 是否启用猜测模式
    control_guidance_start: Union[float, List[float]] = 0.0,
    control_guidance_end: Union[float, List[float]] = 1.0, # 控制引导的起始和结束比例
    unet = None,
):
    controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
    # 条件判断：controlnet = self.controlnet

    # align format for control guidance调整控制指导格式
    # control_guidance_start 和 control_guidance_end 参数用于控制引导的起始和结束比例。这些参数可能是单个值，也可能是值的列表。
    # 为了确保代码在后续操作中能够一致地处理这些参数，需要将它们规范化为列表，并且确保它们的长度一致。
    # 这里列举了所有参数不是列表的情况，并将他们转换为列表
    # 例：执行后：control_guidance_start = [0.2, 0.2, 0.2]，control_guidance_end = [0.5, 0.5, 0.5]
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
            control_guidance_end
        ]

    # 1. Check inputs. Raise error if not correct检查输入。如果不正确，则提示错误
    self.check_inputs(
        prompt,
        control_image,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
    )

    # 2. Define call parameters定义调用的参数
    batch_size = num_images_per_prompt # 设置batch_size

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0 # do_classifier_free_guidance = Ture

    if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)
    # 用于检查有多个controlnet的情况，若有多个controlnet，则将指导系数controlnet_conditioning_scale变为列表

    global_pool_conditions = (
        controlnet.config.global_pool_conditions
        if isinstance(controlnet, ControlNetModel)
        else controlnet.nets[0].config.global_pool_conditions
    ) # 根据 controlnet 对象的类型选择适当的 global_pool_conditions 配置
    guess_mode = guess_mode or global_pool_conditions # guess_mode = global_pool_conditions

    # 以上都是在设置参数，详细的部分还是要在以下代码中看
    # 3. Encode input prompt 编码输入文本
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    ) # text_encoder_lora_scale = None

    _prompt_embeds = []
    for prompt_t in prompt: # 这里的prompt可以是包含多个提示的列表
        _prompt_embeds_t = self._encode_prompt( # 可能是一个模型或管道对象中的 _encode_prompt 方法
            prompt_t,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        ) # 就是根据提示数量，每个提示生成的图片数量，对文本列表编码（不过我只有一个句子）
        _prompt_embeds.append(_prompt_embeds_t)
    prompt_embeds = _prompt_embeds # 得到包含提示编码的列表

    # 4. Prepare image准备图片
    image = self.image_processor.preprocess(image).to(dtype=torch.float32)
    # 预处理通常包括调整图像大小、归一化、裁剪、颜色转换等步骤，以便图像数据能够作为神经网络模型的输入

    # 5. Prepare controlnet_conditioning_image 准备ControlNet的控制图像
    # if语句判断是一个ControlNet还是多个，并决定执行的图片准备过程，我们当然是一个ControlNet
    if isinstance(controlnet, ControlNetModel): # 走这里
        control_image = self.prepare_control_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )  # control_image是准备好的控制图片
    elif isinstance(controlnet, MultiControlNetModel):
        control_images = []

        for control_image_ in control_image:
            control_image_ = self.prepare_control_image(
                image=control_image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

            control_images.append(control_image_)

        control_image = control_images
    else:
        assert False

    # 5. Prepare timesteps 准备时间步
    self.scheduler.set_timesteps(num_inference_steps, device=device) # 设置推理步数
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device) # 获取时间表和推理步数
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    # latent_timestep = [79，799，799，799，......,799]共batch_size * num_images_per_prompt个元素，799是采样25步时的第一个时间步

    # 6. Prepare latent variables  准备潜在变量，加入DDIM反演在这里
    latents = self.prepare_latents(
        image,
        latent_timestep,
        batch_size,
        num_images_per_prompt,
        prompt_embeds[0].dtype,
        device,
        generator,
    )

    # 7. Prepare extra step kwargs. 准备额外参数TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7.1 Create tensor stating which controlnets to keep # 创建controlnet_keep说明哪些时间步需要controlnet的指导
    controlnet_keep = []
    for i in range(len(timesteps)):
        keeps = [
            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

    num_stages = cross_attention_kwargs["num_stages"]

    # 8. Denoising loop 去噪循环
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order # 计算暖启动步骤数
    with self.progress_bar(total=num_inference_steps) as progress_bar: # 在该代码块中创建进度条实例progress_bar
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # 当前时间步的加噪图片潜在表示
            # 计算当前时间步属于哪一个阶段，index_T就是阶段号
            max_timesteps = self.scheduler.config.num_train_timesteps
            index_T = (t / max_timesteps * num_stages).long().view(1)


            # controlnet(s) inference
            if guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds[index_T].chunk(2)[1]
            else: # 走这里
                control_model_input = latent_model_input # 输入ControlNet的是图像的潜在编码
                #controlnet_prompt_embeds = prompt_embeds[index_T] # 提示来自于编码，我要在这改！！！
                result_list = []
                for ii in range(0, len(prompt_embeds), 7):
                    # 取出每7个张量，并堆叠成一个新的(7, 2, 77, 768)张量
                    combined_tensor = torch.stack(prompt_embeds[ii:ii + 7], dim=0)
                    result_list.append(combined_tensor)
                pm0 = result_list[index_T][:, 0, :, :].unsqueeze(0)
                pm1 = result_list[index_T][:, 1, :, :].unsqueeze(0)
                pm = torch.cat([pm0, pm1], dim=0)
                controlnet_prompt_embeds = pm

            if isinstance(controlnet_keep[i], list): # 在生成过程的每个时间步上计算控制网引导强度 cond_scale。
                cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]
            # 计算ControlNet下采样快和中间层的所有输出
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=control_image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )

            if guess_mode and do_classifier_free_guidance: # 不走这里
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

            # predict the noise residual调用Unet预测噪声图，加入了ControlNet的输出
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=pm,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
            # 往下就一样了
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided 去噪完成后，执行回调函数
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    # If we do sequential model offloading, let's offload unet and controlnet
    # manually for max memory savings 手动卸载模型来节省内存
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.unet.to("cpu")
        self.controlnet.to("cpu")
        torch.cuda.empty_cache()

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds[0].dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    image = image.detach()
    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)


    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
StableDiffusionPipeline.__call__ = func_call_stable_diffusion_pipeline
StableDiffusionControlNetImg2ImgPipeline.__call__ = func_call_controlnet_img2img_pipeline