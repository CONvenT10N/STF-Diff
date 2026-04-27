#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import argparse
import logging
from os.path import join as ospj

import diffusers
import accelerate
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision

from PIL import Image
from tqdm import trange
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from custom_unet import CusUNet2DConditionModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

if is_wandb_available(): # 使用 wandb 提供的功能来跟踪和记录实验数据。
    import wandb

check_min_version("0.20.0")
logger = get_logger(__name__) # 获取一个 logger 实例来记录日志信息


class DreamStylerDataset(torch.utils.data.Dataset): # 自定义数据集
    template = "A painting in the style of {}"

    def __init__(
        self,
        image_path, # 图片路径
        tokenizer, # 分词器
        size=512, # 分辨率
        repeats=100, #
        prob_flip=0.5,
        placeholder_tokens="*", # placeholder_tokens=["<sks09>-T0", "<sks09>-T1", "<sks09>-T2", "<sks09>-T3", "<sks09>-T4", "<sks09>-T5"]
        center_crop=False,
        is_train=True,
        num_stages=1,
        context_prompt=None,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_tokens = placeholder_tokens
        self.center_crop = center_crop
        self.prob_flip = prob_flip # 翻转图像的概率 0.5
        self.repeats = repeats if is_train else 1 # self.repeats = 100
        self.num_stages = num_stages

        if not isinstance(self.placeholder_tokens, list):
            self.placeholder_tokens = [self.placeholder_token] # 如果 self.placeholder_tokens不是列表类型，那么它会将 self.placeholder_token包装成一个单元素列表赋值给self.placeholder_tokens
            # self.placeholder_tokens = ["<sks09>-T0", "<sks09>-T1", "<sks09>-T2", "<sks09>-T3", "<sks09>-T4", "<sks09>-T5"]
        self.flip = torchvision.transforms.RandomHorizontalFlip(p=self.prob_flip) # 创建一个水平翻转图像的变换对象，存储在 self.flip 属性中。这个变换会以一定的概率对图像进行水平翻转。

        self.image_path = image_path
        self.prompt = self.template if context_prompt is None else context_prompt # 若文本内容提示为空，则默认为"A painting in the style of {}"

    def __getitem__(self, index):
        image = Image.open(self.image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)
        # 开一个图像文件，将其转换为 RGB 模式，并将其转换为一个 8 位无符号整型的 NumPy 数组。
        prompt = self.prompt

        tokens = []
        for t in range(self.num_stages * 7):
            placeholder_string = self.placeholder_tokens[t] # 依次遍历["<sks09>-T0", "<sks09>-T1", "<sks09>-T2", "<sks09>-T3", "<sks09>-T4", "<sks09>-T5"]
            # 例：placeholder_string = ”<sks09>-T0“
            prompt_t = prompt.format(placeholder_string) # 用placeholder_string替换prompt的“{}”
            # 例：prompt_t = "A painting in the style of <sks09>-T0"

            tokens.append( # 对prompt_t分词得到tonken_id，再将所有循环的prompt_t都append到tokens列表里面
                self.tokenizer(
                    prompt_t,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
            )
        # 这个循环结束后，tokens是装了n个阶段的6层的内容提示的文本（已经将风格伪词放在一起）token_id列表
        num_groups = self.num_stages  # 目标子列表的数量
        group_size = len(tokens) // num_groups  # 每个子列表的大小
        new_tokens = [tokens[i:i + group_size] for i in range(0, len(tokens), group_size)]

        if self.center_crop: # center_crop = False
            h, w = image.shape[0], image.shape[1]
            min_hw = min(h, w)
            image = image[
                (h - min_hw) // 2 : (h + min_hw) // 2,
                (w - min_hw) // 2 : (w + min_hw) // 2,
            ]

        image = Image.fromarray(image) # 使用 PIL 的 Image.fromarray 方法将 NumPy 数组 image 转换为 PIL 图像对象。
        image = image.resize((self.size, self.size), resample=Image.LANCZOS)
        # 使用 resize 方法将图像调整为 self.size x self.size 的大小，使用 Image.LANCZOS 作为重采样滤波器进行高质量的图像缩放。
        image = self.flip(image) # 水平翻转图片
        image = np.array(image).astype(np.uint8) # 将 PIL 图像对象转换回 NumPy 数组，并将其数据类型转换为 8 位无符号整型（uint8）。
        image = (image / 127.5 - 1.0).astype(np.float32) # 对图像进行归一化处理：将像素值从 [0, 255] 变换到 [-1.0, 1.0]。具体操作是将像素值除以 127.5 并减去 1.0，然后将数据类型转换为 32 位浮点型（float32）。
        image = torch.from_numpy(image).permute(2, 0, 1) # 将归一化后的 NumPy 数组转换为 PyTorch 张量，并调整维度顺序。permute(2, 0, 1) 将图像的维度从 (height, width, channels) 变换为 (channels, height, width)，

        return {
            "input_ids": new_tokens,
            "pixel_values": image,
        } # 返回文本tokens和图片

    def __len__(self):
        return self.repeats


def train(opt):
    accelerator = init_accelerator_and_logger(logger, opt) # 实例化加速器，实际是实例化了一些组件
    (
        train_dataset, # 处理好的token和图片
        train_dataloader,
        placeholder_tokens, # 与分阶段数量有关的伪词列表placeholder_tokens=["<sks09>-T0", "<sks09>-T1", "<sks09>-T2", "<sks09>-T3", "<sks09>-T4", "<sks09>-T5"]
        placeholder_token_ids, # 伪词对应的分词id 初始化自初始词 placeholder_token_ids=[5001, 5002, 5003, 5004, 5005, 5006]
        tokenizer, #tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path,subfolder="tokenizer",W)
        text_encoder, # text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path,subfolder="text_encoder",revision=opt.revision,)
        noise_scheduler, # 去噪调度表
        optimizer, # 优化器
        lr_scheduler, # 学习率
        vae, # vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path,subfolder="vae",revision=opt.revision,)
        unet, # unet = UNet2DConditionModel.from_pretrained(opt.pretrained_model_name_or_path,subfolder="unet",revision=opt.revision,)
        weight_dtype, # 优化的embedding的数据类型
    ) = init_model_and_dataset(accelerator, logger, opt)
# 看到这里
    # do we need this?
    if opt.resume_from_checkpoint: # opt.resume_from_checkpoint = None  不走这里
        raise NotImplementedError

    # keep original embeddings as reference
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight.data.clone()
    )
    # 这里是从模型里面提取text_encoder的嵌入层的权重，这是一个包含词嵌入权重的张量。将其复制到orig_embeds_params中

    logger.info("***** Running training *****") # 显示和记录一些信息
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total optimization steps = {opt.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {opt.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")

    text_encoder.train() # 训练模式
    for step in trange(opt.max_train_steps, disable=not accelerator.is_local_main_process): # 用tqdm的方法创建训练循环，如果不是主进程，就不显示进度条
        try:
            batch = next(iters)
        except (UnboundLocalError, StopIteration, TypeError): # 因为没有iters，走这里
            iters = iter(train_dataloader) # 创建一个新的迭代器 iters，从 train_dataloader 中获取
            batch = next(iters) # 从新的迭代器 iters 中获取下一个批次的训练数据。

        with accelerator.accumulate(text_encoder): # 使用加速器 (accelerator) 累积梯度更新 text_encoder
            # convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)) # 对图片进行编码到潜在域
            latents = latents.latent_dist.sample().detach() * vae.config.scaling_factor
            # 将采样的潜在表示乘以 vae.config.scaling_factor 进行缩放，并阻止缩放操作的梯度传播，这个步骤的作用是将潜在表示调整到适当的范围或尺度。

            # sample noise that we'll add to the latents
            noise = torch.randn_like(latents) # 生成一个与 latents 形状相同的随机噪声张量。
            bsz = latents.shape[0] # 获取批次大小，即 latents 张量的第一个维度的大小。为1
            # sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )# 在[0, noise_scheduler.config.num_train_timesteps=1000) 范围内生成随机整数，并将其放在与批次大小相同的张量中，放置在与 latents 相同的设备上。
            # 相当于给每个batch创建一个随机的时间步数，例如若batchsize = 8，则timesteps = [1，34，23，67，445，756，234，54]
            timesteps = timesteps.long() # 将 timesteps 转换为 long 类型

            # Dreamstyler: get index in stage (T) axis
            max_timesteps = noise_scheduler.config.num_train_timesteps # 1000
            index_stage = (timesteps / max_timesteps * opt.num_stages).long()
            # 这里是生成不同批次对应不同的随机阶段
            #例如[3, 2, 3, 4, 3, 4, 3, 4]，因为是long类型，所以没有小数

            # add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) # 图片的潜在噪声表示，他没用反演
            #理解为latents（原图）+noise = noisy_latents（起始噪声），在某一步，noise是timesteps这步加的噪声

            # get the text embedding for conditioning
            # Dreamstyler: batch["input_ids"] is [T x bsz x 77]-dim shape
            # and if bsz > 1, timesteps have multiple arbitary t values
            # so that input_ids variable should be proprocessed
            # to be matched to appropriate timesteps
            # batch["input_ids"]是装了n个阶段的内容提示的文本（已经将风格伪词放在一起）token_id列表
            # 举例batch["input_ids"] = [[11,22,33],[11,22,33]]
            #input_ids = torch.empty_like(batch["input_ids"][0]) # 创建一个与 batch["input_ids"][0] （一个阶段的提示的token_id）形状相同的空张量
            input_ids = [torch.empty(6,77) for _ in range(bsz)]
            # input_ids[8,77]
            for n in range(bsz): # 这里要改
                input_ids[n] = torch.stack([x[0, :] for x in batch["input_ids"][index_stage[n]]])
                # input_id是一个列表，元素数量是batchsize个，每个元素是（6，77）的tensor，每个（1，77）是一个阶段里的一个层接受的张量
            input_ids_tensor = torch.cat(input_ids, dim=0)
            encoder_hidden_states = text_encoder(input_ids_tensor)[0].to(weight_dtype)
            #将input_ids送入文本编码器，得到文本编码
            encoder_hidden_states = encoder_hidden_states.reshape(bsz, 7, 77, 768)
            # encoder_hidden_states是(bsz, 6, 77, 768)的张量，6表示unet的6个注意力层
            # predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            #model_pred(batch_size,4,64,64)
            # ‘’‘’‘’注意一下这里的采样方式，他只预测了其中随机几个去噪步骤的输出‘’‘’‘’

            # get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon": # 走这里
                target = noise # 注意这里的设计，noise是timesteps这步加的噪声
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") # 通过将 reduction 参数设置为 "mean"，计算得到的损失是所有样本误差的平均值，确保损失值是一个标量，便于后续的反向传播和优化步骤。
            #print(loss)
            # noise是timesteps这步加的噪声，要和unet在timesteps这步预测的噪声一样
            accelerator.backward(loss) # 使用 accelerator 执行反向传播，计算损失 loss 相对于模型参数的梯度。
            # 相当于loss.backward
            optimizer.step() # 更新模型参数
            lr_scheduler.step() # 更新学习率调度器 lr_scheduler
            optimizer.zero_grad() # 梯度缓冲区清零

            # let's make sure we don't update any embedding weights 保证除了新词之外的词嵌入没有被更新
            # besides the newly added token将原来复制的不应该被更新的权重复制回去
            index_no_updates = ~torch.isin(
                torch.arange(len(tokenizer)),
                torch.tensor(placeholder_token_ids),
            )
            with torch.no_grad():
                emb1 = accelerator.unwrap_model(text_encoder).get_input_embeddings()
                emb2 = orig_embeds_params[index_no_updates]
                emb1.weight[index_no_updates] = emb2

        if accelerator.sync_gradients: # 检查 accelerator 是否需要同步梯度。这通常在分布式训练中用于确定是否需要在所有设备之间同步梯度。
            if accelerator.is_main_process and (step + 1) % opt.save_steps == 0:
                # 确认当前进程是主进程。在分布式训练中，只有主进程执行某些操作（例如保存模型），以避免重复操作。
                # (step + 1) % opt.save_steps == 0: 检查当前步骤是否是保存模型的步骤
                save( # 调用 save 函数保存模型，传递多个参数，保存以下变量
                    accelerator,
                    text_encoder,
                    placeholder_tokens,
                    placeholder_token_ids,
                    step + 1,
                    opt,
                )

    #到此结束训练
    accelerator.wait_for_everyone() # 确保所有进程同步，在继续执行后续代码之前等待所有进程都达到这一点
    save( # 训练结束后保存模型
        accelerator,
        text_encoder,
        placeholder_tokens,
        placeholder_token_ids,
        "final",
        opt,
    )
    accelerator.end_training() # 结束训练


def init_accelerator_and_logger(logger, opt): # 初始化加速器和记录器
    logging_dir = ospj(opt.output_dir, opt.logging_dir) # 设置日志文件保存位置
    accelerator_project_config = ProjectConfiguration(
        project_dir=opt.output_dir,
        logging_dir=logging_dir,
    ) # 创建一个ProjectConfiguration对象，包含项目目录和日志目录的设置
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        mixed_precision=opt.mixed_precision,
        log_with=opt.report_to,
        project_config=accelerator_project_config,
    ) # 使用配置选项中的梯度累积步数、混合精度、日志记录工具等初始化加速器对象。

    if opt.report_to == "wandb": # 如果配置选项中指定了使用 wandb 进行日志记录，则检查 wandb 是否可用。如果不可用，则抛出一个错误。
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it"
                " for logging during training."
            )

    # make one log on every process with the configuration for debugging.在每个进程上都生成一个日志，其中包含用于调试的配置。
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    ) # 使用 logging.basicConfig 设置日志格式和级别。
    logger.info(accelerator.state, main_process_only=False) # 使用 logger.info 记录加速器状态。
    if accelerator.is_local_main_process: # 根据是否为本地主要进程（accelerator.is_local_main_process），设置 transformers 和 diffusers 库的日志级别。
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # handle the repository creation
    if accelerator.is_main_process: # 如果当前进程是主要进程（accelerator.is_main_process），并且输出目录不为空，则创建输出目录。
        if opt.output_dir is not None:
            os.makedirs(opt.output_dir, exist_ok=True)

    os.makedirs(ospj(opt.output_dir, "embedding"), exist_ok=True) # 无论是否为主要进程，都会创建嵌入目录。
    return accelerator # 返回初始化好的加速器对象。


def init_model_and_dataset(accelerator, logger, opt, without_dataset=False):
    if opt.seed is not None: # 若没有设置种子，则随即粽子
        set_seed(opt.seed)

    if opt.tokenizer_name: # 初始化分词器
        tokenizer = CLIPTokenizer.from_pretrained(opt.tokenizer_name)
    elif opt.pretrained_model_name_or_path: # opt.pretrained_model_name_or_path="E:/models_download/sdv15"
        tokenizer = CLIPTokenizer.from_pretrained(
            opt.pretrained_model_name_or_path,
            subfolder="tokenizer",
        ) # tokenizer是sdv15的CLIP的分词器

    noise_scheduler = DDPMScheduler.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="scheduler",
    ) # 噪声时间表noise_scheduler
    text_encoder = CLIPTextModel.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=opt.revision,
    ) # 文本编码器也来自于sdv15的CLIP的文本编码器
    vae = AutoencoderKL.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="vae",
        revision=opt.revision,
    ) # 图片编码器vea
    unet = CusUNet2DConditionModel.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="unet",
        revision=opt.revision,
    )# 扩散U-Net

    # DreamStyler: TODO: support multi-vector TI
    if opt.num_vectors > 1: # 不管，默认是1
        raise NotImplementedError

    # DreamStyler: add new textual embeddings
    # opt.placeholder_token = "<sks09>"  opt.num_stages = 分阶段数量
    placeholder_tokens = [
        f"{opt.placeholder_token}-T{t}" for t in range(opt.num_stages)
    ] # 根据输入的伪词生成一个伪词列表，有几个阶段就生成几个伪词
    placeholder_tokens = [
        f"{opt.placeholder_token}-T{t}-L{l}" for t in range(opt.num_stages) for l in range(1, 8)
    ]

    # 例子： placeholder_tokens =["<sks09>-T0", "<sks09>-T1", "<sks09>-T2", "<sks09>-T3", "<sks09>-T4", "<sks09>-T5"]
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)# 将上面的一系列伪词添加到分词器的词表中，返回成功添加的词的数量
    if num_added_tokens == 0: # 如果没成功添加就报错
        raise ValueError(
            f"The tokenizer already contains the token {opt.placeholder_token}."
            " Please pass a different `placeholder_token` that is not already in the tokenizer."
        )

    # convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(opt.initializer_token, add_special_tokens=False) # --initializer_token = painting
    # 对opt.initializer_token=”paint“分词，返回分词结果
    if len(token_ids) > 1: # 限制初始词语只能是一个词语
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0] # 就是opt.initializer_token的token ID,因为只有一个词语
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens) # 将前面的伪词列表分词为token列表
    #例如，将["<sks09>-T0", "<sks09>-T1", "<sks09>-T2", "<sks09>-T3", "<sks09>-T4", "<sks09>-T5"]转换为
    # placeholder_token_ids=[5001, 5002, 5003, 5004, 5005, 5006]，即为之前添加的新词表的词对应的id

    # resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer)) # 根据 tokenizer 词汇表的当前大小，调整 text_encoder 的嵌入层大小，以确保模型能够正确处理新增的令牌。因为之前扩充了词表

    # initialize the newly added placeholder token
    # with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    # 这一步使 token_embeds 包含了模型中所有令牌的嵌入向量。相当于token_embeds是一个列表，保存了词表中的词的embedding
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
            # 这里将伪词的词嵌入初始化为初始词的词嵌入，对应文中从初始词的词嵌入出发反演词嵌入


    # freeze vae and unet and text encoder (except for the token embeddings)，冻结图片编码器、unet、文本编码器
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if opt.gradient_checkpointing: # None 不走这里
        # keep unet in train mode if we are using gradient checkpointing to save memory.
        # the dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if opt.enable_xformers_memory_efficient_attention: # False 不管
        if is_xformers_available():
            import version
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs."
                    " If you observe problems during training, please update xFormers"
                    " to at least 0.0.17."
                    " See https://huggingface.co/docs/diffusers/main/en/optimization/xformers"
                    " for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if opt.allow_tf32: # False 不管
        torch.backends.cuda.matmul.allow_tf32 = True

    if opt.scale_lr: # False 不管
        opt.learning_rate = (
            opt.learning_rate
            * opt.gradient_accumulation_steps
            * opt.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(), # 优化目标是整个词表的词嵌入
        lr=opt.learning_rate, # 学习率
        betas=(opt.adam_beta1, opt.adam_beta2), # 超参数，用于一阶和二阶距估计
        weight_decay=opt.adam_weight_decay, # 权重衰减（L2 正则化）系数
        eps=opt.adam_epsilon, #  优化器的ε项，用于数值稳定性
    ) # 实例化优化器

    lr_scheduler = get_scheduler(
        opt.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=opt.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=opt.max_train_steps * accelerator.num_processes,
        num_cycles=opt.lr_num_cycles,
    ) # get_scheduler 函数创建了一个学习率调度器 (lr_scheduler)，用于动态调整优化器的学习率

    if without_dataset: # False 不走这里
        train_dataset, train_dataloader = None, None
        text_encoder, optimizer, lr_scheduler = accelerator.prepare(
            text_encoder,
            optimizer,
            lr_scheduler,
        )
    else: # 走这里
        train_dataset = DreamStylerDataset(
            image_path=opt.train_image_path, # 训练图像的路径
            tokenizer=tokenizer, # 分词器
            size=opt.resolution, # 分辨率
            placeholder_tokens=placeholder_tokens, # 伪词列表，几个阶段有几个元素
            repeats=opt.max_train_steps, # 训练次数
            center_crop=opt.center_crop, # False
            is_train=True,
            context_prompt=opt.context_prompt, # 内容文本
            num_stages=opt.num_stages, # 分阶段数量
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.train_batch_size,
            shuffle=True,
            num_workers=opt.dataloader_num_workers, # 用于数据加载的工作线程数
        )
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) # accelerator.prepare(...)：使用 accelerator 准备模型、优化器、数据加载器和学习率调度器，以便在分布式或加速环境中训练。

    text_encoder, optimizer, lr_scheduler = accelerator.prepare(
        text_encoder,
        optimizer,
        lr_scheduler,
    ) # 同上

    # for mixed precision training we cast all non-trainable weigths
    # (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference,
    # keeping weights in full precision is not required.
    weight_dtype = torch.float32 # 首先将权重的数据类型设为默认的 torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # 然后，根据 accelerator.mixed_precision 的值，决定是否将数据类型切换为 torch.float16 或 torch.bfloat16
    # 这种数据类型的调整通常用于模型训练中的混合精度训练，以提高计算效率和减少内存占用。

    # move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype) # 将unet和vea放到GPU上

    # we need to initialize the trackers we use, and also store our configuration.
    # the trackers initializes automatically on the main process.
    if accelerator.is_main_process: # 初始化一个跟踪器（tracker），用于记录训练过程中的指标和日志信息。
        accelerator.init_trackers("dreamstyler", config=vars(opt))
        # "dreamstyler" 是跟踪器的名称，可以根据需要随意命名。
        # config=vars(opt) 将 opt 对象转换为一个字典，并作为配置传递给跟踪器。vars(opt) 返回一个包含 opt 对象所有属性及其值的字典。

    return ( # 返回
        train_dataset, # 训练dataset
        train_dataloader, # 训练dataloader
        placeholder_tokens, # 与分阶段数量有关的伪词列表placeholder_tokens=["<sks09>-T0", "<sks09>-T1", "<sks09>-T2", "<sks09>-T3", "<sks09>-T4", "<sks09>-T5"]
        placeholder_token_ids, # 伪词对应的分词id placeholder_token_ids=[5001, 5002, 5003, 5004, 5005, 5006]
        tokenizer, #tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path,subfolder="tokenizer",W)
        text_encoder, # text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path,subfolder="text_encoder",revision=opt.revision,)
        noise_scheduler, # 去噪调度表
        optimizer, # 优化器
        lr_scheduler,# 学习率
        vae, # vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path,subfolder="vae",revision=opt.revision,)
        unet, # unet = UNet2DConditionModel.from_pretrained(opt.pretrained_model_name_or_path,subfolder="unet",revision=opt.revision,)
        weight_dtype, # 优化的embedding的数据类型
    )


def save( # 保存模型的函数
    accelerator,
    text_encoder,
    placeholder_tokens,
    placeholder_token_ids,
    prefix,
    opt,
):
    prefix = f"{prefix:04d}" if isinstance(prefix, int) else prefix

    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings()
    embeds_dict = {}
    for token, token_id in zip(placeholder_tokens, placeholder_token_ids):
        embeds_dict[token] = learned_embeds.weight[token_id].detach().cpu()
    torch.save(embeds_dict, ospj(opt.output_dir, "embedding", f"{prefix}.bin"))


def get_options():
    parser = argparse.ArgumentParser()

    # DreamStyler arguments
    parser.add_argument(
        "--context_prompt",
        type=str,
        default=None,
        help="Additional context prompt to enhance training performance.",
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=6,
        help="The number of the stages (denoted as T) used in multi-stage TI.",
    )

    # original textual inversion arguments
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_image_path",
        type=str,
        default=None,
        required=True,
        help="A path of training image.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default="painting",
        help="A token to use as initializer word.",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="style",
        help="Choose between 'object' and 'style'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dreamstyler",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the"
            " train/validation dataset will be resized to this resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Whether or not to use gradient checkpointing"
            " to save memory at the expense of slower backward pass."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=(
            "Scale the learning rate by the number of GPUs,"
            " gradient accumulation steps, and batch size."
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            "The scheduler type to use. Choose between"
            " ['linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
            " 'constant', 'constant_with_warmup']"
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading."
            " 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
            " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs."
            " Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            "The integration to report the results and logs to."
            ' Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`.'
            ' Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=5,
        help=(
            "Number of images that should be generated"
            " during validation with `validation_prompt`.",
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates."
            " These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint."
            " Use a path saved by `--checkpointing_steps`,"
            ' or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true", # bool开关的意思
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help=(
            "If specified save the checkpoint not in `safetensors` format,"
            " but in original PyTorch format instead.",
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank not in (-1, args.local_rank):
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    opt = get_options()
    train(opt)
