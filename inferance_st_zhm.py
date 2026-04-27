import os
from os.path import join as ospj
import click
import torch
import imageio
import numpy as np
from PIL import Image, ImageOps
from diffusers import ControlNetModel, UniPCMultistepScheduler,UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import pipeline as pl
from controlnet_aux.processor import Processor
import controlnet_zhm
import pipeline_zhm
from custom_unet import CusUNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPModel
from torchvision import models, transforms
import torch.nn as nn
import lpips
from skimage.metrics import structural_similarity


def load_model(sd_path, controlnet_path, embedding_path, placeholder_token="<sks1>", num_stages=6):
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")  # 从扩散模型中实例化分词器
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")  # 从扩散模型中实例化文本编码器
    controlnet = controlnet_zhm.ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)  # 实例化预训练的ControlNet模型
    unet = CusUNet2DConditionModel.from_pretrained(
        sd_path,
        subfolder="unet",
        revision=None,
    )
    unet.to('cuda')
    unet.to(torch.float16)
    #placeholder_token = [f"{placeholder_token}-T{t}" for t in range(num_stages)]  # 创建了一个包含多个占位符令牌的列表，像训练一样
    # placeholder_token = ["<sks1>-T0", "<sks1>-T1", "<sks1>-T2", "<sks1>-T3", "<sks1>-T4", "<sks1>-T5"]
    placeholder_token = [
        f"{placeholder_token}-T{t}-L{l}" for t in range(num_stages) for l in range(1, 8)
    ]

    num_added_tokens = tokenizer.add_tokens(placeholder_token)  # 将生成的占位符令牌列表添加到tokenizer的词表中
    if num_added_tokens == 0:  # 报错信息
        raise ValueError("The tokens are already in the tokenizer")
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)  # 获取伪词列表placeholder_token在分词器中的id
    text_encoder.resize_token_embeddings(len(tokenizer))  # 调整text_encoder模型的嵌入矩阵的大小，以匹配当前tokenizer中的token数量

    learned_embeds = torch.load(embedding_path)  # 从final.bin（储存了训练好的伪词的文本编码器输出）文件读取文本编码器预训练输出
    token_embeds = text_encoder.get_input_embeddings().weight.data  # 获取 text_encoder 模型的输入嵌入矩阵
    for token, token_id in zip(placeholder_token, placeholder_token_id):
        token_embeds[token_id] = learned_embeds[token]
    # 将训练好的伪词的词嵌入替换到文本编码器的词嵌入token_embeds中去
    # 现在token_embeds就保存了原本就有的预训练词嵌入，还有之前训练阶段训练得到的伪词的词嵌入

    pipeline = pipeline_zhm.StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        sd_path,
        controlnet=controlnet,  # 从文件中实例化的controlnet模型
        text_encoder=text_encoder.to(torch.float16),  # 文本编码器
        tokenizer=tokenizer,  # 分词器
        torch_dtype=torch.float16,  # 数据类型
        safety_checker=None,
    )  # 会调用custom_pipelines.py文件中的func_call_controlnet_img2img_pipeline函数
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    processor_midas = Processor("depth_midas")

    return pipeline, processor_midas,unet


# 这里接受命令行用户输入的参数，将参数赋值给相应的变量
@click.command()
@click.option("--sd_path")
@click.option("--controlnet_path", default="E:/models_download/controlnet")  # "lllyasviel/control_v11f1p_sd15_depth"
@click.option("--embedding_path")  # D:\PythonWorkSpace\dreamstyler-main\dreamstyler\outputs\sks03\embedding\final.bin
@click.option("--content_image_path")
@click.option("--saveroot", default="./outputs")
@click.option("--prompt", default="A painting of a city skyline, in the style of {}")
@click.option("--placeholder_token", default="<sks1>")
@click.option("--num_stages", default=6)
@click.option("--num_samples", default=5)
@click.option("--resolution", default=512)
@click.option("--seed")
def style_transfer(
        sd_path=None,  # 扩散模型的路径
        controlnet_path="E:/models_download/controlnet",  # "lllyasviel/control_v11f1p_sd15_depth"ControlNet的路径
        embedding_path=None,  # 训练的结果保存的路径
        content_image_path=None,  # 内容图片的路径
        saveroot="./outputs",  # 输出图片路径
        prompt="A painting of a city skyline, in the style of {}",  # 内容图片的内容文本提示
        placeholder_token="<sks09>",  # 伪词
        num_stages=6,  # 分阶段数量
        num_samples=3,  # 输出几张图片
        resolution=512,  # 输出分辨率
        seed=None,  # 随机种子
):
    os.makedirs(saveroot, exist_ok=True)  # 当前目录下创建文件夹/outputs
    pipelinecn, processor,unet = load_model(  # 加载模型，函数在上面
        sd_path,
        controlnet_path,
        embedding_path,
        placeholder_token,
        num_stages,
    )
    generator = None if seed is None else torch.Generator(device="cuda").manual_seed(seed) # 一个随机数生成器
    cross_attention_kwargs = {"num_stages": num_stages}

    content_image = Image.open(content_image_path)
    content_image = content_image.resize((resolution, resolution))
    bw_image = content_image.convert("L")
    pipedp = pl(task="depth-estimation", model="E:\models_download\dpt-large")
    result = pipedp(content_image)["depth"]
    r = np.array(result)
    r = r[:, :, None]
    r = np.concatenate([r, r, r], axis=2)
    r = Image.fromarray(r)
    #r.show()
    pos_prompt = [prompt.format(f"{placeholder_token}-T{t}-L{l}") for t in range(num_stages) for l in range(1, 8)]
    # [<sks03>-01,<sks03>-02,<sks03>-03,<sks03>-04,<sks03>-05]

    outputs = []
    num = 0
    # torch.manual_seed(1)
    for _ in range(num_samples):
        output = pipelinecn(
            prompt=pos_prompt,
            num_inference_steps=50,
            generator=generator,
            image=bw_image,
            control_image=r,
            cross_attention_kwargs=cross_attention_kwargs,
            unet = unet,
            strength=1,
            guidance_scale=7.5
        ).images[0]
        outputs.append(output)

        out = np.asarray(output)
        save_path = ospj(saveroot, f"{content_image_path.split('/')[-1].split('.')[0]}_{num}.png")
        imageio.imsave(save_path, out)
        num += 1
        '''
        style_img = Image.open('D:/PythonWorkSpace/C0NStyle/images/09.png')
        # Clip分数
        model_c_ID = "D:/PythonWorkSpace/Pytorch_ZHM/models_need/clip-vit-base-patch32"
        model_c = CLIPModel.from_pretrained(model_c_ID)
        preprocess_c = CLIPImageProcessor.from_pretrained(model_c_ID)

        def calculate_clip_score(image_a: Image.Image, image_b: Image.Image) -> float:
            # Preprocess the images
            processed_a = preprocess_c(image_a, return_tensors="pt")["pixel_values"]
            processed_b = preprocess_c(image_b, return_tensors="pt")["pixel_values"]

            # Get embeddings
            with torch.no_grad():
                embedding_a = model_c.get_image_features(processed_a)
                embedding_b = model_c.get_image_features(processed_b)

            # Calculate cosine similarity
            similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
            return similarity_score.item()

        img1 = style_img
        img2 = output
        score = calculate_clip_score(img1, img2)
        print(score)
        if score >= 0.61:
            out = np.asarray(output)
            save_path = ospj(saveroot, f"{content_image_path.split('/')[-1].split('.')[0]}_{num}.png")
            imageio.imsave(save_path, out)
            num += 1
        '''
    '''
    #计算指标
    content_img = Image.open('D:/PythonWorkSpace/C0NStyle/images/ship.png')
    style_img = Image.open('D:/PythonWorkSpace/C0NStyle/images/01.png')
    #Clip分数
    model_c_ID = "D:/PythonWorkSpace/Pytorch_ZHM/models_need/clip-vit-base-patch32"
    model_c = CLIPModel.from_pretrained(model_c_ID)
    preprocess_c = CLIPImageProcessor.from_pretrained(model_c_ID)
    def calculate_clip_score(image_a: Image.Image, image_b: Image.Image) -> float:
        # Preprocess the images
        processed_a = preprocess_c(image_a, return_tensors="pt")["pixel_values"]
        processed_b = preprocess_c(image_b, return_tensors="pt")["pixel_values"]

        # Get embeddings
        with torch.no_grad():
            embedding_a = model_c.get_image_features(processed_a)
            embedding_b = model_c.get_image_features(processed_b)

        # Calculate cosine similarity
        similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
        return similarity_score.item()
    total_score = 0
    num_images = len(outputs)
    for img in outputs:
        img1 = style_img
        img2 = img
        score = calculate_clip_score(img1, img2)
        total_score += score
        #print('clip score style:',score)
    average_score = total_score / num_images if num_images > 0 else 0
    print('Average CLIP score style:', average_score)
    total_score = 0
    for img in outputs:
        img1 = content_img
        img2 = img
        score = calculate_clip_score(img1, img2)
        total_score += score
        #print('clip score content:',score)
    average_score = total_score / num_images if num_images > 0 else 0
    print('Average CLIP score content:', average_score)

    # 风格损失和感知损失
    class VGG19(nn.Module):
        def __init__(self):
            super(VGG19, self).__init__()
            self.features = models.vgg19(pretrained=True).features[:21]  # 使用VGG19前21层

        def forward(self, x):
            return self.features(x)

    class GramMatrix(nn.Module):
        def forward(self, input):
            b, c, h, w = input.size()
            features = input.view(b, c, h * w)
            G = torch.bmm(features, features.transpose(1, 2))
            return G / (c * h * w)

    class PerceptualLoss(nn.Module):
        def __init__(self, vgg, content_weight=1.0, style_weight=1.0):
            super(PerceptualLoss, self).__init__()
            self.vgg = vgg
            self.content_weight = content_weight
            self.style_weight = style_weight
            self.gram = GramMatrix()

        def forward(self, generated, content, style):
            # 提取特征
            generated_features = self.vgg(generated)
            content_features = self.vgg(content)
            style_features = self.vgg(style)
            content_loss = torch.mean((generated_features - content_features) ** 2) * self.content_weight
            generated_gram = self.gram(generated_features)
            style_gram = self.gram(style_features)
            style_loss = torch.mean((generated_gram - style_gram) ** 2) * self.style_weight

            return content_loss, style_loss
    def load_image(image, transform):
        image = image.convert('RGB')
        image = transform(image).unsqueeze(0)
        return image
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def compute_perceptual_loss(content_image, style_image, generated_image):
        vgg = VGG19().eval()
        perceptual_loss_fn = PerceptualLoss(vgg, content_weight=1.0, style_weight=1.0)

        content_tensor = load_image(content_image, preprocess)
        style_tensor = load_image(style_image, preprocess)
        generated_tensor = load_image(generated_image, preprocess)

        content_loss, style_loss = perceptual_loss_fn(generated_tensor, content_tensor, style_tensor)

        return content_loss.item(), style_loss.item()

    perceptual_loss_sum = 0
    style_loss_sum = 0
    for img in outputs:
        generated_img = img
        perceptual_loss, style_loss = compute_perceptual_loss(content_img, style_img, generated_img)
        perceptual_loss_sum += perceptual_loss
        style_loss_sum += style_loss
        #print(f"Perceptual Loss: {perceptual_loss}")
        #print(f"Style Loss: {style_loss}")
    average_perceptual_loss = perceptual_loss_sum / num_images if num_images > 0 else 0
    print('Average Perceptual Loss:', average_perceptual_loss)
    average_style_loss = style_loss_sum / num_images if num_images > 0 else 0
    print('Average Style_loss:', average_style_loss)

    #LPIPS
    loss_fn = lpips.LPIPS(net='alex')
    preprocess_l = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    lpips_value_sum = 0
    for img in outputs:
        img = ImageOps.grayscale(img)
        img = Image.merge("RGB", (img, img, img))
        img = preprocess_l(img).unsqueeze(0)  # 增加 batch 维度
        img2 = ImageOps.grayscale(content_img)
        img2 = Image.merge("RGB", (img2, img2, img2))
        img2 = preprocess_l(img2).unsqueeze(0)  # 增加 batch 维
        lpips_value = loss_fn(img, img2)
        lpips_value_sum += lpips_value
        #print('content lpips(gray):',lpips_value)
    average_content_lpips = lpips_value_sum / num_images if num_images > 0 else 0
    print('Average content lpips:', average_content_lpips)
    lpips_value_sum = 0
    for img in outputs:
        img = preprocess_l(img).unsqueeze(0)  # 增加 batch 维度
        img2 = preprocess_l(style_img).unsqueeze(0)  # 增加 batch 维
        lpips_value = loss_fn(img, img2)
        lpips_value_sum += lpips_value
        #print('style lpips:',lpips_value)
    average_style_lpips = lpips_value_sum / num_images if num_images > 0 else 0
    print('Average style lpips:', average_style_lpips)

    #SSIM
    ssim_sum = 0
    for img in outputs:
        ssim_value = structural_similarity(np.array(content_img), np.array(img), data_range=None, channel_axis=-1, multichannel=True)
        ssim_sum += ssim_value
        #print(f'SSIM: {ssim_value}')
    average_ssim = ssim_sum / num_images if num_images > 0 else 0
    print('Average ssim:', average_ssim)
    '''

    outputs = np.concatenate([np.asarray(img) for img in outputs], axis=1)
    save_path = ospj(saveroot, f"{content_image_path.split('/')[-1].split('.')[0]}.png")
    #imageio.imsave(save_path, outputs)
    print("1")
    image = Image.fromarray(outputs.astype('uint8'))
    #image.show()


if __name__ == "__main__":
    style_transfer()
