import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoPipelineForImage2Image

# 加载预训练的CLIP模型和处理器
model = CLIPModel.from_pretrained("D:/PythonWorkSpace/Pytorch_ZHM/models_need/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("D:/PythonWorkSpace/Pytorch_ZHM/models_need/clip-vit-base-patch32")
pipeline = AutoPipelineForImage2Image.from_pretrained("E:/models_download/sdv15", torch_dtype=torch.float16,
                                                      use_safetensors=True, ).to("cuda")
prompt1 = "A Realistic photo of "  # 替换为你的文本
prompt2 = "in the style of realistic"


# 加载和预处理图像
image_path = "D:\PythonWorkSpace\C0NStyle\images/11.png"  # 替换为你的图片路径
image = Image.open(image_path)
image_d = Image.open(image_path).convert("RGB")
image_d.thumbnail((768, 768))
# 文本
text = "A woman walks in the rain, her colorful umbrella brightening the urban scene. Rich hues of red, yellow, blue, and green contrast with the vibrant cityscape, where buildings glow in orange, purple, and green. A lively canvas of movement and contrast."  # ???????
#text = "A blue wooden door with a blue closed window next to it, and a pot of flowers planted in a brown pot under the window,The overall style is watercolor painting."  # 替换为你的文本
words = text.split()
prompt = prompt2+text+prompt1
image_d = pipeline(prompt, image, negative_prompt = "Abstract,Surreal,Cartoonish,Painterly,Impressionistic,Sketchy,Fantasy",num_inference_steps=50, strength=1, guidance_scale=3.5).images[0]
image_l = image_d.convert("L")

# 显示黑白图片
#image_l.show()

# 计算每个词与图像的距离
distances = []
for word in words:
    # 将图像和单词转化为CLIP的输入格式
    inputs = processor(text=[word], images=image, return_tensors="pt")

    # 获取图像和文本的特征向量
    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    # 计算余弦相似度并转换为距离（1 - 相似度）
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    distance = 1 - similarity.item()
    distances.append((word, distance))

distancesl = []
for word in words:
    # 将图像和单词转化为CLIP的输入格式
    inputsl = processor(text=[word], images=image_l, return_tensors="pt")

    # 获取图像和文本的特征向量
    with torch.no_grad():
        outputsl = model(**inputsl)
        image_featuresl = outputsl.image_embeds
        text_featuresl = outputsl.text_embeds

    # 计算余弦相似度并转换为距离（1 - 相似度）
    similarityl = torch.nn.functional.cosine_similarity(image_featuresl, text_featuresl)
    distancel = 1 - similarityl.item()
    distancesl.append((word, distancel))

# 输出结果
for word, distance in distances:
    print(f"Word: {word}, Distance: {distance}")
for word, distance in distancesl:
    print(f"Word: {word}, Distance: {distance}")

# 对比距离并过滤单词
filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0:", final_sentence)

filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0.005 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0.005:", final_sentence)

filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0.01 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0.01:", final_sentence)

filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0.015 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0.015:", final_sentence)

filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0.02 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0.02:", final_sentence)

filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0.03 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0.03:", final_sentence)

filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0.035 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0.035:", final_sentence)

filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0.04 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0.04:", final_sentence)

filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0.045 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0.045:", final_sentence)

filtered_words = []
for (word, distance), (_, distancel) in zip(distances, distancesl):
    if distance<=0.81 and abs(distance - distancel) <=0.05 : # 默认:0.025 如果是0就啥都没了，如果是1就全都留下来
        filtered_words.append(word)

# 输出最终的句子
final_sentence = " ".join(filtered_words)
print("Filtered sentence0.05:", final_sentence)

