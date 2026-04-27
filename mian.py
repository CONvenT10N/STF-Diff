from PIL import Image
import os


def resize_image(input_path, output_path, size=(512, 512)):
    # 打开图片
    img = Image.open(input_path)

    # 调整图片大小
    img_resized = img.resize(size)

    # 检查输出目录是否存在，不存在则创建
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存图片到指定路径
    img_resized.save(output_path)
    print(f"Image saved to {output_path}")


# 示例用法
input_image_path = "D:/PythonWorkSpace/C0NStyle/images/01.png"  # 输入图片的路径
output_image_path = "C:/Users/C0NvenT10N/Desktop/Figure_1.png"  # 输出图片的路径

resize_image(input_image_path, output_image_path)

