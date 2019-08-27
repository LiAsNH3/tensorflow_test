"""
从猫狗测试图片集中选出部分图片拼接，作为论文样本展示
"""
import PIL.Image as Image
import os
import input_data
import random as rd

IMAGES_PATH = 'C:/Users/lsa/Desktop/deep_learning/cats_and_dogs/data/train/'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
IMAGE_SIZE = 128  # 每张小图片的大小
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 6  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'doc/result_figure/final.eps'  # 图片转换后的地址


# 定义图像拼接函数
def image_compose(test):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(test[rd.randint(0, len(test))]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图

if __name__ == "__main__":
    # 获取图片集地址下的所有图片名称
    test, test_label = input_data.get_files(IMAGES_PATH)
    image_compose(test)  # 调用函数
