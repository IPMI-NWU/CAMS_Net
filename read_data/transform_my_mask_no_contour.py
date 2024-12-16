from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

import random
import cv2
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image
from torchvision.transforms import RandomAffine
# from torchvision.transforms.functional import InterpolationMode

def affine(image, shear):
    random_affine = RandomAffine(degrees=0, translate=None, scale=None, shear=shear, resample=Image.BILINEAR)
    return random_affine(image)

# 旋转
def transform_rotate(image, mask):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([0, 36])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle, expand=True)


    for i in range(len(mask)):
        mask[i] = mask[i].rotate(angle, expand=True)

    # image = tf.to_tensor(image)
    # mask = tf.to_tensor(mask)
    return image, mask

# 错切
def transform_shear(image, mask):
    angle = transforms.RandomRotation.get_params([-10, 10])
    # 水平错切
    if random.random() > 0.5:
        image = affine(image, shear=(angle, angle, 0, 0))
        for i in range(len(mask)):
            mask[i] = affine(mask[i], shear=(angle, angle, 0, 0))
    # 垂直错切
    else:
        image = affine(image, shear=(0, 0, angle, angle))
        for i in range(len(mask)):
            mask[i] = affine(mask[i], shear=(0, 0, angle, angle))

    return image, mask

# 水平/垂直翻转
def transform_flip(image, mask):
    if random.random() > 0.5:
        image = tf.hflip(image)
        for i in range(len(mask)):
            mask[i] = tf.hflip(mask[i])
    else:
        image = tf.vflip(image)
        for i in range(len(mask)):
            mask[i] = tf.vflip(mask[i])

    return image, mask

# 水平移动
# padding 为int/tuple，为一个时，上下左右都填充，为两个时分别用于填充left/right和top/bottom,
# 为4时，分别用来填充left,top,right,和bottom
def transform_translate_horizontal(image, mask, scale=0.5):
    # 获取image的长宽
    w, h = image.size
    # 如果随机数大于0.5，则裁剪左边，否则，裁剪右边
    mask_pad = []
    mask_contour_pad = []
    if random.random() > 0.5:
        image = tf.crop(image, top=0, left=0, height=h, width=w - w*scale)
        image_pad = tf.pad(image, padding=[0, 0, int(w*scale), 0], fill=0)

        for i in range(len(mask)):
            mask[i] = tf.crop(mask[i], top=0, left=0, height=h, width=w - w*scale)
            mask_pad.append(tf.pad(mask[i], padding=[0, 0, int(w*scale), 0], fill=0))

    else:
        image = tf.crop(image, top=0, left=w*scale, height=h, width=w - w * scale)
        image_pad = tf.pad(image, padding=[int(w * scale), 0, 0,  0], fill=0)

        for i in range(len(mask)):
            mask[i] = tf.crop(mask[i], top=0, left=w*scale, height=h, width=w - w * scale)
            mask_pad.append(tf.pad(mask[i], padding=[int(w * scale), 0, 0,  0], fill=0))

    return image_pad, mask_pad

# 上下平移
def transform_translate_vertical(image, mask, scale=0.5):
    # 获取image的长宽
    w, h = image.size
    # 如果随机数大于0.5，则裁剪下边，否则，裁剪上边
    mask_pad = []
    mask_contour_pad = []
    if random.random() > 0.5:
        image = tf.crop(image, top=h*scale, left=0, height=h - h*scale, width=w)
        image_pad = tf.pad(image, padding=[0, int(h*scale), 0,  0], fill=0)

        for i in range(len(mask)):
            mask[i] = tf.crop(mask[i], top=h*scale, left=0, height=h - h*scale, width=w)
            mask_pad.append(tf.pad(mask[i], padding=[0, int(h*scale), 0,  0], fill=0))

    else:
        image = tf.crop(image, top=0, left=0, height=h - h*scale, width=w)
        image_pad = tf.pad(image, padding=[0, 0, 0, int(h * scale)], fill=0)

        for i in range(len(mask)):
            mask[i] = tf.crop(mask[i], top=0, left=0, height=h - h*scale, width=w)
            mask_pad.append(tf.pad(mask[i], padding=[0, 0, 0, int(h * scale)], fill=0))

    return image_pad, mask_pad


if __name__ == '__main__':
    img_path = r'E:\研一\数据集\蒙哥马利数据集\MontgomeryCXR\MontgomerySet\CXR_png\MCUCXR_0001_0.png'
    label_path = r'E:\研一\数据集\蒙哥马利数据集\MontgomeryCXR\MontgomerySet\ManualMask\leftMask\MCUCXR_0001_0.png'
    img = Image.open(img_path).convert('L')
    label = Image.open(label_path).convert('L')

    print(img.size)
    image = transform_translate_vertical(img, label)
    plt.imshow(image, cmap='gray')
    plt.show()

    plt.imshow(img, cmap='gray')
    plt.show()

    # img_tensor, label_tensor = transform_rotate(img, label)
    #
    # img_rotate = transforms.ToPILImage()(img_tensor).convert('L')
    # label_rotate = transforms.ToPILImage()(label_tensor).convert('L')
    #
    # plt.imshow(img_rotate)
    # plt.show()
    #
    # plt.imshow(label_rotate)
    # plt.show()
