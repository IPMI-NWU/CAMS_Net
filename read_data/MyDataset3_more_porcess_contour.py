from torch.utils.data import Dataset
import torch
import torch.utils.data as Data
import scipy.io as scio
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import os
import pdb
import numpy as np
from transform_my_mask_contour import transform_rotate, transform_translate_horizontal, transform_translate_vertical, transform_flip, transform_shear

class RandomGaussianBlur(object):
    def __call__(self, img):
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img


def default_loader(path, IorM='rgb'):
    if IorM == 'rgb':
        return Image.open(path)
    else:
        return Image.open(path).convert('L')


'''
    img_label_txt: 存储img和label的txt文件，其中第一个为原图，第二个为所有骨，第三个为后肋，第四个为前肋
'''

class MyDataset(Dataset):
    def __init__(self, img_label_txt, loader=default_loader, mode='test'):
        # print(imgtxt)
        img_label = []
        path = open(img_label_txt, 'r')
        for line in path:
            line = line.strip('\n')
            line = line.rstrip()
            img_label.append(line)

        self.img_label = img_label
        self.imgs_num = len(img_label) # 记住共有多少个文件名

        self.toTensor = transforms.ToTensor()
        self.HorizontalFlip = transforms.RandomHorizontalFlip(p=0.5)

        self.degrees = random.uniform(0, 10)
        # self.RandomAffine_degrees = transforms.RandomAffine(degrees=self.degrees)  # 转

        self.ColorJitter = transforms.ColorJitter(brightness=0.1)
        self.RandomGaussianBlur = RandomGaussianBlur()
        # self.resize = transforms.Resize((320, 320))

        self.resize = transforms.Resize((512, 512))
        self.loader = loader
        self.mode = mode

        # self.RandomGaussianBlur = tr.RandomGaussianBlur()


    def __getitem__(self, index):

        # imgname = self.imgs[index]
        imglabel = self.img_label[index]

        # 0 原图的存放路径
        # 1 所有骨的存放路径
        # 2 锁骨的存放路径
        # 3 后肋的存放路径
        # 4 前肋的存放路径

        # 将路径进行分割
        temp = imglabel.strip().split('\t')
        # print(index)
        # print(temp)

        # 原图片
        img = self.loader(temp[0], IorM='L')

        # 所有骨mask
        label_mask = self.loader(temp[1], IorM='Binary')
        # 锁骨mask
        labelClavicel_mask = self.loader(temp[2], IorM='Binary')
        # 后肋mask
        labelPosteriorrib_mask = self.loader(temp[3], IorM='Binary')
        # 前肋mask
        labelPrerib_mask = self.loader(temp[4], IorM='Binary')

        # 所有骨mask_contour
        label_mask_contour = self.loader(temp[5], IorM='Binary')
        # 锁骨mask_contour
        labelClavicel_mask_contour = self.loader(temp[6], IorM='Binary')
        # 后肋mask_contour
        labelPosteriorrib_mask_contour = self.loader(temp[7], IorM='Binary')
        # 前肋mask_contour
        labelPrerib_mask_contour = self.loader(temp[8], IorM='Binary')


        mask = [label_mask, labelClavicel_mask, labelPosteriorrib_mask, labelPrerib_mask]
        mask_contour = [label_mask_contour, labelClavicel_mask_contour, labelPosteriorrib_mask_contour, labelPrerib_mask_contour]
        # print(os.path.join(self.img_path[index//self.imgs_num], imgname))
        # print(os.path.join(self.label_path[0], imgname))

        if self.mode == 'train':
            rand = random.random()
            if random.random() > 0.5:  # 水平翻转
                img, mask, mask_contour = transform_flip(img, mask, mask_contour)

            # 平移
            if random.random() < 0.25:
                img, mask, mask_contour = transform_translate_horizontal(img, mask, mask_contour, scale=random.uniform(0, 0.05))
            if random.random() < 0.25:
                img, mask, mask_contour = transform_translate_vertical(img, mask, mask_contour, scale=random.uniform(0, 0.05))

            # 旋转
            if random.random() < 0.25:
                img, mask, mask_contour = transform_rotate(img, mask, mask_contour)

            # 错切
            if random.random() < 0.25:
                img, mask, mask_contour = transform_shear(img, mask, mask_contour)

            if random.random() >= 0.6:  # 对比度变换
                img = self.ColorJitter(img)

            # img = self.RandomGaussianBlur(img)

        img = self.resize(img)
        img = self.toTensor(img)

        for i in range(len(mask)):
            mask[i] = self.resize(mask[i])
            mask[i] = self.toTensor(mask[i])

        for i in range(len(mask_contour)):
            mask_contour[i] = self.resize(mask_contour[i])
            mask_contour[i] = self.toTensor(mask_contour[i])

        return img, mask, mask_contour, temp[0]

    def __len__(self):
        return len(self.img_label)  # 与index相关的



if __name__ == '__main__':
    img_label_txt = r'/home/zdd2020/zdd_experiment/All_Data/bone/fold_txt/fold'+str(1)+'/train_contour.txt'

    train_datasets = MyDataset(img_label_txt, mode='train')
    trainloader = Data.DataLoader(dataset=train_datasets, batch_size=2, shuffle=False, num_workers=0)

    for i in range(10):
        print(i)
        for step, (imgs, mask, mask_contour, _) in enumerate(trainloader):
            print(mask[0].shape)
            pass
            # print(len(_))
            # print(_[1])

            # imgs = imgs.float()
            # label_mask = label_mask.float()
            # labelClavicel_mask = labelClavicel_mask.float()
            # labelPosteriorrib_mask = labelPosteriorrib_mask.float()
            # labelPrerib_mask = labelPrerib_mask.float()



