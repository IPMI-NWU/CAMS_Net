import torch

# 读数据
from read_data.MyDataset3_more_porcess_no_contour import MyDataset
# from read_data.MyDataset3_sobel.MyDataset3_more_porcess_no_contour import MyDataset  # sobel

import torch.utils.data as Data
from LossAndEval import dice_coef, precision, recall, iou_score, acc

import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


# 保存图像, path为保存图片的父父目录，reg_result_name为4个类，img_name为test中的文件名
def save_img(output, img_name, fold, model_name):
    path = r'../../result/xiaorong_MFAB_class/' + model_name
    mkdir(path)

    path = os.path.join(path, 'fold_' + str(fold))
    mkdir(path)

    seg_result_name = ['prerib','all_bone','clavicel', 'postrib']#

    for name_index in range(len(seg_result_name)):
        result_path = os.path.join(path, seg_result_name[name_index])
        mkdir(result_path)

        bone = getResult(output[seg_result_name[name_index]])

        w, h = 512, 512
        plt.figure(figsize=(w, h), dpi=1)
        # img = Image.open(img)
        plt.imshow(bone, cmap='gray')

        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        # 图片的路径
        save_img_path = os.path.join(result_path, img_name)
        plt.savefig(save_img_path, format='png', transparent=True, dpi=1, pad_inches=0)


def getResult(bone):
    bone = bone.squeeze(0).squeeze(0).cpu()
    bone = (bone > 0.5).float() * 255

    return bone


# 计算指标和存储图像
def infer(testloader, outTxtPath, fold, model_index, model_name, txt_path_img_dice):
    # 导入模型
    model_pth = [
        # CAMS-Net 0
        [
            'model_pth/fold_0/unet_resblock20-0.898231.pth',
            'model_pth/fold_1/unet_resblock21-0.904950.pth',
            'model_pth/fold_2/unet_resblock20-0.905194.pth',
            'model_pth/fold_3/unet_resblock20-0.895635.pth'
        ],
    ]

    model_path = ''
    if fold == 0:
        #     第一折
        model_path = os.path.join(r'../../', model_pth[model_index][0])
    elif fold == 1:
        # 第二折
        model_path = os.path.join(r'../../', model_pth[model_index][1])
    elif fold == 2:
        # 第三折
        model_path = os.path.join(r'../../', model_pth[model_index][2])
    elif fold == 3:
        # 第四折
        model_path = os.path.join(r'../../', model_pth[model_index][3])

    net = torch.load(model_path)
    net.eval()

    test_loss = 0
    # dice, iou(jaccard), precision, recall = 0, 0, 0, 0
    dice = [0, 0, 0, 0]
    iou = [0, 0, 0, 0]
    precision_ = [0, 0, 0, 0]
    recall_ = [0, 0, 0, 0]
    acc_ = [0, 0, 0, 0]

    mask_num = 4
    # labels_name = ['所有骨', '锁骨', '后肋', '前肋']
    labels_name = ['all_bone', 'clavicle', 'postrib', 'prerib']
    with torch.no_grad():
        for step, (imgs, mask, file_name) in enumerate(testloader):
            mask_zeros = torch.zeros(1, 1, 512, 512).cuda()

            imgs = imgs.float()
            # 所有骨，锁骨，后肋，前肋
            for i in range(len(mask)):
                mask[i] = mask[i].float()

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                for i in range(len(mask)):
                    mask[i] = mask[i].cuda()

            output1, output2, output3, output4 = net(imgs, [mask_zeros, mask_zeros, mask_zeros, mask_zeros])  #

            all_bone = getResult(output1)
            clavicel = getResult(output2)
            postrib = getResult(output3)
            prerib = getResult(output4)

            output = {'all_bone': all_bone, 'clavicel': clavicel, 'postrib': postrib, 'prerib': prerib}

            img_name = file_name[0].split('/')[-1]
            # print(img_name)
            # save_img(output, img_name, fold, model_name)

            masks_probs = []
            masks_probs_binary = []
            masks_probs.append(F.sigmoid(output1))
            masks_probs.append(F.sigmoid(output2))
            masks_probs.append(F.sigmoid(output3))
            masks_probs.append(F.sigmoid(output4))

            for i in range(mask_num):
                masks_probs_binary.append((masks_probs[i] > 0.5).float())

            # 真实的label
            true_mask_binary = []
            for i in range(len(mask)):
                true_mask_binary.append((mask[i] > 0.5).float())

            img_dice = img_name + ':\t'
            # 计算指标
            for i in range(mask_num):
                dice[i] = dice[i] + dice_coef(masks_probs_binary[i], true_mask_binary[i])
                iou[i] = iou[i] + iou_score(masks_probs_binary[i], true_mask_binary[i])
                precision_[i] = precision_[i] + precision(masks_probs_binary[i], true_mask_binary[i])
                recall_[i] = recall_[i] + recall(masks_probs_binary[i], true_mask_binary[i])
                acc_[i] = acc_[i] + acc(masks_probs_binary[i], true_mask_binary[i])

                img_dice = img_dice + labels_name[i] + '_dice: {:.4}'.format(
                    dice_coef(masks_probs_binary[i], true_mask_binary[i]).data.cpu().item()) + '\t'
                # str(dice_coef(masks_probs_binary[i], true_mask_binary[i]).data.cpu().item()) + '\t'

            with open(txt_path_img_dice, 'a+') as shuchuDice:
                shuchuDice.write(img_dice + '\n')

            # loss = net.loss
            # test_loss += loss.data.cpu().item()

    # 平均的指标
    for i in range(mask_num):
        dice[i] = dice[i] / (step + 1)
        iou[i] = iou[i] / (step + 1)
        precision_[i] = precision_[i] / (step + 1)
        recall_[i] = recall_[i] / (step + 1)
        acc_[i] = acc_[i] / (step + 1)

    shuchu = 'Test Loss: {:.6f}'.format(test_loss / (step + 1))

    with open(outTxtPath, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')
    print(shuchu)

    for i in range(mask_num):
        shuchu = labels_name[i] + ' dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4} acc:{:.4}'. \
            format(dice[i], precision_[i], recall_[i], iou[i], acc_[i])
        with open(outTxtPath, 'a+') as shuchuTxt:
            shuchuTxt.write(shuchu + '\n')
        print(shuchu)

    dice_total = 0
    iou_total = 0
    precision_total = 0
    recall_total = 0
    acc_total = 0
    for i in range(mask_num):
        dice_total = dice[i] + dice_total
        iou_total = iou_total + iou[i]
        precision_total = precision_total + precision_[i]
        recall_total = recall_total + recall_[i]
        acc_total = acc_total + acc_[i]

    print('average: ' + 'dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4} acc:{:.4}'. \
          format(dice_total / mask_num, precision_total / mask_num, recall_total / mask_num, iou_total / mask_num, acc_total / mask_num))

    shuchu = 'average: ' + 'dice: {:.6f} precision:{:.4} recall:{:.4} iou:{:.4} acc:{:.4}'. \
        format(dice_total / mask_num, precision_total / mask_num, recall_total / mask_num, iou_total / mask_num, acc_total / mask_num)
    with open(outTxtPath, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')


if __name__ == '__main__':
    num = 4
    model_index = 0
    txt_path = r'./xiaorong_acc.txt'

    txt_img_dice_path = r'./Xiaorong_acc_IMG_DICE.txt'

    shuchu_list = ['cams-net']

    shuchu = shuchu_list[model_index] + '修正hou数据'
    print(shuchu)

    with open(txt_path, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')
    with open(txt_img_dice_path, 'a+') as shuchuTxt:
        shuchuTxt.write(shuchu + '\n')

    for i in range(0, num):
        shuchu = "=" * 50 + '\n' + "第" + str(i + 1) + "折" + '\n' + "=" * 50
        with open(txt_path, 'a+') as shuchuTxt:
            shuchuTxt.write(shuchu + '\n')

        with open(txt_img_dice_path, 'a+') as shuchuTxt:
            shuchuTxt.write(shuchu + '\n')

        print(shuchu)

        # 此处为四折test的txt文件的路径，可以根据自己的进行修改
        test_img_label_txt = r'/home/zdd2020/zdd_experiment/3_15_experiment_set/image_preprocess/images/txt/fold_' + str(
            i) + '/test.txt'

        # 使用Mydataset读入数据
        test_datasets = MyDataset(test_img_label_txt, mode='test')
        testloader = Data.DataLoader(dataset=test_datasets, batch_size=1, shuffle=False, num_workers=0)

        infer(testloader, txt_path, i, model_index, shuchu_list[model_index], txt_img_dice_path)
