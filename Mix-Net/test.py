import torch
from PIL import Image
import random
from torchvision.utils import save_image
from torch.nn import functional as F
from Mix_Net import Dilated_UNET
import warnings
import os

from dataloader import MyDataset
from torch.utils import data
from train import training, validation, training_onehot, validation_onehot
from utils import read_image_pth
import time
from extramodels import Unet

warnings.filterwarnings("ignore")

full_to_train = {-1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3,
                 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                 26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18}
train_to_full = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26,
                 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 19: 0}
# full_to_colour = {0: (128, 64, 128), 7: (244, 35, 232), 8: (70, 70, 70), 11: (102, 102, 156), 12: (190, 153, 153),
#                   13: (153, 153, 153), 17: (250, 170, 30), 19: (220, 220, 0), 20: (107, 142, 35), 21: (152, 251, 152),
#                   22: (0, 130, 180), 23: (220, 20, 60), 24: (255, 0, 0), 25: (0, 0, 142), 26: (0, 0, 70),
#                   27: (0, 60, 100), 28: (0, 80, 100), 31: (0, 0, 230), 32: (119, 11, 32), 33: (0, 0, 0)}
full_to_colour = {0: (0, 0, 0), 7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70), 12: (102, 102, 156),
                  13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30), 20: (220, 220, 0), 21: (107, 142, 35),
                  22: (152, 251, 152), 23: (70, 130, 180), 24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142),
                  27: (0, 0, 70), 28: (0, 60, 100), 31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}

pth = 'C:\\Users\\tanjingbin\\Desktop\\Semantic_Segmentation-main\\ShuffleNet-v2\\Mix_Net\\weights\\10_t.pth'
crop_size = 512
num_classes = 20


# if not os.path.exists("images/test/test_result.csv"):
#     root = 'data/'
#     split = 'test/'
#     test_result_csv(root, split)
#
# # test_data_loader
# validation_dataset = MyDataset('images/test/test_result.csv', transform=None)
# val_loader_args = dict(batch_size=1, shuffle=False, num_workers=0)
# val_loader = data.DataLoader(validation_dataset, **val_loader_args)
#
# path = val_loader


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = Dilated_UNET()
    model1.load_state_dict(torch.load(pth))
    net = model1.to(device)
    net.eval()

    read_pth = 'C:\\Users\\tanjingbin\\Desktop\\Semantic_Segmentation-main\\data\\leftImg8bit\\test\\'
    read_pth_result = read_image_pth(read_pth)
    start_time = time.time()
    # for i in range(len(read_pth_result)):
    picture_nums = 4
    for i in range(picture_nums):
        input = Image.open(read_pth_result[i])
        w, h = input.size
        w_resize = int(w / 2)
        h_resize = int(h / 2)
        input = input.resize((w_resize, h_resize), Image.ANTIALIAS)

        # x1, y1 = random.randint(0, w - crop_size), random.randint(0, h - crop_size)
        # input = input.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        w, h = input.size
        input = torch.ByteTensor(torch.ByteStorage(input.tobytes())).view(h, w, 3).permute(2, 0, 1).float().div(
            255)

        input[0].add_(-0.485).div_(0.229)
        input[1].add_(-0.456).div_(0.224)
        input[2].add_(-0.406).div_(0.225)
        input = input.to(device)
        input = input.unsqueeze(0)
        output = F.log_softmax(net(input))
        b, _, h, w = output.size()
        pred = output.permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, h, w)
        pred = pred.data.cpu()
        pred_remapped = pred.clone()

        for k, v in train_to_full.items():
            pred_remapped[pred == k] = v
        pred = pred_remapped
        pred_colour = torch.zeros(b, 3, h, w)
        for k, v in full_to_colour.items():
            pred_r = torch.zeros(b, 1, h, w)
            pred = pred.reshape(1, 1, h, -1)
            pred_r[(pred == k)] = v[0]
            pred_g = torch.zeros(b, 1, h, w)
            pred_g[(pred == k)] = v[1]
            pred_b = torch.zeros(b, 1, h, w)
            pred_b[(pred == k)] = v[2]
            pred_colour.add_(torch.cat((pred_r, pred_g, pred_b), 1))
        # print(pred_colour[0].float())
        # print('-----------------')
        pred = pred_colour[0].float().div(255)
        # print(pred)
        # save_pth = 'C:\\Users\\tanjingbin\\Desktop\\Semantic_Segmentation-main\\images\\'
        # img_pth = save_image_pth(save_pth)
        print('正在处理%s文件夹的图片' % (read_pth_result[i].split('\\')[-2]), '图片名称为：',
              read_pth_result[i].split('\\')[-1], '它是第%d/%d张图片' % (i + 1, len(read_pth_result)))
        save_image(pred,
                   'C:\\Users\\tanjingbin\\Desktop\\Semantic_Segmentation-main\\ShuffleNet-v2\\images\\test\\' +
                   read_pth_result[i].split('\\')[-2] + '\\' + str(
                       i) + '_test.png')
    Time = time.time() - start_time
    FPS = picture_nums / Time
    print('处理所有图片所用时间为：', Time, '秒')
    print('FPS为：', FPS, 'FPS')
    # save_image(pred_colour, 'C:\\Users\\tanjingbin\\Desktop\\Semantic_Segmentation-main\\images\\1116_test.png')


test()
