import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import torchvision
import warnings
import pdb
from dataloader import MyDataset
from Mix_Net import Dilated_UNET
from train import training, validation, training_onehot, validation_onehot
from utils import data_csv, diceLoss, init_weights
import random
import pandas as pd
from extramodels import Unet
from datetime import datetime
from dice_loss import MulticlassDiceLoss
from Focal_loss import FocalLoss

if __name__ == '__main__':

    # arguments for num_epochs and batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='arg1', type=int, help="Number of Epochs")
    parser.add_argument(dest='arg2', type=int, default=16, help="Batch Size")

    args = parser.parse_args()
    num_epochs = args.arg1
    batch_size = args.arg2

    print(num_epochs, batch_size)

    # Making folders to save reconstructed images, input images and weights
    if not os.path.exists("Mix_Net/outputs"):
        os.mkdir("Mix_Net/outputs")

    if not os.path.exists("Mix_Net/inputs"):
        os.mkdir("Mix_Net/inputs")

    if not os.path.exists("Mix_Net/weights"):
        os.mkdir("Mix_Net/weights")

    if not os.path.exists("data/train_data.csv"):
        root = 'C:/Users/tanjingbin/Desktop/Semantic_Segmentation-main/data/'
        for split in ['train/', 'val/', 'test/']:
            # for split in ['test/']:
            data_csv(root, split)

    if not os.path.exists("Mix_Net/final_train_data"):
        os.mkdir("Mix_Net/final_train_data")

    if not os.path.exists("Mix_Net/final_train_data/train_data.csv"):
        # 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
        df = pd.DataFrame(columns=['Epoch no', 'Train Loss', 'Time'])  # 列名
        df.to_csv("Mix_Net/final_train_data/train_data.csv", index=False)  # 路径可以根据需要更改

    if not os.path.exists("Mix_Net/final_val_data"):
        os.mkdir("Mix_Net/final_val_data")

    if not os.path.exists("Mix_Net/final_val_data/val_data.csv"):
        df1 = pd.DataFrame(columns=['Epoch no', 'Train Loss', 'Time', 'Validation Loss', 'Mean IOU', 'Avg Mean IOU'])
        df1.to_csv("Mix_Net/final_val_data/val_data.csv", index=False)

    if not os.path.exists("Mix_Net/optimizer_data"):
        os.mkdir("Mix_Net/optimizer_data")

    if not os.path.exists("Mix_Net/optimizer_data/optimizer_data.csv"):
        df1 = pd.DataFrame(columns=['optimizer'])
        df1.to_csv("Mix_Net/optimizer_data/optimizer_data.csv", index=False)

    warnings.filterwarnings('ignore')

    ##transformer characteristics
    augment = [
        transforms.RandomRotation((0, 10)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomCrop(256)
    ]
    tfs = transforms.Compose(augment)

    ##Train Loader
    train_dataset = MyDataset('data/train_data.csv', transform=None)
    train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    # print(len(train_loader))

    # val data_loader
    validation_dataset = MyDataset('data/val_data.csv', transform=None)
    val_loader_args = dict(batch_size=1, shuffle=False, num_workers=0)
    val_loader = data.DataLoader(validation_dataset, **val_loader_args)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MODEL instance and xavier initialization of the weights
    model = Dilated_UNET()
    model.apply(init_weights)
    model = model.to(device)

    # Optimizer, criterion and scheduler
    # optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    # criterion = MulticlassDiceLoss()
    # 动态更新学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.1, patience=5, verbose=True,
                                                           threshold=1e-4, threshold_mode='rel',
                                                           cooldown=5, min_lr=1e-6, eps=1e-08)
    # Path = 'weights/100_t.pth'
    # Path = 'Mix_Net/weights/ff0_t.pth'
    # model.load_state_dict(torch.load(Path))
    # print(optimizer)

    # inp, output, Validation_Loss, Mean_IOU, Avg_Mean_IOU = validation(model, val_loader, criterion)
    # # name = 'outputs/final_out.npy'
    # # name_in = 'inputs/final_label.npy'
    # name = 'Depthwise_Separable_Mix_Net/outputs/final_50_out.npy'
    # name_in = 'Depthwise_Separable_Mix_Net/inputs/final_50_label.npy'
    # np.save(name, output)
    # del output
    # np.save(name_in, inp)
    # del inp
    # del Validation_Loss
    # del Mean_IOU
    # del Avg_Mean_IOU

    # Num epochs for Training and Validation functions
    for epoch in range(num_epochs):
        start_time = time.time()
        print('Epoch no: ', epoch)
        train_loss = training(model, train_loader, criterion, optimizer)

        # Saving weights after every 20 epochs
        if epoch % 5 == 0:
            inp, output, Validation_Loss, Mean_IOU, Avg_Mean_IOU = validation(model, val_loader, criterion)
            name = 'Mix_Net/outputs/' + str(epoch) + '.npy'
            name_in = 'Mix_Net/inputs/' + str(epoch) + '.npy'
            np.save(name, output)
            del output
            np.save(name_in, inp)
            del inp
            Time1 = time.time() - start_time
            list2 = [epoch, train_loss, Time1, Validation_Loss, Mean_IOU, Avg_Mean_IOU]
            data = pd.DataFrame([list2])
            data.to_csv('Mix_Net/final_val_data/val_data.csv', mode='a', header=False,
                        index=False)

        if epoch % 5 == 0:
            path = 'Mix_Net/weights/' + str(epoch) + '_t.pth'
            torch.save(model.state_dict(), path)
            print(optimizer)
            list3 = [optimizer]
            data = pd.DataFrame([list3])
            data.to_csv('Mix_Net/optimizer_data/optimizer_data.csv', mode='a',
                        header=False,
                        index=False)

        scheduler.step(train_loss)
        Time = time.time() - start_time
        print("Time : ", Time)
        print('=' * 50)
        list1 = [epoch, train_loss, Time]
        data = pd.DataFrame([list1])
        data.to_csv('Mix_Net/final_train_data/train_data.csv', mode='a', header=False,
                    index=False)
