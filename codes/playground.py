from utils.train import run
from utils.FIW300 import FIW300
from mobilenetv1 import MobileNetV1
from wingloss import WingLoss

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=100, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--loss', default=False, action='store_true', help='augment the data by duration.')
    # parser.add_argument('--no-aug', dest='aug', action='store_false', help='do not augment the data by duration')
    parser.add_argument('--gpus', default=1, type=int)
    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batchsize
    num_gpus = args.gpus

    fiw300 = FIW300('/projects/facialLandmark/data/300w_cropped/')
    train_sampler = SubsetRandomSampler(range(400))
    val_sampler = SubsetRandomSampler(range(400, 500))

    if torch.cuda.device_count() > 1 and num_gpus > 1:
        model = nn.DataParallel(MobileNetV1())
    else:
        model = MobileNetV1()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training with {num_gpus} GPUs, batch size = {batch_size}, # epochs = {num_epochs}')
    
    train_loader = DataLoader(fiw300, batch_size=batch_size, sampler=train_sampler, num_workers=6)
    val_loader = DataLoader(fiw300, batch_size=batch_size, sampler=val_sampler, num_workers=6)
    # criterion = nn.SmoothL1Loss()
    criterion = WingLoss(w=0.01, eps=0.002, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    try:
        run(num_epochs, device, model, train_loader, val_loader, criterion, optimizer)
    except KeyboardInterrupt:
        pass

    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()

    torch.save(state_dict, f'./models/checkpoint_{int(time.time())}.pt')
    torch.cuda.empty_cache()