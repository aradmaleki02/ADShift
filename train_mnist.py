import subprocess

import torch
from torchvision.datasets import ImageFolder

from mvtec import MVTEC
from resnet import wide_resnet50_2, resnet18
from resnet_TTA import wide_resnet50_2 as wide_TTA
from resnet_TTA import resnet18 as res_TTA

from de_resnet import de_wide_resnet50_2, de_resnet18
from torch.nn import functional as F
import torchvision.transforms as transforms
from dataset import AugMixDatasetMVTec, get_data_transforms, prepare_br35h_dataset_files, \
    prepare_brats2015_dataset_files, BrainTrain, MNIST_Dataset
import argparse
from tqdm import tqdm
import os
from inference_mvtec_ATTA import evaluation_ATTA
from pathlib import Path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_fucntion(a, b):
    # mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        # print(a[item].shape)
        # print(b[item].shape)
        # loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def loss_fucntion_last(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # for item in range(len(a)):
    #     # print(a[item].shape)
    #     # print(b[item].shape)
    #     # loss += 0.1*mse_loss(a[item], b[item])
    item = 0
    loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                    b[item].view(b[item].shape[0], -1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        # loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss


def train(_class_, backbone, batch_size, epochs, save_step, image_size, cp_path):
    print(_class_)
    epochs = epochs
    learning_rate = 0.005
    batch_size = batch_size
    image_size = image_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    resize_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train),
    ])


    train_data = MNIST_Dataset(train=True, transform=resize_transform)

    train_data = AugMixDatasetMVTec(train_data, preprocess)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    if backbone == 'wide':
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False)
    else:
        encoder, bn = resnet18(pretrained=True)
        decoder = de_resnet18(pretrained=False)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))

    for epoch in (range(epochs)):
        print(f'class: {_class_}, epoch: {epoch + 1}')
        bn.train()
        decoder.train()
        loss_list = []
        for normal, augmix_img, gray_img in train_dataloader:
            normal = normal.to(device)
            inputs_normal = encoder(normal)
            bn_normal = bn(inputs_normal)
            outputs_normal = decoder(bn_normal)

            augmix_img = augmix_img.to(device)
            inputs_augmix = encoder(augmix_img)
            bn_augmix = bn(inputs_augmix)
            outputs_augmix = decoder(bn_augmix)

            gray_img = gray_img.to(device)
            inputs_gray = encoder(gray_img)
            bn_gray = bn(inputs_gray)

            loss_bn = loss_fucntion([bn_normal], [bn_augmix]) + loss_fucntion([bn_normal], [bn_gray])
            outputs_gray = decoder(bn_gray)

            loss_last = loss_fucntion_last(outputs_normal, outputs_augmix) + loss_fucntion_last(outputs_normal,
                                                                                                outputs_gray)

            loss_normal = loss_fucntion(inputs_normal, outputs_normal)
            loss = loss_normal * 0.9 + loss_bn * 0.05 + loss_last * 0.05

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if (epoch + 1) % save_step == 0:
            ckp_path = cp_path + str(_class_) + '_' + str(epoch) + '.pth'
            Path(cp_path).mkdir(exist_ok=True)

            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)

            arguments_to_pass = [
                '--epochs', str(epoch + 1),
                '--image_size', str(image_size),
                '--backbone', str(backbone),
                '--cp_path', str(ckp_path),
                '--clazz', _class_
            ]

            result = subprocess.run(["python", "eval_mnist.py"] + arguments_to_pass, capture_output=False, text=True)

            # Check the result
            if result.returncode == 0:
                print("Script executed successfully.")
                print("Output:")
                print(result.stdout)
            else:
                print("Script execution failed.")
                print("Error:")
                print(result.stderr)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_step', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--backbone', type=str, choices=['wide', 'res18'], default='wide')
    parser.add_argument('--cp_path', type=str, default='./checkpoints/')

    args = parser.parse_args()

    train('mn', args.backbone, args.batch_size, args.epochs, args.save_step, args.image_size, args.cp_path)
