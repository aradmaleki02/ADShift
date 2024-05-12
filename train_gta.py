import subprocess
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from PIL import Image
from mvtec import MVTEC
from resnet import wide_resnet50_2, resnet18
from resnet_TTA import wide_resnet50_2 as wide_TTA
from resnet_TTA import resnet18 as res_TTA

from de_resnet import de_wide_resnet50_2, de_resnet18
from torch.nn import functional as F
import torchvision.transforms as transforms
from dataset import AugMixDatasetMVTec, get_data_transforms, prepare_br35h_dataset_files, \
    prepare_brats2015_dataset_files, BrainTrain
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


def get_cityscape_globs():
    from glob import glob
    import random
    normal_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/ID/*')
    anomaly_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/OOD/*')

    random.seed(42)
    random.shuffle(normal_path)
    train_ratio = 0.7
    separator = int(train_ratio * len(normal_path))
    normal_path_train = normal_path[:separator]
    normal_path_test = normal_path[separator:]

    return normal_path_train, normal_path_test, anomaly_path


def get_gta_globs():
    from glob import glob
    nums = [f'0{i}' for i in range(1, 10)] + ['10']
    globs_id = []
    globs_ood = []
    for i in range(10):
        id_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/ID/*'
        ood_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/OOD/*'
        globs_id.append(glob(id_path))
        globs_ood.append(glob(ood_path))
        print(i, len(globs_id[-1]), len(globs_ood[-1]))

    glob_id = []
    glob_ood = []
    for i in range(len(globs_id)):
        glob_id += globs_id[i]
        glob_ood += globs_ood[i]
    random.seed(42)
    random.shuffle(glob_id)
    train_ratio = 0.7
    separator = int(train_ratio * len(glob_id))
    glob_train_id = glob_id[:separator]
    glob_test_id = glob_id[separator:]

    return glob_train_id, glob_test_id, glob_ood


class GTA(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)



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

    normal_path_train, normal_path_test, anomaly_path = get_cityscape_globs()
    test_path = normal_path_test + anomaly_path
    test_label = [0] * len(normal_path_test) + [1] * len(anomaly_path)
    train_label = [0] * len(normal_path_train)
    glob_train_id, glob_test_id, glob_ood = get_gta_globs()

    train_data = GTA(image_path=normal_path_train, labels=train_label,
                     transform=resize_transform)

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

            result = subprocess.run(["python", "eval_gta.py"] + arguments_to_pass, capture_output=False, text=True)

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

    train('gta', args.backbone, args.batch_size, args.epochs, args.save_step, args.image_size, args.cp_path)
