import argparse

import torch
import random

from PIL import Image
from torch.utils.data import Dataset

from dataset import get_data_transforms, BrainTest
from resnet_TTA import wide_resnet50_2, resnet18
from de_resnet import de_wide_resnet50_2, de_resnet18
from test import evaluation_ATTA
from PIL import Image


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


class GTA_Test(Dataset):
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


def test_mvtec(_class_, epoch, backbone, image_size, cp_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Class: ', _class_)
    data_transform, gt_transform = get_data_transforms(image_size, image_size)

    normal_path_train, normal_path_test, anomaly_path = get_cityscape_globs()
    test_path = normal_path_test + anomaly_path
    test_label = [0] * len(normal_path_test) + [1] * len(anomaly_path)
    glob_train_id, glob_test_id, glob_ood = get_gta_globs()
    # load data
    test_data1 = GTA_Test(image_path=test_path, labels=test_label,
                          transform=data_transform)
    test_data2 = GTA_Test(image_path=glob_test_id + glob_ood, labels=[0] * len(glob_test_id) + [1] * len(glob_ood),
                          transform=data_transform)

    test_dataloader1 = torch.utils.data.DataLoader(test_data1, batch_size=1, shuffle=False)
    test_dataloader2 = torch.utils.data.DataLoader(test_data2, batch_size=1, shuffle=False)

    # load model
    if backbone == 'wide':
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False)
    elif backbone == 'res18':
        encoder, bn = resnet18(pretrained=True)
        decoder = de_resnet18(pretrained=False)
    else:
        raise NotImplementedError()

    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = decoder.to(device)

    # load checkpoint
    ckp = torch.load(cp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    lamda = 0.5

    list_results = []
    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader1, device,
                               type_of_test='EFDM_test',
                               img_size=image_size, lamda=lamda, dataset_name='gta', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('main Auroc{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader2, device,
                               type_of_test='EFDM_test',
                               img_size=image_size, lamda=lamda, dataset_name='gta', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('shifted Auroc{:.4f}'.format(auroc_sp))

    print(list_results)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--backbone', type=str, choices=['wide', 'res18'], default='wide')
    parser.add_argument('--cp_path', type=str, default='./checkpoints/')
    parser.add_argument('--clazz', type=str, default='carpet')

    args = parser.parse_args()
    test_mvtec(args.clazz, args.epochs, args.backbone, args.image_size, args.cp_path)
    print('===============================================')
    print('')
    print('')
