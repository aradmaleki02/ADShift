import pickle
import shutil
from pathlib import Path

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random
from torch.utils.data import Dataset

IMAGE_SIZE = 256
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_data_transforms_augmix(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.AugMix(severity=10, all_ops=False),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type


class MVTecDatasetOOD(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, _class_):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join('/home/cttri/anomaly/data/mvtec/' + _class_, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type


class MNIST_M(Dataset):
    def __init__(self, path_to_image_tests, labs_test, transform):
        self.full_filenames = path_to_image_tests
        self.labels = labs_test
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


class PACSDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.img_path = root

        self.transform = transform
        # load dataset
        self.img_paths = glob.glob(self.img_path + '/*.png')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, 0


def load_data(dataset_name='mnist', normal_class=0, batch_size='16'):
    if dataset_name == 'cifar10':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            # transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
        dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
        print("Cifar10 DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
        test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'mnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/MNIST/train", exist_ok=True)
        dataset = MNIST('./Dataset/MNIST/train', train=True, download=True, transform=img_transform)
        print("MNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/MNIST/test", exist_ok=True)
        test_set = MNIST("./Dataset/MNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'fashionmnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/FashionMNIST/train", exist_ok=True)
        dataset = FashionMNIST('./Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
        print("FashionMNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/FashionMNIST/test", exist_ok=True)
        test_set = FashionMNIST("./Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)


    elif dataset_name == 'retina':
        data_path = 'Dataset/OCT2017/train'

        orig_transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(root=data_path, transform=orig_transform)

        test_data_path = 'Dataset/OCT2017/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    else:
        raise Exception(
            "You enter {} as dataset, which is not a valid dataset for this repository!".format(dataset_name))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.Resampling.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.Transform.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.Resampling.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.Transform.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.Resampling.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.Transform.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.Resampling.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.Transform.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.Resampling.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def augmvtec(image, preprocess, severity=3, width=3, depth=-1, alpha=1.):
    aug_list = [
        autocontrast, equalize, posterize, solarize, color, sharpness
    ]

    severity = random.randint(0, severity)

    ws = np.float32(np.random.dirichlet([1] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    preprocess_img = preprocess(image)
    mix = torch.zeros_like(preprocess_img)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess_img + m * mix
    return mixed


def augpacs(image, preprocess, severity=3, width=3, depth=-1, alpha=1.):
    aug_list = [
        autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness
    ]
    severity = random.randint(0, severity)

    ws = np.float32(np.random.dirichlet([1] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    preprocess_img = preprocess(image)
    mix = torch.zeros_like(preprocess_img)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess_img + m * mix
    return mixed


class AugMixDatasetMVTec(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess
        self.gray_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                 std=std_train),
            transforms.Grayscale(3)
        ])

    def __getitem__(self, i):
        x, _ = self.dataset[i]
        return self.preprocess(x), augmvtec(x, self.preprocess), self.gray_preprocess(x)

    def __len__(self):
        return len(self.dataset)


class AugMixDatasetPACS(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess
        self.gray_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                 std=std_train),
            transforms.Grayscale(3)
        ])

    def __getitem__(self, i):
        x, _ = self.dataset[i]
        return self.preprocess(x), augpacs(x, self.preprocess), self.gray_preprocess(x)

    def __len__(self):
        return len(self.dataset)


class BrainTrain(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.image_paths = glob.glob('./Br35H/dataset/train/normal/*')
        brats_mod = glob.glob('./brats/dataset/train/normal/*')
        random.seed(1)
        random_brats_images = random.sample(brats_mod, 50)
        self.image_paths.extend(random_brats_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, 0


class MNIST_Dataset(Dataset):
    def __init__(self, train, test_id=1, transform=None):
        self.transform = transform
        self.train = train
        self.test_id = test_id
        if train:
            with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/train_normal.pkl',
                      'rb') as f:
                normal_train = pickle.load(f)
            self.images = normal_train['images']
            self.labels = [0] * len(self.images)
        else:
            if test_id == 1:
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_normal_main.pkl',
                          'rb') as f:
                    normal_test = pickle.load(f)
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_abnormal_main.pkl',
                          'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])
            else:
                with open('/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_normal_shifted.pkl',
                          'rb') as f:
                    normal_test = pickle.load(f)
                with open(
                        '/kaggle/input/diagvib-6-mnist-dataset/content/mnist_shifted_dataset/test_abnormal_shifted.pkl',
                        'rb') as f:
                    abnormal_test = pickle.load(f)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])

    def __getitem__(self, index):
        image = torch.tensor(self.images[index])

        if self.transform is not None:
            image = self.transform(image)

        height = image.shape[1]
        width = image.shape[2]
        target = 0 if self.train else self.labels[index]

        return image, target

    def __len__(self):
        return len(self.images)



class BrainTest(torch.utils.data.Dataset):
    def __init__(self, transform, test_id=1):

        self.transform = transform
        self.test_id = test_id

        test_normal_path = glob.glob('./Br35H/dataset/test/normal/*')
        test_anomaly_path = glob.glob('./Br35H/dataset/test/anomaly/*')

        self.test_path = test_normal_path + test_anomaly_path
        self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

        if self.test_id == 2:
            test_normal_path = glob.glob('./brats/dataset/test/normal/*')
            test_anomaly_path = glob.glob('./brats/dataset/test/anomaly/*')

            self.test_path = test_normal_path + test_anomaly_path
            self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    def __len__(self):
        return len(self.test_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.test_path[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        has_anomaly = 0 if self.test_label[idx] == 0 else 1

        # return img, , has_anomaly, img_path
        return img, 1, has_anomaly, img_path


def prepare_br35h_dataset_files():
    normal_path35 = '/kaggle/input/brain-tumor-detection/no'
    anomaly_path35 = '/kaggle/input/brain-tumor-detection/yes'

    print(f"len(os.listdir(normal_path35)): {len(os.listdir(normal_path35))}")
    print(f"len(os.listdir(anomaly_path35)): {len(os.listdir(anomaly_path35))}")

    Path('./Br35H/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/anomaly', f))

    anom35 = os.listdir(anomaly_path35)
    for f in anom35:
        shutil.copy2(os.path.join(anomaly_path35, f), './Br35H/dataset/test/anomaly')


    normal35 = os.listdir(normal_path35)
    random.shuffle(normal35)
    ratio = 0.7
    sep = round(len(normal35) * ratio)

    Path('./Br35H/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('./Br35H/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/normal', f))

    flist = [f for f in os.listdir('./Br35H/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/train/normal', f))

    for f in normal35[:sep]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/train/normal')
    for f in normal35[sep:]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/test/normal')


import pandas as pd

def prepare_brats2015_dataset_files():
    labels = pd.read_csv('/kaggle/input/brain-tumor/Brain Tumor.csv')
    labels = labels[['Image', 'Class']]
    labels.tail() # 0: no tumor, 1: tumor

    labels.head()

    brats_path = '/kaggle/input/brain-tumor/Brain Tumor/Brain Tumor'
    lbl = dict(zip(labels.Image, labels.Class))

    keys = lbl.keys()
    normalbrats = [x for x in keys if lbl[x] == 0]
    anomalybrats = [x for x in keys if lbl[x] == 1]

    Path('./brats/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)
    Path('./brats/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('./brats/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./brats/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/test/anomaly', f))

    flist = [f for f in os.listdir('./brats/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/test/normal', f))

    flist = [f for f in os.listdir('./brats/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('./brats/dataset/train/normal', f))

    ratio = 0.7
    random.shuffle(normalbrats)
    bratsep = round(len(normalbrats) * ratio)

    for f in anomalybrats:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/test/anomaly')
    for f in normalbrats[:bratsep]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/train/normal')
    for f in normalbrats[bratsep:]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), './brats/dataset/test/normal')


class Waterbird(torch.utils.data.Dataset):
    def __init__(self, root, df, transform, train=True, count_train_landbg=-1, count_train_waterbg=-1, mode='bg_all'):
        self.transform = transform
        self.train = train
        self.df = df
        lb_on_l = df[(df['y'] == 0) & (df['place'] == 0)]
        lb_on_w = df[(df['y'] == 0) & (df['place'] == 1)]
        self.normal_paths = []
        self.labels = []

        normal_df = lb_on_l.iloc[:count_train_landbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_landbg])
        normal_df = lb_on_w.iloc[:count_train_waterbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_waterbg])

        if train:
            self.image_paths = self.normal_paths
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.train:
            return img, 0
        else:
            return img, 1, self.labels[idx], 0


class APTOS(Dataset):
    def __init__(self, image_path, labels, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        self.train = train
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

        if self.train:
            return image, 0
        return image, 0, self.labels[index], 0



    def __len__(self):
        return len(self.image_files)
