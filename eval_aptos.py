import argparse
from glob import glob
import pandas as pd
import torch
from dataset import get_data_transforms, BrainTest, APTOS
from resnet_TTA import wide_resnet50_2, resnet18
from de_resnet import de_wide_resnet50_2, de_resnet18
from test import evaluation_ATTA


def test_mvtec(_class_, epoch, backbone, image_size, cp_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Class: ', _class_)
    data_transform, gt_transform = get_data_transforms(image_size, image_size)

    test_anomaly_path = glob('/kaggle/input/aptos-dataset/APTOS/test/ABNORMAL/*')
    test_anomaly_label = [1] * len(test_anomaly_path)
    test_normal_path = glob('/kaggle/input/aptos-dataset/APTOS/test/NORMAL/*')
    test_normal_label = [0] * len(test_normal_path)

    test_label = test_anomaly_label + test_normal_label
    test_path = test_anomaly_path + test_normal_path

    df = pd.read_csv('/kaggle/input/ddrdataset/DR_grading.csv')
    label = df["diagnosis"].to_numpy()
    path = df["id_code"].to_numpy()

    normal_path = path[label == 0]
    anomaly_path = path[label != 0]

    shifted_test_path = list(normal_path) + list(anomaly_path)
    shifted_test_label = [0] * len(normal_path) + [1] * len(anomaly_path)

    shifted_test_path = ["/kaggle/input/ddrdataset/DR_grading/DR_grading/" + s for s in shifted_test_path]

    # load data
    test_data1 = APTOS(image_path=test_path, labels=test_label, transform=data_transform, train=False)
    test_data2 = APTOS(image_path=shifted_test_path, labels=shifted_test_label, transform=data_transform, train=False)

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
                               img_size=image_size, lamda=lamda, dataset_name='aptos', _class_=_class_)
    list_results.append(round(auroc_sp, 4))
    print('main Auroc{:.4f}'.format(auroc_sp))

    auroc_sp = evaluation_ATTA(encoder, bn, decoder, test_dataloader2, device,
                               type_of_test='EFDM_test',
                               img_size=image_size, lamda=lamda, dataset_name='aptos', _class_=_class_)
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
