import torch, os
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import os.path as osp
from sklearn.neighbors import KNeighborsClassifier


class FocalLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(FocalLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.softmax(inputs)

        tmp = log_probs[range(log_probs.shape[0]), targets]  # batch
        if self.use_gpu:
            targets = targets.cuda()
        # targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-((1 - tmp) ** 2) * torch.log(tmp)).mean(0).sum()
        else:
            loss = (-((1 - tmp) ** 2) * torch.log(tmp)).sum(1)
        return loss


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).cpu(), 1
        )
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)
        return loss


def cal_acc_(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            output_f = netF.forward(inputs)  # a^t
            outputs = netC(output_f)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent


"""def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor(), normalize
    ])"""


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )


def image_target(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_shift(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


"""def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(), normalize
    ])"""


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [
                (val.split()[0], np.array([int(la) for la in val.split()[1:]]))
                for val in image_list
            ]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def l_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("L")


class ImageList(Dataset):
    def __init__(
        self, image_list, labels=None, transform=None, target_transform=None, mode="RGB",  root='/scratch/lg154/sseg/dataset'
    ):
        imgs = make_dataset(image_list, labels)

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def office_load(args):
    train_bs = args.batch_size
    if args.office31 == True:  # and not args.home and not args.visda:
        ss = args.dset.split("2")[0]
        tt = args.dset.split("2")[1]
        if ss == "a":
            s = "amazon"
        elif ss == "d":
            s = "dslr"
        elif ss == "w":
            s = "webcam"

        if tt == "a":
            t = "amazon"
        elif tt == "d":
            t = "dslr"
        elif tt == "w":
            t = "webcam"

        s_tr, s_ts = "./data/office/{}_list.txt".format(
            s
        ), "./data/office/{}_list.txt".format(s)

        txt_src = open(s_tr).readlines()
        dsize = len(txt_src)
        """tv_size = int(1.0 * dsize)
        print(dsize, tv_size, dsize - tv_size)
        s_tr, s_ts = torch.utils.data.random_split(txt_src,
                                                   [tv_size, dsize - tv_size])"""
        s_tr = txt_src
        s_ts = txt_src

        t_tr, t_ts = "./data/office/{}_list.txt".format(
            t
        ), "./data/office/{}_list.txt".format(t)
        prep_dict = {}
        prep_dict["source"] = image_train()
        prep_dict["target"] = image_target()
        prep_dict["test"] = image_test()
        train_source = ImageList(s_tr, transform=prep_dict["source"], root='/scratch/lg154/sseg/dataset')
        test_source = ImageList(s_tr, transform=prep_dict["source"],  root='/scratch/lg154/sseg/dataset')
        train_target = ImageList(open(t_tr).readlines(), transform=prep_dict["target"],  root='/scratch/lg154/sseg/dataset')
        test_target = ImageList(open(t_ts).readlines(), transform=prep_dict["test"],  root='/scratch/lg154/sseg/dataset')
    if args.home == True:
        ss = args.dset.split("2")[0]
        tt = args.dset.split("2")[1]
        if ss == "a":
            s = "Art"
        elif ss == "c":
            s = "Clipart"
        elif ss == "p":
            s = "Product"
        elif ss == "r":
            s = "Real_World"

        if tt == "a":
            t = "Art"
        elif tt == "c":
            t = "Clipart"
        elif tt == "p":
            t = "Product"
        elif tt == "r":
            t = "Real_World"

        s_tr, s_ts = "./data/office-home/{}.txt".format(
            s
        ), "./data/office-home/{}.txt".format(s)

        txt_src = open(s_tr).readlines()
        s_tr = txt_src
        s_ts = txt_src

        t_tr, t_ts = "./data/office-home/{}.txt".format(
            t
        ), "./data/office-home/{}.txt".format(t)
        prep_dict = {}
        prep_dict["source"] = image_train()
        prep_dict["target"] = image_target()
        prep_dict["test"] = image_test()
        train_source = ImageList(s_tr, transform=prep_dict["source"],  root='/scratch/lg154/sseg/dataset')
        test_source = ImageList(s_ts, transform=prep_dict["source"],  root='/scratch/lg154/sseg/dataset')
        train_target = ImageList(open(t_tr).readlines(), transform=prep_dict["target"],  root='/scratch/lg154/sseg/dataset')
        test_target = ImageList(open(t_ts).readlines(), transform=prep_dict["test"], root='/scratch/lg154/sseg/dataset')

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(
        train_source,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["source_te"] = DataLoader(
        test_source,
        batch_size=train_bs * 2,  # 2
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["target"] = DataLoader(
        train_target,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["test"] = DataLoader(
        test_target,
        batch_size=train_bs * 3,  # 3
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    return dset_loaders
