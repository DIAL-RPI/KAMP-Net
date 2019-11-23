"""
Author@ Hengtao Guo <https://github.com/Tonight1121>
This script tests the KAMP-Net
The output of dual-stream network is combined with SVM probability based on network_prob_ratio
SVM is trained independently using 4 clinical measurements
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, models, transforms
import torchvision.models.resnet as resnet
import os
from os import path
import networks
from sklearn.metrics import roc_curve, auc
from PIL import Image
import argparse

################

desc = 'Training registration generator'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-d', '--device_no',
                    type=int,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='GPU device number [0-7]',
                    default=2)
parser.add_argument('-f', '--data_folder',
                    type=str,
                    choices=['color-DSN', 'origin-DSN'],
                    help='choose the data folder',
                    default='color-DSN')
parser.add_argument('-l', '--learning_rate',
                    type=float,
                    help='set learning rate',
                    default=5e-7)
parser.add_argument('-e', '--epoch_scratch',
                    type=int,
                    help='set the training epochs',
                    default=10)

args = parser.parse_args()

device_no = args.device_no
datafolder = args.data_folder
learning_rate = args.learning_rate
epoch_scratch = args.epoch_scratch

hostname = os.uname().nodename
data_path = '/zion/common/shared/KAMP'
on_arc = False
if 'arc' == hostname:
    on_arc = True
    device = torch.device("cuda:{}".format(device_no))
    data_path = '/zion/shared/KAMP'
    batch_size = 128
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1

fn_roi_list = './data/roi_centers.txt'

''' train DSN using grey scale images '''
img_type = datafolder[:-4]
slice_folder = path.join(data_path, '{}-slice'.format(img_type))
patch_folder = path.join(data_path, '{}-patch'.format(img_type))

center_list = np.loadtxt(fn_roi_list, dtype=np.int)
roi_w = 160 // 2
roi_h = 160 // 2
roi_d = 3 // 2
network_prob_ratio = 0.5
training_progress = np.zeros((epoch_scratch, 4))


smean_std_path = path.join(slice_folder, 'img_mean_std.txt')
smean_std = np.loadtxt(smean_std_path)
svec_mean = [smean_std[0]]
svec_std = [smean_std[1]]

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms_slice = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(svec_mean, svec_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(svec_mean, svec_std)
    ]),
}

pmean_std_path = path.join(patch_folder, 'img_mean_std.txt')
pmean_std = np.loadtxt(pmean_std_path)
pvec_mean = [pmean_std[0]]
pvec_std = [pmean_std[1]]

data_transforms_patch = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(pvec_mean, pvec_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(pvec_mean, pvec_std)
    ]),
}


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

class MortalityRiskDataset(Dataset):

    def __init__(self, root_dir, transform=None, transform_slice=None, transform_patch=None):
        """
        """
        classes, class_to_idx = find_classes(root_dir)
        samples = make_dataset(root_dir, class_to_idx, IMG_EXTENSIONS)

        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.transform_slice = transform_slice
        self.transform_patch = transform_patch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img_path, target = self.samples[idx]

        # find last .png
        # extract the code right before it
        norm_path = path.normpath(img_path)
        res = norm_path.split(os.sep)
        img_name = res[-1]
        img_id = img_name[:6]

        idx = np.where(center_list[:, 0] == int(img_id))[0][0]
        center = center_list[idx, 1:]

        with open(img_path, 'rb') as f:
            img = Image.open(f)
            image = img.convert('RGB')

            xl = center[0] - roi_w - 1
            xu = center[0] + roi_w
            yl = center[1] - roi_h - 1
            yu = center[1] + roi_h
            patch = image.crop((xl, yl, xu, yu))

        if self.transform:
            image = self.transform_slice(image)
            patch = self.transform_patch(patch)

        return image, patch, target, img_id

def test_model(model):
    ''' Test the trained models '''
    test_scores = []
    test_labels = []
    running_corrects = 0
    svm_results = np.loadtxt('data/probs/id_probs{}.txt'.format(k))

    # Iterate over data.
    file = open('data/calc_sen_spe/sen-spe{}.txt'.format(network_prob_ratio), 'a')
    file2 = open('data/probs&label.txt', 'a')
    for inputs, input_patch, labels, img_id in dataloader:
        inputs = inputs.to(device)
        input_patch = input_patch.to(device)
        labels = labels.to(device)

        outputs = model(inputs, input_patch)

        ''' Combine the probability of Network and SVM to make prediction '''
        network_probs = torch.nn.functional.softmax(outputs, dim=1).data.cpu().numpy()
        network_probs = np.asarray(network_probs)
        id_numpy = np.asarray(img_id).astype(int)
        id_numpy = np.reshape(id_numpy, (id_numpy.shape[0], 1))
        index = np.nonzero(id_numpy == svm_results[:, 0])[1]
        svm_probs = np.take(svm_results, index, axis=0)[:, 1:3]
        overall_probs = network_probs * network_prob_ratio + svm_probs * (1 - network_prob_ratio)
        overall_probs = torch.tensor(overall_probs).to(device)
        _, preds = torch.max(overall_probs, 1)

        ''' Save the label and preds to calculate sensitivity and specificity '''
        preds_np = preds.data.cpu().numpy()
        label_np = labels.data.cpu().numpy()
        preds_np = np.reshape(preds_np, (preds_np.shape[0], 1))
        label_np = np.reshape(label_np, (label_np.shape[0], 1))
        results_np = np.concatenate((label_np, preds_np), axis=1)
        np.savetxt(file, results_np)

        probs_label = np.concatenate((network_probs, svm_probs), axis=1)
        probs_label = np.concatenate((probs_label, label_np), axis=1)
        np.savetxt(file2, probs_label)

        test_scores.extend(overall_probs.data.cpu().numpy()[:, 1])
        test_labels.extend(labels.data.cpu().numpy())

        running_corrects += torch.sum(preds == labels.data)
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    epoch_auc = auc(fpr, tpr)

    print('{}/10: test AUC = {:.4f}'.format(k+1, epoch_auc))
    return test_scores, test_labels, epoch_auc

# %% 10-fold cross validation
k_tot = 10
for k in range(k_tot):
    print('Cross validating fold {}/{} of KAMP'.format(k+1, k_tot))
    data_dir = path.join(data_path, '{}/fold_{}'.format(datafolder, k))
    image_datasets = {x: MortalityRiskDataset(os.path.join(data_dir, x), True,
                                              data_transforms_slice[x],
                                              data_transforms_patch[x])
                      for x in ['val']}

    dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=0)
                   for x in ['val']}

    print('size of dataloader: {}'.format(dataloader.__sizeof__()))
    dataset_size = {x: len(image_datasets[x]) for x in ['val']}

    model_slice = networks.resnet50(pretrained=False)
    model_slice.cuda()
    num_ftrs_slice = model_slice.fc.in_features
    model_slice.fc = nn.Linear(num_ftrs_slice, 2)
    model_slice = model_slice.to(device)

    model_patch = networks.resnet34(pretrained=False)
    model_patch.cuda()
    num_ftrs_patch = model_patch.fc.in_features
    model_patch.fc = nn.Linear(num_ftrs_patch, 2)
    model_patch = model_patch.to(device)

    model_all = networks.MultiContext(model_slice, model_patch)
    model_all.cuda()
    model_all.fc = nn.Linear(num_ftrs_patch + num_ftrs_slice, 2)
    model_all = model_all.to(device)

    all_model_path = os.path.join(data_dir, 'best_multi_context.pth')
    model_all.load_state_dict(torch.load(all_model_path, map_location='cuda:0'))

    test_model(model_all)
