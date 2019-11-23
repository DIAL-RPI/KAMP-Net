# -*- coding: utf-8 -*-
"""
Author@ Hengtao Guo <https://github.com/Tonight1121>
This script trains dual-stream network, loading pretrained ResNet34/50 as building components
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, models, transforms
import torchvision.models.resnet as resnet
import time
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
                    default='origin-DSN')
parser.add_argument('-l', '--learning_rate',
                    type=float,
                    help='set learning rate',
                    default=5e-6)
parser.add_argument('-e', '--epoch_scratch',
                    type=int,
                    help='set the training epochs',
                    default=5)

args = parser.parse_args()

device_no = args.device_no
datafolder = args.data_folder
learning_rate = args.learning_rate
epoch_scratch = args.epoch_scratch

hostname = os.uname().nodename
jbhi_folder = '/zion/common/shared/JBHI'
on_arc = False
if 'arc' == hostname:
    on_arc = True
    device = torch.device("cuda:{}".format(device_no))
    jbhi_folder = '/zion/shared/JBHI'
    batch_size = 128
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32

this_folder = path.join(jbhi_folder, datafolder)

fn_roi_list = './data/roi_centers.txt'


''' train DSN using grey scale images '''
img_type = datafolder[:-4]
slice_folder = path.join(jbhi_folder, '{}-slice'.format(img_type))
patch_folder = path.join(jbhi_folder, '{}-patch'.format(img_type))

center_list = np.loadtxt(fn_roi_list, dtype=np.int)
roi_w = 160 // 2
roi_h = 160 // 2
roi_d = 3 // 2
start_fold = 0
end_fold = 10
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

        return image, patch, target, int(img_id)

######################################################################
# Training the model
# ------------------
def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()

    best_acc = 0.0
    best_ep = 0
    best_auc = 0.0

    tv_hist = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        test_scores = []
        test_labels = []
        test_ids = []
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, input_patch, labels, img_id in dataloaders[phase]:
                # Get images from inputs

                inputs = inputs.to(device)
                input_patch = input_patch.to(device)
                labels = labels.to(device)
                img_id = img_id.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, input_patch)
                    network_probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(network_probs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        scores = nn.functional.softmax(outputs, dim=1)
                        test_scores.extend(scores.data.cpu().numpy()[:, 1])
                        test_labels.extend(labels.data.cpu().numpy())
                        test_ids.extend(img_id.data.cpu().numpy())

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            tv_hist[phase].append([epoch_loss, epoch_acc])

            ''' Calculate this round of AUC score '''
            epoch_auc = 0.0
            if phase == 'val' and len(test_scores) != 0:
                fpr, tpr, _ = roc_curve(test_labels, test_scores)
                epoch_auc = auc(fpr, tpr)
                if epoch_auc < 0.5:
                    test_scores = np.asarray(test_scores)
                    test_scores = np.ones_like(test_scores) - test_scores
                    test_scores = test_scores.tolist()
                    fpr, tpr, _ = roc_curve(test_labels, test_scores)
                    epoch_auc = auc(fpr, tpr)
                print('roc_auc {:.4f}'.format(epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_auc >= best_auc:
                best_auc = epoch_auc
                best_acc = epoch_acc
                best_ep = epoch
                torch.save(model.state_dict(), fn_save)
                print('**** best model updated with auc={:.4f} ****'.format(epoch_auc))

                test_ids = np.asarray(test_ids).reshape((len(test_ids), 1))
                test_scores = np.asarray(test_scores).reshape((len(test_scores), 1))
                test_labels = np.asarray(test_labels).reshape((len(test_labels), 1))
                val_data = np.concatenate((test_ids, test_scores, test_labels), axis=1)
                val_data_path = path.join(data_dir, 'val_data.txt')
                np.savetxt(val_data_path, val_data)
                print('val_data shape {} saved'.format(val_data.shape))

        print('ep {}/{} - Train loss: {:.4f} acc: {:.4f}, Val loss: {:.4f} acc: {:.4f}'.format(
            epoch + 1, num_epochs, 
            tv_hist['train'][-1][0], tv_hist['train'][-1][1],
            tv_hist['val'][-1][0], tv_hist['val'][-1][1]))
        training_progress[epoch][0] = tv_hist['train'][-1][0]
        training_progress[epoch][1] = tv_hist['train'][-1][1]
        training_progress[epoch][2] = tv_hist['val'][-1][0]
        training_progress[epoch][3] = tv_hist['val'][-1][1]
        txt_path = path.join(data_dir, 'training_progress_multi_context.txt')
        np.savetxt(txt_path, training_progress)

    time_elapsed = time.time() - since
    print('*'*10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*'*10 + 'Best val Acc: {:4f} at epoch {}'.format(best_acc, best_ep))
    print()
    auc_diary_path = path.join(jbhi_folder, '{}/auc_diary_multi_context.txt'.format(datafolder))
    aucs = np.loadtxt(auc_diary_path)
    aucs[k] = best_auc
    np.savetxt(auc_diary_path, aucs)
    return tv_hist


# %% 10-fold cross validation
k_tot = 10
for k in range(k_tot):
    print('Cross validating fold {}/{} of multi_context'.format(k+1, k_tot))
    data_dir = path.join(jbhi_folder, '{}/fold_{}'.format(datafolder, k))
    slice_dir = path.join(slice_folder, 'fold_{}'.format(k))
    patch_dir = path.join(patch_folder, 'fold_{}'.format(k))
    image_datasets = {x: MortalityRiskDataset(os.path.join(data_dir, x), True,
                                              data_transforms_slice[x],
                                              data_transforms_patch[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    print('size of dataloader: {}'.format(dataloaders.__sizeof__()))
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

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

    slice_model_path = os.path.join(slice_dir, 'best_scratch_ResNet-50.pth')
    patch_model_path = os.path.join(patch_dir, 'best_scratch_ResNet-34.pth')

    '''Load the pretrained streams!'''
    model_slice.load_state_dict(torch.load(slice_model_path, map_location='cuda:0'))
    model_patch.load_state_dict(torch.load(patch_model_path, map_location='cuda:0'))
    '''Loading ends here'''

    model_all = networks.MultiContext(model_slice, model_patch)

    model_all.cuda()
    model_all.fc = nn.Linear(num_ftrs_patch + num_ftrs_slice, 2)
    model_all = model_all.to(device)

    all_model_path = os.path.join(data_dir, 'best_multi_context.pth')
    model_all.load_state_dict(torch.load(all_model_path, map_location='cuda:0'))

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_all.parameters()), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)

    # Train and evaluate
    fn_best_model = os.path.join(data_dir, 'best_multi_context.pth')
    hist_ft = train_model(model_all, criterion, optimizer_ft, exp_lr_scheduler,
                          fn_best_model, num_epochs=epoch_scratch)
    fn_hist = os.path.join(data_dir, 'hist_multi_context.npy')
    np.save(fn_hist, hist_ft)
    txt_path = path.join(data_dir, 'training_progress_multi_context.txt')
    np.savetxt(txt_path, training_progress)
