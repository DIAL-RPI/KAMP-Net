# -*- coding: utf-8 -*-
"""
Author@ Hengtao Guo <https://github.com/Tonight1121>
This script trains ResNet34/50 for classifying NLST patches/slices
Preprocessings, including color-coding, are finished in advance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
import torchvision.models.resnet as resnet
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from os import path
from PIL import Image
from sklearn.metrics import roc_curve, auc
import tools


# datafolder = '181031_multifc'
# datafolder = '181107_avgtest'
# datafolder = '181114_slice'
# datafolder = '181007_50N_A'
# datafolder = '181127_interest'
# datafolder = '190104_multicontext'
# datafolder = '01slice'
# datafolder = '01slice_origin'
# datafolder = '01slice_unresized'
datafolder = '02patch'
# datafolder = '02patch_origin'
# datafolder = '01slice_clinics'
# datafolder = '02autopatch_grey'
# datafolder = '02patch_origin_r3'
# datafolder = 'nlst_manual'


roi_w = 160 // 2
roi_h = 160 // 2
roi_d = 3 // 2
training_part = 'patch'
epoch_scratch = 300
use_last_pretrained = False
training_progress = np.zeros((epoch_scratch, 5))

# network = ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152']
network = ['ResNet-34']  # if training slice, use RestNet-50 instead

mean_std = np.loadtxt('img_mean_std.txt')
vec_mean = [mean_std[0]]
vec_std = [mean_std[1]]

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        # transforms.Resize(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(vec_mean, vec_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.Resize(320),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(vec_mean, vec_std)
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    def __init__(self, root_dir, transform=None):
        """
        """
        classes, class_to_idx = find_classes(root_dir)
        samples = make_dataset(root_dir, class_to_idx, IMG_EXTENSIONS)

        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img_path, target = self.samples[idx]

        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image = np.asarray(image)
            patch = tools.cascade_detect(image)

            image = Image.fromarray(np.uint8(image))
            patch = Image.fromarray(np.uint8(patch))
            image = image.convert('RGB')
            patch = patch.convert('RGB')

        if self.transform:
            image = self.transform(image)
            patch = self.transform(patch)

        return image, patch, target

######################################################################
# Training the model
# ------------------

def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()

    best_acc = 0.0
    best_ep = 0
    best_auc = 0.0
    test_scores = []
    test_labels = []

    tv_hist = {'train': [], 'val': []}

    for epoch in range(num_epochs):
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
            for inputs, input_patch, labels in dataloaders[phase]:


                inputs = inputs.to(device)
                input_patch = input_patch.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if training_part == 'slice':
                        outputs = model(inputs)
                    else:
                        outputs = model(input_patch)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        scores = nn.functional.softmax(outputs, dim=1)
                        test_scores.extend(scores.data.cpu().numpy()[:, 1])
                        test_labels.extend(labels.data.cpu().numpy())

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

        print('ep {}/{} - Train loss: {:.4f} acc: {:.4f}, Val loss: {:.4f} acc: {:.4f}'.format(
            epoch + 1, num_epochs, 
            tv_hist['train'][-1][0], tv_hist['train'][-1][1],
            tv_hist['val'][-1][0], tv_hist['val'][-1][1]))
        training_progress[epoch][0] = tv_hist['train'][-1][0]
        training_progress[epoch][1] = tv_hist['train'][-1][1]
        training_progress[epoch][2] = tv_hist['val'][-1][0]
        training_progress[epoch][3] = tv_hist['val'][-1][1]
        training_progress[epoch][4] = epoch_auc
        np.savetxt(txt_path, training_progress)

    time_elapsed = time.time() - since
    print('*'*10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*'*10 + 'Best val Acc: {:4f} at epoch {}'.format(best_acc, best_ep))
    return tv_hist

# %% 10-fold cross validation
k_tot = 10

for net in network:
    if net == 'ResNet-18':
        base_model = resnet.resnet18
    elif net == 'ResNet-34':
        base_model = resnet.resnet34
    elif net == 'ResNet-50':
        base_model = resnet.resnet50
    elif net == 'ResNet-101':
        base_model = resnet.resnet101
    elif net == 'ResNet-152':
        base_model = resnet.resnet152
    else:
        print('The network of {} is not supported!'.format(net))

    for k in range(k_tot):
        print('Cross validating fold {}/{} of {}'.format(k+1, k_tot, net))
        data_dir = path.expanduser('/zion/guoh9/JBHI-revision/{}/fold_{}'.format(datafolder, k))
        image_datasets = {x: MortalityRiskDataset(os.path.join(data_dir, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                    shuffle=True, num_workers=0)
                    for x in ['train', 'val']}

        print('size of dataloader: {}'.format(dataloaders.__sizeof__()))
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes


        ######################################################################
        # Train from scratch

        model_ft = base_model(pretrained=False)
        model_ft.cuda()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft = model_ft.to(device)
        if use_last_pretrained:
            fn_best_model = os.path.join(data_dir, 'best_scratch_{}.pth'.format(net))
            model_ft.load_state_dict(torch.load(fn_best_model))
            model_ft.eval()
            print('loaded last time\'s network!')
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-5)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)

        # Train and evaluate
        fn_best_model = os.path.join(data_dir, 'best_scratch_{}.pth'.format(net))
        txt_path = path.join(data_dir, '{}_training_progress_{}.txt'.format(training_part, net))
        hist_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                              fn_best_model, num_epochs=epoch_scratch)
        fn_hist = os.path.join(data_dir, 'hist_scratch_{}.npy'.format(net))
        np.save(fn_hist, hist_ft)
