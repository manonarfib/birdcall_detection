from __future__ import print_function, division


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

import keyboard
from src.utils import ImprovedPANNsLoss, find_classes
from src.models import get_optimizer, get_scheduler, AttBlock, models

# Utils


def transform_PIL_Array(image_PIL):
    image = np.array(image_PIL)
    image = np.swapaxes(image, 0, 2)
    image = torch.from_numpy(image/255.0)
    image = image.float()
    return image


def transform_labels(labels, num_classes):
    siz = labels.size()
    new_labels = torch.zeros((siz[0], num_classes))
    for i, label in enumerate(labels):
        new_labels[i, label] = 1
    return new_labels


__keep_running__ = True


def stop_running():
    "appuyer sur '$' termine la derniere epoch puis fini l'entrainement"
    global __keep_running__
    __keep_running__ = False


keyboard.add_hotkey('$', stop_running)


def train_model(model, device, criterion, optimizer, scheduler, model_name, dataloaders, dataset_sizes, class_names, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = -np.inf

    for epoch in range(num_epochs):
        if not __keep_running__:
            continue
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                labels = transform_labels(labels, len(class_names))
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if model_name == "eff_th04":
                        outputs["segmentwise_output"], _ = outputs["segmentwise_output"].max(
                            dim=1)
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                ###
                result = outputs['framewise_output']
                result, _ = torch.max(result, dim=1)
                # least squares
                running_corrects -= torch.sum((result-labels)**2)
                ###
                del inputs, outputs, loss, result
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            del epoch_acc, epoch_loss, running_corrects, running_loss
        print()

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train(models, device, key_model=None, num_epochs=25, lr=0.001, batch_size=2):

    image_datasets = {x: datasets.ImageFolder('train/temp/' + x, transform=transform_PIL_Array)
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    if not key_model:
        key_model = models.keys()
    for model in key_model:
        print(model)
        model_conv = models[model]

        for param in model_conv.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc1.in_features
        model_conv.att_block = AttBlock(
            num_ftrs, len(class_names), activation="sigmoid")
        model_conv = model_conv.to(device)
        print()

        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        if model != "eff_th04":
            criterion = ImprovedPANNsLoss()
        else:
            criterion = ImprovedPANNsLoss('segmentwise_output')

        optimizer_conv = get_optimizer(
            model_conv, {"optimizer": {'name': 'Adam', 'params': {'lr': lr}}})
        exp_lr_scheduler = get_scheduler(optimizer_conv, {'scheduler': {
                                         'name': 'CosineAnnealingLR', 'params': {'T_max': 10}}})

        print('ok')
        model_conv = train_model(model_conv, device, criterion, optimizer_conv,
                                 exp_lr_scheduler, model, dataloaders, dataset_sizes, class_names, num_epochs=num_epochs)
        torch.save(model_conv.state_dict(), 'weights_trained/'+model+'.pth')
        torch.cuda.empty_cache()
        del model_conv
        torch.cuda.empty_cache()
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help="""Chose if you want to train model "1", "2", "3", "4" or "all" """)
    args = parser.parse_args()
    keys = args.model
    if keys == 'all':
        keys = ["ref2_th03", "ref2_th04",
                "eff_th04", "ext"]
    elif keys in ['1', '2', '3', '4']:
        keys = [["ref2_th03", "ref2_th04",
                 "eff_th04", "ext"][int(keys)-1]]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    cudnn.benchmark = False
    # batch 20 et 15 pour eff_th04 (8GB VRAM) # if you don't have enough ram you'd better launch one by one
    train(models, device, keys, num_epochs=20, batch_size=15)

    classes, _ = find_classes('train/temp/train/')
    np.save('inv_bird_code.npy', classes)
