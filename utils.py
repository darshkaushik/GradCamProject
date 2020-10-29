import torch

import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import time
from datetime import datetime


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.6373545, 0.44605875, 0.46191868])
    std = np.array([0.27236816, 0.22500427, 0.24329403])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def set_parameter_requires_grad(model, num_freeze):
    for i, param in enumerate(model.parameters()):
        if i >= num_freeze:
            break
        param.requires_grad = False


def train_model(model, dataloaders, criterion, optimizer, scheduler, start_epoch=0, num_epochs=25, best_acc=0.0, hist={'val_acc': False, 'train_acc': False}):
    since = time.time()

    val_acc_history = hist['val_acc'] or []
    train_acc_history = hist['train_acc'] or []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = best_acc

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # print('beginning of forwardprop')
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, torch.max(labels, 1)[1])

                    # print('beginning of backprop')
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds ==
                                              torch.max(labels, 1)[1].data)
            if phase == 'val':
                # print('LR Decreased')
                print('LR', optimizer.param_groups[0]['lr'])
                # scheduler.step(loss)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Saving as best model')
                torch.save(model.state_dict(),
                           best_model_path)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                if epoch % 4 == 0:
                    print("Saving Checkpoint")
                    print("Best Acc", best_acc.item())
                    torch.save({
                        "epoch": epoch,
                        "loss": loss,
                        "model_state_dict": model.state_dict(),
                        "best_acc": best_acc,
                        "hist": {'val_acc': val_acc_history, 'train_acc': train_acc_history}
                    }, os.path.join(checkpoint_dir, 'Epoch={0:0=3d}.pt'.format(epoch)))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, {'val_acc': val_acc_history, 'train_acc': train_acc_history}


def evaluate(model, dataloader, criterion):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, torch.max(labels, 1)[1])

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == torch.max(labels, 1)[1].data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        time_elapsed = time.time() - since
        print('Inference Time {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    return model
