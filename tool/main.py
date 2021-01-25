import os
import torch
import mlflow

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from data.loader import ImageDataLoader
from data.samplers import randomSequentialStratifiedSampler, randomSequentialSampler
from data import transforms
from model.mobilenetv2_classifier import SampleClassificationNetwork, init_weights

from util.configurator import get_default_config

from albumentations import (
    ShiftScaleRotate,
    Blur,
    CoarseDropout,
    MedianBlur,
    ImageCompression,
    Compose,
    OneOf,
    MotionBlur,
    CLAHE,
    IAASharpen,
    RandomBrightnessContrast,
    ChannelShuffle,
    IAAPerspective,
    HueSaturationValue,
    RandomGamma,
    GaussNoise,
)

N_CLASSES = 5


def save_experiment_data(prefix, epoch_idx, train_loss, val_loss):
    root = os.path.join('../', 'experiments')
    filename = prefix + '.csv'
    path_to_save = os.path.join(root, filename)

    with open(path_to_save, 'a+') as f:
        f.write(f'{epoch_idx},{train_loss},{val_loss}\n')


def debug(data, target):
    print(target)
    plt.imshow(data.squeeze(), cmap='gray')
    plt.show()


def validate(model, criterion, val_loader, config):
    model.eval()

    with torch.no_grad():
        val_iter = iter(val_loader)
        loss_value = 0

        for i in tqdm(val_loader):
            data, target = val_iter.next()

            '''Handle residual batches'''
            if data.shape[0] != config.pipeline.batch_size:
                continue

            data = data.to('cuda')
            target = target.long().to('cuda')
            output = model(data)
            loss = criterion(output, target)

            loss_value += loss.item()

        return loss_value / len(val_loader)


def load_parameters_from_cpkt(model, optimizer, path_to_weights):
    checkpoint = torch.load(path_to_weights)
    model.load_state_dict(checkpoint['model_state_dict'])

    if checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    prev_loss = checkpoint['loss']

    return model, optimizer, start_epoch, prev_loss


def load_config(argp_cfg):
    config = get_default_config()
    config.merge_from_file(argp_cfg.config)
    config.freeze()

    return config


def main():
    args = ArgumentParser()
    args.add_argument("--config", type=str, required=True, help='Path to experiment config')
    argp = args.parse_args()

    config = load_config(argp)

    augs = {
        'CLAHE': dict(clip_limit=4, p=0.5),
        'IAASharpen': dict(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        'RandomBrightnessContrast': dict(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
        'ChannelShuffle': dict(p=0.5),
        'HueSaturationValue': dict(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        'RandomGamma': dict(gamma_limit=(80, 120), eps=None, p=0.5)
    }

    transforms_list = [
        Compose([
            ShiftScaleRotate(p=0.3, scale_limit=(-0.05, 0), border_mode=0, rotate_limit=15, shift_limit=0.05),
            IAAPerspective(scale=(0.01, 0.05), p=0.3),
            GaussNoise(var_limit=(100, 300), p=0.5),

            Compose([
                CLAHE(**augs['CLAHE']),
                IAASharpen(**augs['IAASharpen']),
                RandomBrightnessContrast(**augs['RandomBrightnessContrast']),
                ChannelShuffle(**augs['ChannelShuffle']),
                HueSaturationValue(**augs['HueSaturationValue']),
                RandomGamma(**augs['RandomGamma']),
            ], p=0.5),
            OneOf([
                Blur(p=0.3, blur_limit=3),
                MedianBlur(p=0.3, blur_limit=3),
                MotionBlur(p=0.5)
            ], p=0.2),
            CoarseDropout(p=0.0, max_holes=8),
            ImageCompression(p=0.1, quality_lower=50),
        ], p=0.5),
        transforms.ResizeNormalize((config.data.input_size[0], config.data.input_size[1]), imagenet=True)]

    train_dataset = ImageDataLoader(path=config.data.train.dataset,
                                    transforms=transforms_list)

    val_dataset = ImageDataLoader(path=config.data.val.dataset,
                                  transforms=[transforms.ResizeNormalize(
                                      (config.data.input_size[0], config.data.input_size[1]), imagenet=True)])

    model = SampleClassificationNetwork(nclasses=N_CLASSES).to(config.pipeline.device)
    model.apply(init_weights)

    print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pipeline.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    prev_loss = np.inf

    if 'weights' in config.pipeline:
        print('Loading weights...')
        model, optimizer, start_epoch, prev_loss = load_parameters_from_cpkt(model, optimizer, config.pipeline.weights)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.pipeline.batch_size,
        shuffle=False, sampler=randomSequentialStratifiedSampler(train_dataset, config.pipeline.batch_size),
        num_workers=int(config.pipeline.n_workers))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.pipeline.batch_size,
        shuffle=False, sampler=randomSequentialSampler(val_dataset, config.pipeline.batch_size),
        num_workers=int(config.pipeline.n_workers))

    nepochs = config.pipeline.n_epochs
    nbatches = len(train_loader)

    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = config.pipeline.learning_rate

    '''Set Cosine Annealing Scheduler'''
    if not config.pipeline.warmup:
        cosine_annealing_epochs_duration = 10
        cosine_annealing_mult = 2
        scheduler_updates_ctr = 0
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=cosine_annealing_epochs_duration * nbatches,
                                                                  eta_min=1e-9,
                                                                  last_epoch=-1)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.pipeline.learning_rate

    """TRAIN LOOP"""
    for epoch in range(start_epoch, nepochs):
        train_iter = iter(train_loader)
        loss_value = 0

        with tqdm(train_loader, postfix=[dict(l=0, lr=0)]) as t:
            for i in range(nbatches):
                data, target = train_iter.next()
                model.train()
                optimizer.zero_grad()

                '''Handle residual batches'''
                if data.shape[0] != config.pipeline.batch_size:
                    continue

                data = data.to(config.pipeline.device)
                target = target.long().to(config.pipeline.device)
                output = model(data)

                loss = criterion(output, target)
                loss.backward()

                #nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optimizer.step()

                loss_value += loss.item()

                '''Cosine annealing part'''
                if not config.pipeline.warmup:
                    scheduler_updates_ctr += 1
                    lr_scheduler.step()

                    if scheduler_updates_ctr == cosine_annealing_epochs_duration * nbatches - 1:
                        cosine_annealing_epochs_duration *= cosine_annealing_mult
                        print(f'\nLR UPDATE, {cosine_annealing_epochs_duration} epochs now\n')
                        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                  T_max=cosine_annealing_epochs_duration * nbatches,
                                                                                  eta_min=1e-9)
                        scheduler_updates_ctr = 0

                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']

                t.postfix[0]['l'] = float('%.4f' % (loss_value / (i + 1)))
                t.postfix[0]['lr'] = current_lr
                t.update()

            mean_val_loss = validate(model, criterion, val_loader)
            print(f'Epoch {epoch}: \n\tloss: {loss_value / nbatches}')
            print(f'\tval_loss: {mean_val_loss}')

            save_experiment_data(config.pipeline.save_prefix, epoch, loss_value / nbatches, mean_val_loss)

            if config.pipeline.save_all or (mean_val_loss < prev_loss):
                path = os.path.join('workspace', config.pipeline.save_prefix ,'weights', config.pipeline.save_prefix)
                if not os.path.isdir(path):
                    os.mkdir(path)
                weights_filename = f"e{epoch}_{mean_val_loss:.4f}_{config.pipeline.save_prefix}.pth"
                path_to_save = os.path.join(path, weights_filename)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': mean_val_loss
                },
                    path_to_save)

                prev_loss = mean_val_loss

                mlflow.log_metric("train_loss", loss_value / nbatches)
                mlflow.log_metric("val_loss", mean_val_loss)


if __name__ == '__main__':
    main()

