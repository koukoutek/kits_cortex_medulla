import torch
import os
import random
import traceback
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import warnings
import csv
import pandas as pd
warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from monai.data import DataLoader, CacheDataset, Dataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.data.utils import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import (ToTensord, Compose, LoadImaged, ToTensord, Spacingd, Transposed, Flipd, RandRotated, 
                              EnsureType, Compose, AsDiscrete, RandSpatialCropSamplesd, SpatialPadd, RandShiftIntensityd, 
                              RandGaussianNoised, ThresholdIntensityd, RandAdjustContrastd, RandGaussianSmoothd)
from utils import *
from pathlib import Path


# from tqdm import tqdm # can be added if not running in the background

def save_checkpoint(model_state_dict, 
                    optimizer_seg_state_dict, 
                    save_path=None):
    """Save checkpoint while training the model

    Args:
        model_state_dict (dict): Dictionary containing model state i.e. weights and biases
            Required: True
        optimizer_state_dict (dict): Dictionary containing optimizer state for the segmentation part i.e. gradients
            Required: True
        save_path (str): Path to save the checkpoint
            Required: False     Default: None  
    Returns:
        -
    """
    torch.save({'model_state_dict': model_state_dict,
                'optimizer_seg_state_dict': optimizer_seg_state_dict,
                }, save_path)
    
def key_error_raiser(ex): raise Exception(ex)

def train(config, log_path, logger):

    cases = sorted([f for f in Path(config['kits_data_dir']).glob('*') if f.is_dir()])
    masks = sorted([f for f in Path(config['data_dir']).glob('*')])
    case_id = sorted([str(f).split('\\')[-1].split('_')[-1].split('.')[0] for f in masks])
    cases = sorted([f for f in cases if str(f).split('\\')[-1].split('_')[-1] in case_id])

    image_files = sorted([os.path.join(f, k) for f in cases for k in os.listdir(f) if 'img' in k])

    train_transforms_config = config['train_transforms']
    eval_transforms_config = config['eval_transforms']
    train_transforms = Compose([
                            LoadImaged(keys=train_transforms_config['LoadImaged_im']['keys'], ensure_channel_first=train_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            ReadNrrdMaskd(keys=train_transforms_config['ReadNrrdMaskd']['keys']),
                            Spacingd(keys=train_transforms_config['Spacingd_im']['keys'], 
                                     pixdim=train_transforms_config['Spacingd_im']['pixdim']),
                            Spacingd(keys=train_transforms_config['Spacingd_seg']['keys'], 
                                     pixdim=train_transforms_config['Spacingd_seg']['pixdim'], mode=train_transforms_config['Spacingd_seg']['mode']),
                            Transposed(keys=train_transforms_config['Transposed']['keys'], 
                                       indices=train_transforms_config['Transposed']['indices']),
                            WindowindCTBasedOnPercentiled(keys=train_transforms_config['WindowindCTBasedOnPercentiled']['keys']),
                            SpatialPadd(keys=train_transforms_config['SpatialPadd']['keys'], spatial_size=train_transforms_config['SpatialPadd']['spatial_size']),
                            RandSpatialCropSamplesd(keys=train_transforms_config['RandSpatialCropSamplesd']['keys'], 
                                                    roi_size=train_transforms_config['RandSpatialCropSamplesd']['roi_size'], 
                                                    num_samples=train_transforms_config['RandSpatialCropSamplesd']['num_samples'], 
                                                    random_size=train_transforms_config['RandSpatialCropSamplesd']['random_size']),
                            # RandAdjustContrastd(keys=train_transforms_config['RandAdjustContrastd']['keys'], prob=train_transforms_config['RandAdjustContrastd']['prob'],
                            #                     gamma=train_transforms_config['RandAdjustContrastd']['gamma']),
                            # RandGaussianSmoothd(keys=train_transforms_config['RandGaussianSmoothd']['keys'], prob=train_transforms_config['RandGaussianSmoothd']['prob'],
                            #                     sigma_x=train_transforms_config['RandGaussianSmoothd']['sigma_x'], sigma_y=train_transforms_config['RandGaussianSmoothd']['sigma_y'], 
                            #                     sigma_z=train_transforms_config['RandGaussianSmoothd']['sigma_z']),
                            # RandShiftIntensityd(keys=train_transforms_config['RandShiftIntensityd']['keys'], 
                            #                     offsets=train_transforms_config['RandShiftIntensityd']['offsets'], 
                            #                     prob=train_transforms_config['RandShiftIntensityd']['prob']),
                            # RandGaussianNoised(keys=train_transforms_config['RandGaussianNoised']['keys'], 
                            #                    prob=train_transforms_config['RandGaussianNoised']['prob'], 
                            #                    mean=train_transforms_config['RandGaussianNoised']['mean'], 
                            #                    std=train_transforms_config['RandGaussianNoised']['std']), 
                            # ThresholdIntensityd(keys=train_transforms_config['ThresholdIntensityd_clip_upper']['keys'], 
                            #                     threshold=train_transforms_config['ThresholdIntensityd_clip_upper']['threshold'], 
                            #                     above=train_transforms_config['ThresholdIntensityd_clip_upper']['above'], 
                            #                     cval=train_transforms_config['ThresholdIntensityd_clip_upper']['cval']),
                            # ThresholdIntensityd(keys=train_transforms_config['ThresholdIntensityd_clip_lower']['keys'], 
                            #                     threshold=train_transforms_config['ThresholdIntensityd_clip_lower']['threshold'], 
                            #                     above=train_transforms_config['ThresholdIntensityd_clip_lower']['above'], 
                            #                     cval=train_transforms_config['ThresholdIntensityd_clip_lower']['cval']),
    ])                            

    val_transforms = Compose([
                            LoadImaged(keys=eval_transforms_config['LoadImaged_im']['keys'], ensure_channel_first=eval_transforms_config['LoadImaged_im']['ensure_channel_first']),
                            ReadNrrdMaskd(keys=eval_transforms_config['ReadNrrdMaskd']['keys']),
                            Spacingd(keys=eval_transforms_config['Spacingd_im']['keys'], 
                                     pixdim=eval_transforms_config['Spacingd_im']['pixdim']),
                            Spacingd(keys=eval_transforms_config['Spacingd_seg']['keys'], 
                                     pixdim=eval_transforms_config['Spacingd_seg']['pixdim'], mode=eval_transforms_config['Spacingd_seg']['mode']),
                            Transposed(keys=eval_transforms_config['Transposed']['keys'], 
                                       indices=eval_transforms_config['Transposed']['indices']),
                            WindowindCTBasedOnPercentiled(keys=eval_transforms_config['WindowindCTBasedOnPercentiled']['keys']),
                            SpatialPadd(keys=eval_transforms_config['SpatialPadd']['keys'], spatial_size=eval_transforms_config['SpatialPadd']['spatial_size']),
    ])

    datadict = [{'image': im, 'image_path': im, 'mask': ma} for im, ma in zip(image_files, masks)]
    train_val_test_split = config['train_val_test_split'] if 'train_val_test_split' in config.keys() else key_error_raiser("Train, validation and test splits not defined in config.")

    train_dict = datadict[:int(train_val_test_split[0]*len(datadict))]
    val_dict = datadict[int(train_val_test_split[0]*len(datadict)):int(train_val_test_split[0]*len(datadict)) + int(train_val_test_split[1]*len(datadict))]
    test_dict = datadict[int(train_val_test_split[0]*len(datadict)) + int(train_val_test_split[1]*len(datadict)):]

    # test_cases = [f['image'] for f in test_dict]
    # # write test cases to csv 
    # with open('test_cases.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['image_path'])
    #     for row in test_dict:
    #         writer.writerow([row['image']])

    # define dataset
    if config['dataset'] == 'Dataset':
        train_dataset = Dataset(data=train_dict, transform=train_transforms)
        val_dataset = Dataset(data=val_dict, transform=val_transforms)
    elif config['dataset'] == 'CacheDataset':
        train_dataset = CacheDataset(data=train_dict, transform=train_transforms, cache_rate=config['cache_rate'])
        val_dataset = CacheDataset(data=val_dict, transform=val_transforms, cache_rate=config['cache_rate'])

    
    for i in range(len(train_dataset)):
        sample = train_dataset[i][0]
        print(sample['image_meta_dict']['filename_or_obj'])
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        X = np.hstack((sample['image'][0], sample['mask'][0], sample['mask'][1], sample['mask'][2], sample['mask'][3]))
        tracker = IndexTracker(ax, X, vmin=np.amin(X), vmax=np.amax(X))
        fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
        plt.show()
    exit(1)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    logger.info('Train size {}. Val size {}.'.format(train_size, val_size))

    # initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'] , 
                              shuffle=config['dataloader']['shuffle'], 
                              num_workers=config['dataloader']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=1, 
                            shuffle=config['dataloader']['shuffle'],
                            num_workers=config['dataloader']['num_workers'])

    # initialize model
    if config['model']['name'] == 'UNet':
        model = UNet(spatial_dims=config['model']['spatial_dims'], in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'],
                     kernel_size=config['model']['kernel_size'], up_kernel_size=config['model']['up_kernel_size'], channels=config['model']['channels'],
                     strides=config['model']['strides'], norm=config['model']['norm'], dropout=config['model']['dropout'], 
                     num_res_units=config['model']['num_res_units'])
    else: 
        raise Exception("No model has been defined in the config file")
    logger.info('Model {}.'.format(config['model']['name']))
    
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model) # use multiple GPUs
    model.to(device=torch.device(config['device']))
        
    # initialize optimizer
    if config['optimizer']['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['learning_rate'], betas=config['optimizer']['betas'], 
                                     weight_decay=config['optimizer']['weight_decay']) 
    elif config['optimizer']['name'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['optimizer']['learning_rate'], momentum=config['optimizer']['momentum'],
                                    weight_decay=config['optimizer']['weight_decay'])
    else: 
        raise Exception("No optimizer has been defined in the config file")
    logger.info('Training with optimizer {} '.format(optimizer))

    # initialize scheduler
    if 'scheduler' in config.keys() and config['scheduler']['name'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=config['scheduler']['T_0'], eta_min=config['scheduler']['eta_min'])
        logger.info('Scheduler {}'.format(scheduler))
    else:
        logger.info('No scheduler for the learning rate has been defined')

    # initialize loss
    if config['loss']['name'] == 'DiceLoss':
        loss = DiceLoss(softmax=config['loss']['softmax'], include_background=config['loss']['include_background'])
    elif config['loss']['name'] == 'DiceCELoss':
        loss = DiceCELoss(softmax=config['loss']['softmax'], include_background=config['loss']['include_background'])
    else:
        raise Exception("No loss has been defined in the config file")
    logger.info('Loss function to minimize {}'.format(loss))

    if config['save_model_when']:
        metric_threshold = config['save_model_when']
    else:
        metric_threshold = 0.95

    writer = SummaryWriter()

    losses = []
    val_losses = []

    post_pred_transforms = config['post_pred_transforms']
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=post_pred_transforms['AsDiscrete']['argmax'], 
                                                  to_onehot=post_pred_transforms['AsDiscrete']['to_onehot'])])
    post_label = Compose([EnsureType()])

    if config['metric']['name'] == 'DiceMetric':
        train_metric = DiceMetric(include_background=config['metric']['include_background'], reduction=config['metric']['reduction'])
        val_metric =  DiceMetric(include_background=config['metric']['include_background'], reduction=config['metric']['reduction'])
    else:
        raise Exception("No metric has been defined in the config file")
    logger.info('Metric {}'.format(train_metric))

    sliding_window_roi = config['sliding_window_roi'] if 'sliding_window_roi' in config.keys() else key_error_raiser("Sliding window ROI not defined in config.")

    for epoch in range(config['epochs']):
        model.train()

        for batch, train_data in enumerate(train_loader, 1):
            image, segmentation = train_data['image'].float().to(device=torch.device(config['device'])), train_data['mask'].float().to(device=torch.device(config['device']))

            try:
                optimizer.zero_grad()
                out = model(image)

                loss_s = loss(out, segmentation)
                loss_s.backward()

                _outputs = [post_pred(i) for i in decollate_batch(out)]
                _labels = [post_label(i) for i in decollate_batch(segmentation)]

                optimizer.step()
                train_metric(y_pred=_outputs, y=_labels)

            except Exception as e:
                print('Caught the following exception {}'.format(traceback.format_exc()))
            losses.append(loss_s.item())
        metric = train_metric.aggregate().item() if config['metric']['reduction'] == 'mean' else train_metric.aggregate()
        if epoch > 400 and 'scheduler' in config.keys(): scheduler.step()

        if epoch > 400 and epoch % 50 == 0 and torch.mean(metric) > 0.99:
            logger.info(f'Writing images to Tensorboard...')
            plot_2d_or_3d_image(data=image, step=0, writer=writer, frame_dim=-1, tag=f'image at epoch: {epoch}')
            plot_2d_or_3d_image(data=segmentation, step=0, writer=writer, frame_dim=-1, tag=f'label at epoch: {epoch}')
            plot_2d_or_3d_image(data=out, step=0, writer=writer, frame_dim=-1, tag=f'model output at epoch: {epoch}')

        writer.add_scalar(tag='Loss/train', scalar_value=losses[-1], global_step=epoch)
        logger.info(f'Epoch {epoch} of {config["epochs"]} with Train loss {losses[-1]}')
        if epoch % 10 == 0: logger.info(f'Train metric per class {metric}')
        logger.info(f'Train metric mean {torch.mean(metric)}')
        logger.info(f'-------------- Finished epoch {epoch} -------------')
        train_metric.reset()

        if epoch % config['val_interval'] == 0:
            with torch.no_grad():
                # evaluate model
                model.eval()

                for _, val_data in enumerate(val_loader, 1):
                    val_image, val_segm = val_data['image'].float().to(device=torch.device(config['device'])), val_data['mask'].float().to(device=torch.device(config['device']))

                    try:
                        val_out = sliding_window_inference(inputs=val_image, roi_size=sliding_window_roi, sw_batch_size=12, predictor=model)

                        loss_s = loss(val_out, val_segm)

                        val_outputs = [post_pred(i) for i in decollate_batch(val_out)]
                        val_labels = [post_label(i) for i in decollate_batch(val_segm)]

                        val_metric(val_outputs, val_labels)
                    except Exception as e:
                        print(f'Exception caught while validating in {traceback.format_exc()}. Aborting...')
                    # record loss
                    val_losses.append(loss_s.item())
                metric = val_metric.aggregate().item() if config['metric']['reduction'] == 'mean' else val_metric.aggregate()

                writer.add_scalar(tag='Loss/eval', scalar_value=val_losses[-1], global_step=epoch)
                logger.info(f'Eval loss {val_losses[-1]}')
                logger.info(f'Eval metric per class {metric}')
                logger.info(f'Eval metric mean {torch.mean(metric)}')
                logger.info(f'-------------- Finished epoch {epoch} -------------') 
                val_metric.reset()

                # save models
                if torch.mean(metric) > metric_threshold:
                    if not os.path.exists(log_path.joinpath(config['logs']).joinpath('models')):
                        os.makedirs(log_path.joinpath(config['logs']).joinpath('models'))
                    save_checkpoint(model_state_dict=model.state_dict(), optimizer_seg_state_dict=optimizer.state_dict(), 
                                    save_path=log_path.joinpath(config['logs']).joinpath('models/model{}.tar'.format(epoch)))

    return model  