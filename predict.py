import torch
import os
import traceback
import numpy as np
import nibabel as nib
import pandas as pd
import time

from monai.data import DataLoader, Dataset
from monai.data.utils import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, UNet
from monai.transforms import (Compose, LoadImaged, Spacingd, Transposed, 
                              EnsureType, Compose, AsDiscrete, Resize)
from utils import *
from pathlib import Path
from nibabel import Nifti1Image, save

metrics = {'case': [], 'time': [], #  case and time
           'dice_cortex': [], 'dice_medulla': []
           } 

val_cases = ['case_00419', 'case_00451', 'case_00458', 'case_00463', 'case_00488']
train_cases = ['case_00009', 'case_00039', 'case_00047', 'case_00070', 'case_00079', 
               'case_00098', 'case_00113', 'case_00189', 'case_00204', 'case_00205', 
               'case_00248', 'case_00267', 'case_00283', 'case_00405', 'case_00407']

cases = sorted([f for f in Path('/PATH/TO/KITS21').glob('*') if f.is_dir()]) # change to your path
cases = [f for f in cases if f.name in val_cases]
masks = sorted([f for f in Path('/PATH/TO/MASKS').glob('*')]) # change to your path
masks = [f for f in masks if f"case_{f.name.split('.')[0].split('_')[-1]}" in val_cases]
case_id = sorted([str(f).split('\\')[-1].split('_')[-1].split('.')[0] for f in masks])
cases = sorted([f for f in cases if str(f).split('\\')[-1].split('_')[-1] in case_id])

image_files = sorted([os.path.join(f, k) for f in cases for k in os.listdir(f) if 'img' in k])

datadict = [{"image": im, 'image_path': im, "image_copy": im, "mask": ma} for im, ma 
            in zip(image_files, masks)]

val_transforms = Compose([
                        LoadImaged(keys=['image', 'image_copy'], ensure_channel_first=True),  
                        ReadNrrdMaskd(keys=['mask']),
                        Spacingd(keys=['image'], 
                                    pixdim=[2.5, 2.5, 2.5]),
                        Transposed(keys=['image'], 
                                    indices=[0, 2, 3, 1]),
                        WindowindCTBasedOnPercentiled(keys=['image']),
                        ])

val_dataset = Dataset(data=datadict, transform=val_transforms)
val_size = len(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=1, 
                            shuffle=False,
                            num_workers=0)

# # UNET
model_path = Path('/PATH/TO/UNET.tar') # change to your path
model = UNet(spatial_dims=3, in_channels=1, out_channels=4,
                     kernel_size=3, up_kernel_size=3, channels=[32, 64, 128, 256, 512],
                     strides=[2, 2, 2, 2], norm='instance', dropout=0.3, 
                     num_res_units=3)

# # UNETR
# model_path = Path('/PATH/TO/UNETR.tar') # change to your path
# model = UNETR(in_channels=1, out_channels=4, img_size=[96, 96, 96], 
#                       feature_size=16, hidden_size=768, mlp_dim=3072, 
#                       num_heads=12, pos_embed='perceptron', norm_name='instance', 
#                       conv_block=True, res_block=True, dropout_rate=0.3)

saved_model = torch.load(model_path, map_location='cpu')
model.load_state_dict(saved_model['model_state_dict'])
model.float()#.cuda()

post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=4)]) 
post_label = Compose([EnsureType()])

with torch.no_grad():
    model.eval()

    for k, val_data in enumerate(val_loader):
        start_time = time.time()
        print(f'{k} ---- {image_files[k]}')
        val_image, image_copy = val_data['image'].float().to(device=torch.device('cpu')), val_data['image_copy'].float().to(device=torch.device('cpu'))
        val_segm = val_data['mask'].float().to(device=torch.device('cpu'))

        try:
            val_out = sliding_window_inference(inputs=val_image, roi_size=(96,96,96), sw_batch_size=48, predictor=model, overlap=.25, mode='constant')
            val_outputs = [post_pred(i) for i in decollate_batch(val_out)]

            out = val_outputs[0].cpu().detach().numpy().astype(np.float32)
            pred_cortex, pred_medulla = out[1], out[2]
            
            pred_cortex = np.transpose(pred_cortex, (2, 0, 1))
            pred_medulla = np.transpose(pred_medulla, (2, 0, 1))
            
            pred_cortex = np.squeeze(Resize(spatial_size=image_copy[0, 0].shape, mode='trilinear')(np.expand_dims(pred_cortex.astype(np.float32), axis=0)))
            pred_medulla = np.squeeze(Resize(spatial_size=image_copy[0, 0].shape, mode='trilinear')(np.expand_dims(pred_medulla.astype(np.float32), axis=0)))

            pred_cortex[pred_cortex >= 0.25] = 1
            pred_cortex[pred_cortex < 0.25] = 0
            pred_medulla[pred_medulla >= 0.25] = 1
            pred_medulla[pred_medulla < 0.25] = 0

            metrics['case'].append(datadict[k]['image'])
            metrics['time'].append(time.time() - start_time)

            man_cortex = val_segm[0,1].detach().cpu().numpy()
            man_medulla = val_segm[0,2].detach().cpu().numpy()

            metrics['dice_cortex'].append(dice(pred_cortex, man_cortex))
            metrics['dice_medulla'].append(dice(pred_medulla, man_medulla))
            print(f'Cortex dice {metrics["dice_cortex"][-1]}')
            print(f'Medulla dice {metrics["dice_medulla"][-1]}')
            print(f'Done processing case...')

            # Create a Nifti1Image
            kidney = pred_cortex + pred_medulla
            kidney[np.where(pred_medulla == 1)] = 2
            image = nib.load(val_image.meta['filename_or_obj'][0])
            nifti_image = Nifti1Image(kidney, affine=image.affine, header=image.header)
            output_path = os.path.join('predictions', f'prediction_case_{case_id[k]}.nii.gz')
            # Save the Nifti image
            os.makedirs('predictions', exist_ok=True)
            save(nifti_image, output_path)
            print(f'Nifti image saved to {output_path}')

        except Exception as e:
            print(f'Exception caught while validating in {traceback.format_exc()}. Aborting...')

df = pd.DataFrame(metrics)
df.to_csv('validation_metrics_UNET.csv', index=False)
print(f'Validation metrics saved to validation_metrics_UNET.csv')