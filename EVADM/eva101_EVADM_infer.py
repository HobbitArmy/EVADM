# -*- coding: utf-8 -*-
"""
Created on 2024/8/29 22:40

@author: Luxy
"""

import os
import torch

from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from eva100_ldm_infer_batch_only import run_batch
from PIL import Image
random_seed = 2023
torch.manual_seed(random_seed)  # cpu
torch.cuda.manual_seed(random_seed)  # gpu


if __name__ == '__main__':
    # 1. input data
    img_dir = r'img_test/'
    # image list
    img_list = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.jpg')]

    # 2. model define
    prj_dir_dict= {
                    'evadm_x2':r'weights/EVADM_x2/',
                    # 'evadm_x4':r'weights/EVADM_x4_pt/',
    }
    for model_name, prj_dir in prj_dir_dict.items():
        if model_name.split('_')[-1]=='x2':
            up_f = 2; custom_steps = 16; batch_size = 1; patch_size = 256
        elif model_name.split('_')[-1]=='x4':
            up_f = 4; custom_steps = 32; batch_size = 1; patch_size = 128
        elif model_name.split('_')[-1]=='x8':
            up_f = 8; custom_steps = 32; batch_size = 1; patch_size = 128

        # find project.yaml file end with -project.yaml
        config_path = [prj_dir + '/configs/' + file for file in os.listdir(prj_dir + '/configs/') if file.endswith('-project.yaml')][0]
        ckpt_path = [prj_dir + '/checkpoints/trainstep_checkpoints/' + file for file in os.listdir(prj_dir + '/checkpoints/trainstep_checkpoints/') if file.endswith('.ckpt')][-1]
        config = OmegaConf.load(config_path)
        model, _ = load_model(config, ckpt_path, None, None)
        out_dir = img_dir+r'../SR_%s/'%model_name
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        # 3. infer over the input
        for filename in img_list:
            print("%s x%i" % (filename.split('/')[-1], up_f))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if os.path.exists(out_dir + filename):  # check if save_dir already has this file
                continue
            # read image
            img_path = os.path.join(img_dir, filename)
            # if img size < 256, batch_size=1  use PIL.Image
            img = Image.open(img_path)
            if img.size[0] < 256 or img.size[1] < 256: batch_size = 1
            image_sr_array = run_batch(model, selected_path=img_path, task='superresolution', custom_steps=custom_steps,
                                       resize_enabled=True, up_f=up_f, patch_size=patch_size,
                                       batch_size=batch_size)
            img_PIL = Image.fromarray(image_sr_array)
            img_PIL.save(out_dir + filename)



