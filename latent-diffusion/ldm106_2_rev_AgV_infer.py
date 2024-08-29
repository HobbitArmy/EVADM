# -*- coding: utf-8 -*-
"""
Created on 2024/3/7 17:57

@author: LU
"""

import os
import torch

from scripts_ldm.sample_diffusion import load_model
from omegaconf import OmegaConf
from tqdm import tqdm
from ldm103_ldm_infer_batch_only import run_batch
from PIL import Image
random_seed = 2023
torch.manual_seed(random_seed)  # cpu
torch.cuda.manual_seed(random_seed)  # gpu

if __name__ == '__main__':
    # 1. input data
    img_dir = r'E:\data\dataset\x2022 agriculture_vision\Agriculture-Vision-2021\test\images\test1o10/'
    # image list
    img_list = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.jpg')]
    # 2. SR task ratio
    task = 'superresolution'
    resize_enabled = True
    # # 2x
    # up_f=2
    # custom_steps=16
    # batch_size = 2
    # patch_size = 256
    # # 4x
    # up_f = 4
    # custom_steps = 32
    # batch_size = 4
    # patch_size = 128
    # # 8x
    # up_f=8
    # custom_steps=32
    # batch_size = 2
    # patch_size = 128
    # 3. model define
    prj_dir_dict= {
                    # 'evadm_x4':r'E:\CropSR_files\workdirs\workdir_select\Diff\2023-09-19T14-45-52_EVADM_full_noattn_x4_256_32a2_vq4/',
                    # 'ldm_x4':r'E:\CropSR_files\workdirs\workdir_select\Diff\2023-09-08T22-50-52_ldm_x4_256_32a2_vq4/',
                    # 'evadm_x2':r'E:\CropSR_files\workdirs\workdir_select\Diff\2023-09-21T15-35-45_EVADM_full_noattn_x2_256_16a4_vq2/',
                    'evadm_x8':r'E:\CropSR_files\workdirs\workdir_select\Diff\2023-09-25T17-33-59_EVADM_full_noattn_x8_512_8a8_vq8/',
    }
    for model_name, prj_dir in prj_dir_dict.items():
        if model_name.split('_')[-1]=='x2':
            up_f = 2; custom_steps = 16; batch_size = 2; patch_size = 256
        elif model_name.split('_')[-1]=='x2':
            up_f = 4; custom_steps = 32; batch_size = 4; patch_size = 128
        elif model_name.split('_')[-1]=='x8':
            up_f = 8; custom_steps = 32; batch_size = 2; patch_size = 128

        # find project.yaml file end with -project.yaml
        config_path = [prj_dir + '/configs/' + file for file in os.listdir(prj_dir + '/configs/') if file.endswith('-project.yaml')][0]
        ckpt_path = [prj_dir + '/checkpoints/trainstep_checkpoints/' + file for file in os.listdir(prj_dir + '/checkpoints/trainstep_checkpoints/') if file.endswith('.ckpt')][-1]
        config = OmegaConf.load(config_path)
        model, _ = load_model(config, ckpt_path, None, None)
        out_dir = prj_dir+r'/AgV_test1o10/'
        # 4. infer over the input
        for filename in tqdm(img_list):
            print("%s x%i" % (filename.split('/')[-1], up_f))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if os.path.exists(out_dir + filename):  # 判断save_dir中是否已经存在该文件
                continue
            # 读取图像
            img_path = os.path.join(img_dir, filename)

            image_sr_array = run_batch(model, selected_path=img_path, task=task, custom_steps=custom_steps,
                                       resize_enabled=resize_enabled, up_f=up_f, patch_size=patch_size,
                                       batch_size=batch_size)
            img_PIL = Image.fromarray(image_sr_array)
            img_PIL.save(out_dir + filename)





