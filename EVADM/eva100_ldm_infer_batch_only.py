# -*- coding: utf-8 -*-
"""
Created on 2023/10/19 21:17

@author: LU
"""

import os
import numpy as np
from PIL import Image
from einops import rearrange, repeat
import torch, torchvision
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
import time

def ldm_tensor2img(sample):
    sample = torch.clamp(sample, -1., 1.)
    sample = (sample + 1.) / 2. * 255
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (0, 2, 3, 1))
    return sample

def get_cond_batch(mode, selected_path, up_f, patch_size=128, batch_size=4, mend_gap=True):
    example_list = dict()
    # mode = "superresolution"
    # visualize_cond_img(selected_path)
    c = Image.open(selected_path)
    c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    # cut c to [1,3,patch_size,patch_size]
    patch_list = []
    row_num, col_num = c.shape[2]//patch_size, c.shape[3]//patch_size
    for i in range(row_num):
        for j in range(col_num):
            patch_list.append(c[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size])
    if mend_gap:
        row_num+=1
        if c.shape[2]%patch_size != 0:
            for j in range(col_num):
                patch_list.append(c[:,:,-patch_size:,j*patch_size:(j+1)*patch_size])

    patch_batch_list = []
    c_up_batch_list = []
    for i in range(0, len(patch_list), batch_size):
        c_batch = torch.cat(patch_list[i:i+batch_size], dim=0)
        c_up_batch = torchvision.transforms.functional.resize(c_batch, size=[up_f * patch_size, up_f * patch_size])
        c_up_batch = rearrange(c_up_batch, 'b c h w -> b h w c')
        c_up_batch_list.append(c_up_batch)
        c_batch = rearrange(c_batch, 'b c h w -> b h w c')
        c_batch = 2. * c_batch - 1.     # c=>[0,1] -> c=>[-1,1]
        c_batch = c_batch   # .to(torch.device("cuda"))
        patch_batch_list.append(c_batch)

    example_list["LR_image"] = patch_batch_list     # LR 推理 采用LR_image 作为condition
    example_list["image"] = c_up_batch_list     # 采用上采样的LR作为 输入x
    example_list["row_num"] = row_num
    example_list["col_num"] = col_num
    example_list["mend_gap"] = mend_gap
    return example_list

@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, img_callback=None,
                    temperature=1., noise_dropout=0., score_corrector=None,
                    corrector_kwargs=None, x_T=None, log_every_t=None
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    # print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_T=x_T)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(batch, model, mode="vanilla", custom_steps=None, eta=1.0, swap_mode=False, masked=False,
                              invert_mask=True, quantize_x0=False, custom_schedule=None, decode_interval=1000,
                              resize_enabled=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True,
                              ddim_use_x0_pred=False):
    log = dict()
    # z: encoded x (autoencoder),
    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)

    log_every_t = 1 if save_intermediate_vid else None

    z0 = None

    log["input"] = x

    with model.ema_scope("Plotting"):
        t0 = time.time()
        img_cb = None

        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                                eta=eta,
                                                quantize_x0=quantize_x0, img_callback=img_cb, mask=None, x0=z0,
                                                temperature=temperature, noise_dropout=noise_dropout,
                                                score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_T=x_T, log_every_t=log_every_t)
        t1 = time.time()

        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0

    return log

def run_batch(model, selected_path, task, custom_steps=32, resize_enabled=False, classifier_ckpt=None, global_step=None,
              up_f=4, patch_size=128, batch_size=4, mend_gap=True):
    img_shape = Image.open(selected_path).size
    sr_image_array = np.zeros((img_shape[1]*up_f, img_shape[0]*up_f, 3), dtype=np.uint8)
    example_list = get_cond_batch(task, selected_path, up_f=up_f, patch_size=patch_size, batch_size=batch_size,
                                  mend_gap=mend_gap)
    LR_image_list = example_list["LR_image"]
    # print("%s mini-patch-num:%i" % (selected_path.split('/')[-1], len(LR_image_list)))

    for i in range(len(LR_image_list)):
        # if (i*batch_size)%64!=0:
        #     continue  # debug
        example = {"LR_image": LR_image_list[i].to(torch.device("cuda")), "image": example_list["image"][i]}

        save_intermediate_vid = False
        masked = False
        guider = None
        ckwargs = None
        mode = 'ddim'
        ddim_use_x0_pred = False
        temperature = 1.
        eta = 1.
        make_progrow = True
        custom_shape = None
        # height, width = example["image"].shape[1:3]
        # if up_f==2:split_input = height > 128 and width > 128
        # if up_f==4:split_input = height > 512 and width > 512
        # if up_f==8:split_input= height >= 64 and width >= 64
        #
        # # Slide
        # if split_input:
        #     ks = 128        # f=4, 128 -x4> 512
        #     stride = 64
        #     vqf = up_f  #
        #     if up_f == 8:
        #         ks = 64     # f=8, 64 -x8-> 512
        #         stride = 32
        #         vqf = 8  #
        #     if up_f == 2:
        #         ks = 192    # f=2, 192+64*1 -x2-> 512
        #         stride = 64
        #         vqf = 2  #
        #     model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
        #                                 "vqf": vqf,
        #                                 "patch_distributed_vq": True,
        #                                 "tie_braker": False,
        #                                 "clip_max_weight": 0.5,
        #                                 "clip_min_weight": 0.01,
        #                                 "clip_max_tie_weight": 0.5,
        #                                 "clip_min_tie_weight": 0.01}
        # else:
        #     if hasattr(model, "split_input_params"):
        #         delattr(model, "split_input_params")
        invert_mask = False

        x_T = None
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)    # x 的输入 直接随机
            x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])

        logs = make_convolutional_sample(example, model,
                                         mode=mode, custom_steps=custom_steps,
                                         eta=eta, swap_mode=False, masked=masked,
                                         invert_mask=invert_mask, quantize_x0=True,
                                         custom_schedule=None, decode_interval=10,
                                         resize_enabled=resize_enabled, custom_shape=custom_shape,
                                         temperature=temperature, noise_dropout=0.,
                                         corrector=guider, corrector_kwargs=ckwargs, x_T=x_T,
                                         save_intermediate_vid=save_intermediate_vid,
                                         make_progrow=make_progrow, ddim_use_x0_pred=ddim_use_x0_pred
                                         )
        sample1 = logs['sample'].detach().cpu()
        sample_img = ldm_tensor2img(sample1)
        # paste sample_img to sr_image_array
        col_num, row_num = example_list["col_num"], example_list["row_num"]
        sr_size = patch_size*up_f
        row_idx, col_idx = (i*batch_size) // col_num, (i*batch_size) % col_num
        left_top_coord = (row_idx * sr_size, col_idx * sr_size)
        if example_list["mend_gap"] and row_idx==row_num-1:
            left_top_coord = (sr_image_array.shape[0]-sr_size, col_idx * sr_size)
        for j in range(batch_size):
            sr_image_array[left_top_coord[0] : left_top_coord[0] + 1 * sr_size,
                            left_top_coord[1]+j*sr_size:left_top_coord[1]+(j+1)*sr_size,:] = sample_img[j]

        # # view
        # import matplotlib.pyplot as plt
        # plt.close()
        # import matplotlib
        # matplotlib.use('Qt5Agg')
        # plt.imshow(sr_image_array)
        # Image.fromarray(sr_image_array).save(os.path.join(os.path.dirname(selected_path), 'sr_%s.png' % i))
        # if i>2: break

    return sr_image_array
