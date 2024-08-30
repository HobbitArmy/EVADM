# -*- coding: utf-8 -*-
"""
Created on 2024/8/29 23:08

Calculate SRFI metrics
The FID calculation is based on the implementation of pytorch-fid:
https://github.com/mseitzer/pytorch-fid

@author: Luxy
"""

def rfid(fid_sr, fid_up):
    """
    please refer to relative_fid_score.py for detail
    """
    return 1-fid_sr/fid_up

def SRFI(rfid, ssim, r, alpha=2):
    return 10**((ssim+alpha*rfid)/r)

