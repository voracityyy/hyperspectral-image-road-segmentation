# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:01:53 2026

@author: joshp
"""

import segmentation_models_pytorch as smp

def build_unet():
    model = smp.Unet(
        "resnet18", 
        encoder_weights="imagenet", 
        classes=1
    )
    return model
