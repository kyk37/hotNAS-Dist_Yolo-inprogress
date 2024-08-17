# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:35:13 2024

@author: HuLab
"""
from resnet_imagenet import _resnet, BasicBlock, Bottleneck
import torch
import os

def test_resnet18_pretrained():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=True, progress=True, device=device)
    #model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=True, progress=True, device=device)
    #model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=True, progress=True, device=device)
    input_shape = (1, 3, 224, 224)  # ImageNet input shape
    dummy_input = torch.randn(input_shape).to(device)

    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f'resnet18 output shape: {output.shape}')

if __name__ == "__main__":
    test_resnet18_pretrained()
