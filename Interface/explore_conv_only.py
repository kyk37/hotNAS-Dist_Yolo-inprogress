from torchvision import models

from torchvision.models import *
from torch import nn
import torch
import sys
import math
sys.path.append("../Performance_Model")
sys.path.append("../cifar10_models")
import cifar10_models
import PM_Config
import PM_Layer
import PM_FPGA_Template
from search_space import *
from CONV_PM_IF import *
import argparse



def get_max_k(model):
    max_k = 0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            cur_k = is_same(layer.kernel_size)
            if cur_k > max_k:
                max_k = cur_k
    return  max_k

def get_performance(model,dataset_name, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p):
    if dataset_name=="imagenet":
        input = torch.Tensor(torch.Size([1, 3, 224, 224])).to(torch.float32)
    elif dataset_name=="cifar10":
        input = torch.Tensor(torch.Size([1, 3, 32, 32])).to(torch.float32)
    cTT = 0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            input_shape = list(input.shape)
            input_shape[1] = layer.in_channels
            input = torch.Tensor(torch.Size(input_shape)).to(torch.float32)
            input = layer(input)

            [B, M, N, R, C, K, S, T, P] = (
                1, layer.out_channels, layer.in_channels, input.shape[2], input.shape[3], is_same(layer.kernel_size),
                is_same(layer.stride), tell_conv_type(layer.in_channels, layer.groups), is_same(layer.padding))

            if T == "cconv":
                [r_Ports, r_DSP, r_BRAM, r_BRAM_Size, BITWIDTH] = (HW_constraints["r_Ports_BW"], HW_constraints["r_DSP"],
                                                                   HW_constraints["r_BRAM_Size"], HW_constraints["r_BRAM"],
                                                                   HW_constraints["BITWIDTH"])

                Layer = PM_Layer.Layer_Class(B, M, N, R, C, K, S, "cconv", P)
                acc_1 = PM_FPGA_Template.FPGA_Templates(Tm, Tn, Tr, Tc,
                                                        Tk, W_p, I_p, O_p, "cconv", r_Ports, r_DSP, r_BRAM, r_BRAM_Size,
                                                        BITWIDTH)
                if acc_1.Success==False:
                    return -1
                else:
                    cTT += acc_1.get_layer_latency(Layer)[0]

        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer,
                                                                                                      nn.AvgPool2d):
            input = layer(input)
    # print("\tTotal Time:", (cTT) / 10 ** 5)
    return cTT / 10 ** 5



def do_exploration(model,dataset_name):
    (rangeTm,rangeTc,rangeTr,range_Wp,range_Ip,range_Op) = search_space['hw_only_cconv']

    best_lat = 999999999999
    best_design = []
    Tk = get_max_k(model)
    for Tm in rangeTm:
        Tn = math.floor(HW_constraints["r_DSP"] / Tm)
        for Tc in rangeTc:
            for Tr in rangeTr:
                for W_p in range_Wp:
                    for I_p in range_Ip:
                        for O_p in range_Op:
                            cur_lat = get_performance(model,dataset_name, Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p)
                            if cur_lat!=-1:
                                print([Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p], cur_lat)
                            if cur_lat!=-1 and cur_lat<best_lat:
                                best_lat = cur_lat
                                best_design = [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p]
    return best_lat,best_design


import time

if __name__== "__main__":

    parser = argparse.ArgumentParser('Parser User Input Arguments')
    parser.add_argument(
        '-d', '--dataset',
        default='cifar10'
    )

    parser.add_argument(
        '-m', '--model',
        default='resnet18'
    )



    args = parser.parse_args()

    dataset_name = args.dataset



    model_name = args.model

    if dataset_name == "imagenet":
        model = globals()[model_name]()
    elif dataset_name == "cifar10":
        model = getattr(cifar10_models, model_name)(pretrained=True)


    print("="*10,"Model:",model_name,"="*10)
    print("-"*10,"Search Space","-"*10)
    print("\tTm", search_space["hw_only_cconv"][0])
    print("\tTr", search_space["hw_only_cconv"][1])
    print("\tTc", search_space["hw_only_cconv"][2])
    print("\tW_p", search_space["hw_only_cconv"][3])
    print("\tI_p", search_space["hw_only_cconv"][4])
    print("\tO_p", search_space["hw_only_cconv"][5])



    start_time = time.time()

    best_lat,best_design = do_exploration(model,dataset_name)

    end_time = time.time()

    print(model_name, best_lat,best_design, (end_time - start_time))

    # Model_Zoo = [ 'alexnet', 'densenet121', 'densenet161',  'densenet169', 'densenet201', 'squeezenet1_0', 'squeezenet1_1',  'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn','wide_resnet101_2', 'wide_resnet50_2',  'vgg11',  'resnet50', 'resnet101', 'resnet152', 'resnet18', 'resnet34' ]


    # for model_name in Model_Zoo:


        # sys.exit(0)
