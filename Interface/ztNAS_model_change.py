from torch import nn
import torch
import sys
from utility import is_same
import pattern_kernel
import copy_conv2d
import torch.nn.functional as F


def get_last_attr_idx(model,seq):

    last_not_digit = 0
    pre_attr = model
    last_attr = []


    a = model
    for idx in range(len(seq)):
        var = seq[idx]
        if var.isdigit():
            a = a[int(var)]
        else:
            pre_attr = a
            a = getattr(a, var)
            last_not_digit = idx
            last_attr = a
    return pre_attr,last_attr,last_not_digit


def ztNAS_modify_kernel_shape(model,layer, layer_name,var_k,increase=True):

    # print("Debug:")
    # print("name:",layer_name)

    [M, N, K, S, G, P, b] = (
        layer.out_channels, layer.in_channels, is_same(layer.kernel_size),
        is_same(layer.stride), layer.groups, is_same(layer.padding), layer.bias)

    ## Weiwen: 03-29
    ## Step 1: Translate layer name to locate layer module
    ##
    seq = layer_name.split(".")
    (pre_attr,last_attr,last_not_digit) = get_last_attr_idx(model, seq)

    ## Weiwen: 03-29
    ## Step 2: Backup weights and bias if exist
    ##
    is_b = False
    if type(b)==nn.Parameter:
        ori_para_w = model.state_dict()[layer_name + ".weight"][:]
        ori_para_b = model.state_dict()[layer_name + ".bias"][:]
        is_b = True
    else:
        ori_para_w = model.state_dict()[layer_name + ".weight"][:]

    ## Weiwen: 03-29
    ## Step 3: Translate layer name to locate layer module
    ##
    # print("last_not_digit:", last_not_digit, "in", len(seq))
    if last_not_digit == len(seq) - 1:
        # last one is the attribute, directly setattr
        new_conv = nn.Conv2d(N, M, kernel_size=(K + var_k, K + var_k), stride=(S, S), padding=(int(P + var_k/2), int(P + var_k/2)), groups=G, bias=is_b)
        setattr(pre_attr, seq[-1], new_conv)
    elif last_not_digit == len(seq) - 2:
        # one index last_attr[]
        new_conv = nn.Conv2d(N, M, kernel_size=(K + var_k, K + var_k), stride=(S, S), padding=(int(P + var_k/2), int(P + var_k/2)), groups=G, bias=is_b)
        last_attr[int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-2], last_attr)
    elif last_not_digit == len(seq) - 3:
        # two index last_attr[][]
        new_conv = nn.Conv2d(N, M, kernel_size=(K + var_k, K + var_k), stride=(S, S), padding=(int(P + var_k/2), int(P + var_k/2)), groups=G, bias=is_b)
        last_attr[int(seq[-2])][int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-3], last_attr)
    else:
        print("more than 2 depth of index from last layer is not support!")
        sys.exit(0)

    ## Weiwen: 03-29
    ## Step 4: Setup new parameters from backup
    ##
    if is_b:
        model.state_dict()[layer_name + ".bias"][:] = ori_para_b

    if var_k>0:
        pad_fun = torch.nn.ZeroPad2d(int(var_k/2))
        model.state_dict()[layer_name + ".weight"][:] = pad_fun(ori_para_w)



    return model



def ztNAS_add_kernel_mask(model,layer, layer_name, is_pattern, pattern, pattern_ones):
    [M, N, K, S, G, P, b] = (
        layer.out_channels, layer.in_channels, is_same(layer.kernel_size),
        is_same(layer.stride), layer.groups, is_same(layer.padding), layer.bias)

    ## Weiwen: 03-29
    ## Step 1: Translate layer name to locate layer module
    ##
    seq = layer_name.split(".")
    (pre_attr, last_attr, last_not_digit) = get_last_attr_idx(model, seq)


    ## Weiwen: 03-29
    ## Step 2: Backup weights and bias if exist
    ##
    is_b = False
    if type(b)==nn.Parameter:
        ori_para_b = model.state_dict()[layer_name + ".bias"][:]
        is_b = True
    ori_para_w = model.state_dict()[layer_name + ".weight"][:]

    ## Weiwen: 03-29
    ## Step 3: Translate layer name to locate layer module
    ##
    if last_not_digit == len(seq) - 1:
        # last one is the attribute, directly setattr
        new_conv = copy_conv2d.Conv2d_Custom(N, M, kernel_size=(K,K), stride=(S, S),
                             padding=(P,P), groups=G, bias=is_b, is_pattern=is_pattern, pattern=pattern, pattern_ones = pattern_ones)
        setattr(pre_attr, seq[-1], new_conv)
    elif last_not_digit == len(seq) - 2:
        # one index last_attr[]
        new_conv = copy_conv2d.Conv2d_Custom(N, M, kernel_size=(K,K), stride=(S, S),
                             padding=(P,P), groups=G, bias=is_b, is_pattern=is_pattern, pattern=pattern, pattern_ones = pattern_ones)
        last_attr[int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-2], last_attr)
    elif last_not_digit == len(seq) - 3:
        # two index last_attr[][]
        new_conv = copy_conv2d.Conv2d_Custom(N, M, kernel_size=(K,K), stride=(S, S),
                             padding=(P,P), groups=G, bias=is_b, is_pattern=is_pattern, pattern=pattern, pattern_ones = pattern_ones)
        last_attr[int(seq[-2])][int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-3], last_attr)
    else:
        print("more than 2 depth of index from last layer is not support!")
        sys.exit(0)

    ## Weiwen: 03-29
    ## Step 4: Setup new parameters from backup
    ##
    if is_b:
        model.state_dict()[layer_name + ".bias"][:] = ori_para_b
    model.state_dict()[layer_name + ".weight"][:] = ori_para_w




def ztNAS_add_kernel_quant(model,layer, layer_name, is_quant, quan_paras,is_std_conv=False):

    if isinstance(layer,copy_conv2d.Conv2d_Custom):
        layer.is_quant = is_quant
        layer.quan_paras =  quan_paras
        return


    [M, N, K, S, G, P, b] = (
        layer.out_channels, layer.in_channels, is_same(layer.kernel_size),
        is_same(layer.stride), layer.groups, is_same(layer.padding), layer.bias)

    ## Weiwen: 03-29
    ## Step 1: Translate layer name to locate layer module
    ##
    seq = layer_name.split(".")
    (pre_attr, last_attr, last_not_digit) = get_last_attr_idx(model, seq)


    ## Weiwen: 03-29
    ## Step 2: Backup weights and bias if exist
    ##
    is_b = False
    if type(b)==nn.Parameter:
        ori_para_b = model.state_dict()[layer_name + ".bias"][:]
        is_b = True
    ori_para_w = model.state_dict()[layer_name + ".weight"][:]

    ## Weiwen: 03-29
    ## Step 3: Translate layer name to locate layer module
    ##
    if last_not_digit == len(seq) - 1:
        # last one is the attribute, directly setattr
        new_conv = copy_conv2d.Conv2d_Custom(N, M, kernel_size=(K,K), stride=(S, S),
                             padding=(P,P), groups=G, bias=is_b, is_quant=is_quant, quan_paras=quan_paras,
                                             is_std_conv=is_std_conv)
        setattr(pre_attr, seq[-1], new_conv)
    elif last_not_digit == len(seq) - 2:
        # one index last_attr[]
        new_conv = copy_conv2d.Conv2d_Custom(N, M, kernel_size=(K,K), stride=(S, S),
                             padding=(P,P), groups=G, bias=is_b, is_quant=is_quant, quan_paras=quan_paras,
                                             is_std_conv=is_std_conv)
        last_attr[int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-2], last_attr)
    elif last_not_digit == len(seq) - 3:
        # two index last_attr[][]
        new_conv = copy_conv2d.Conv2d_Custom(N, M, kernel_size=(K,K), stride=(S, S),
                             padding=(P,P), groups=G, bias=is_b, is_quant=is_quant, quan_paras=quan_paras,
                                             is_std_conv=is_std_conv)
        last_attr[int(seq[-2])][int(seq[-1])] = new_conv
        setattr(pre_attr, seq[-3], last_attr)
    else:
        print("more than 2 depth of index from last layer is not support!")
        sys.exit(0)

    ## Weiwen: 03-29
    ## Step 4: Setup new parameters from backup
    ##
    if is_b:
        model.state_dict()[layer_name + ".bias"][:] = ori_para_b
    model.state_dict()[layer_name + ".weight"][:] = ori_para_w




def ztNAS_cut_channel(model,conv_modify,bn_modifiy):

    INDEX = {}

    for k,v in conv_modify.items():
        force = False
        layer_name = k
        layer = v[0]
        [M, N, K, S, G, P, b] = (
            int(v[2]), int(v[1]), is_same(layer.kernel_size),
            is_same(layer.stride), layer.groups, is_same(layer.padding), layer.bias)
        if len(v)==5:
            # print(layer_name,v)
            force = v[4]
        ori_N = layer.in_channels

        ## Weiwen: 03-29
        ## Step 1: Translate layer name to locate layer module
        ##
        seq = layer_name.split(".")
        (pre_attr, last_attr, last_not_digit) = get_last_attr_idx(model, seq)

        ## Weiwen: 03-29
        ## Step 2: Backup weights and bias if exist
        ##
        # print(layer_name)
        if M!= layer.out_channels and N != layer.in_channels:
            print(layer_name)
            print([M, N, K, S, G, P])
            print(layer)
            print("Not support cut channels for both IFM and OFM at once, do it sequentially")
            sys.exit(0)

        if M != layer.out_channels or force:
            # OFM Filtering
            print(layer_name,force,M, layer.out_channels)
            W = model.state_dict()[layer_name + ".weight"][:]
            Idx = W.norm(dim=(2, 3)).sum(dim=1).topk(M)[1]

            is_b = False
            if type(b) == nn.Parameter:
                ori_para_b = model.state_dict()[layer_name + ".bias"][Idx]
                is_b = True
            ori_para_w = model.state_dict()[layer_name + ".weight"][Idx]

            if len(v[3])!=0:
                for layer_n in v[3]:
                    INDEX[layer_n] = Idx


        elif N != layer.in_channels:
            if layer_name not in INDEX.keys():
                print("[ERROR]: IFM index must be obtained from previous OFM")
                sys.exit(0)

            # IFM Filtering
            Idx = INDEX[layer_name]

            # W = model.state_dict()[layer_name + ".weight"][:]
            # Idx = W.norm(dim=(2, 3)).sum(dim=0).topk(N)[1]

            is_b = False
            if type(b) == nn.Parameter:
                ori_para_b = model.state_dict()[layer_name + ".bias"][:]
                is_b = True

            if G==ori_N:# dconv
                ori_para_w = model.state_dict()[layer_name + ".weight"][Idx]
                G = N
                M = N
            # print(layer_name + ".weight", model.state_dict()[layer_name + ".weight"].shape,Idx)
            else:
                ori_para_w = model.state_dict()[layer_name + ".weight"].transpose(0, 1)[Idx].transpose(0, 1)

        else:
            is_b = False
            if type(b) == nn.Parameter:
                ori_para_b = model.state_dict()[layer_name + ".bias"][:]
                is_b = True
            ori_para_w = model.state_dict()[layer_name + ".weight"][:]

        ## Weiwen: 03-29
        ## Step 3: Translate layer name to locate layer module
        ##
        if last_not_digit == len(seq) - 1:
            # last one is the attribute, directly setattr
            new_conv = nn.Conv2d(N, M, kernel_size=(K,K), stride=(S, S),
                                 padding=(P,P), groups=G, bias=is_b)
            setattr(pre_attr, seq[-1], new_conv)
        elif last_not_digit == len(seq) - 2:
            # one index last_attr[]
            new_conv = nn.Conv2d(N, M, kernel_size=(K,K), stride=(S, S),
                                 padding=(P,P), groups=G, bias=is_b)
            last_attr[int(seq[-1])] = new_conv
            setattr(pre_attr, seq[-2], last_attr)
        elif last_not_digit == len(seq) - 3:
            # two index last_attr[][]
            new_conv = nn.Conv2d(N, M, kernel_size=(K,K), stride=(S, S),
                                 padding=(P,P), groups=G, bias=is_b)
            last_attr[int(seq[-2])][int(seq[-1])] = new_conv
            setattr(pre_attr, seq[-3], last_attr)
        else:
            print("more than 2 depth of index from last layer is not support!")
            sys.exit(0)

        ## Weiwen: 03-29
        ## Step 4: Setup new parameters from backup
        ##
        # print(N,M,ori_para_w.shape)
        # print(model.state_dict()[layer_name + ".weight"].shape)
        if type(b)==nn.Parameter:
            model.state_dict()[layer_name + ".bias"][:] = ori_para_b
        model.state_dict()[layer_name + ".weight"][:] = ori_para_w



    for k,v in bn_modifiy.items():
        layer_name = k
        layer = v[0]
        [eps, momentum, affine, track_running_stats] = (layer.eps, layer.momentum, layer.affine, layer.track_running_stats)
        ch = v[1]

        ## Weiwen: 03-29
        ## Step 1: Translate layer name to locate layer module
        ##
        seq = layer_name.split(".")
        (pre_attr, last_attr, last_not_digit) = get_last_attr_idx(model, seq)

        ## Weiwen: 03-29
        ## Step 2: Backup weights and bias if exist
        ##
        if layer_name not in INDEX.keys():
            print("[WARNING]: Batchnorm index must be obtained from previous conv, THIS indicates that the batchnorm will not be changed")
            return
        if affine:
            ori_para_b = model.state_dict()[layer_name + ".bias"][INDEX[layer_name]]
            ori_para_w = model.state_dict()[layer_name + ".weight"][INDEX[layer_name]]
        if track_running_stats:
            ori_para_E = model.state_dict()[layer_name + ".running_mean"][INDEX[layer_name]]
            ori_para_V = model.state_dict()[layer_name + ".running_var"][INDEX[layer_name]]

        ## Weiwen: 03-29
        ## Step 3: Translate layer name to locate layer module
        ##
        if last_not_digit == len(seq) - 1:
            # last one is the attribute, directly setattr
            new_conv = nn.BatchNorm2d(ch, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
            setattr(pre_attr, seq[-1], new_conv)
        elif last_not_digit == len(seq) - 2:
            # one index last_attr[]
            new_conv = nn.BatchNorm2d(ch, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
            last_attr[int(seq[-1])] = new_conv
            setattr(pre_attr, seq[-2], last_attr)
        elif last_not_digit == len(seq) - 3:
            # two index last_attr[][]
            new_conv = nn.BatchNorm2d(ch, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
            last_attr[int(seq[-2])][int(seq[-1])] = new_conv
            setattr(pre_attr, seq[-3], last_attr)
        else:
            print("more than 2 depth of index from last layer is not support!")
            sys.exit(0)

        ## Weiwen: 03-29
        ## Step 4: Setup new parameters from backup
        ##

        if affine:
            model.state_dict()[layer_name + ".bias"][:] = ori_para_b
            model.state_dict()[layer_name + ".weight"][:] = ori_para_w
        if track_running_stats:
            model.state_dict()[layer_name + ".running_mean"][:] = ori_para_E
            model.state_dict()[layer_name + ".running_var"][:] = ori_para_V
