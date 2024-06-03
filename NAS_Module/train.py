from __future__ import print_function
import datetime
import os
import time
import sys

import copy
import bit_hyperrule
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms

sys.path.append("../Interface")
sys.path.append("..")
import cifar10_models
from ztNAS_model_change import *
from model_modify import *
import utils
import bottleneck_conv_only
import bottlenect_conv_dconv
from search_space import *
from rl_input import *

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# from model_search_space.ss_mnasnet0_5 import mnasnet0_5_space
# from model_search_space.ss_resnet18 import resnet_18_space
# from model_search_space.ss_mobilenet_v2 import mobilenet_v2_space

from model_search_space import ss_mnasnet1_0, ss_mnasnet0_5, ss_resnet18, ss_mobilenet_v2, ss_proxyless_mobile
from model_search_space import ss_resnet18_cifar, ss_big_transfer,ss_mobilenet_cifar,ss_densenet121_cifar
from model_search_space import ss_xception

try:
    from apex import amp
except ImportError:
    amp = None

best_acc5 = 0

def save_chk_point(model_without_ddp,optimizer,lr_scheduler,epoch,acc5):
    global args
    global best_acc5
    if acc5>best_acc5:
        best_acc5 = acc5
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args}
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'ckp_{}.pth'.format(acc5)))

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, model_without_ddp, lr_scheduler, apex=False,
                    data_loader_test=0,isreinfoce=False, stop_batch=6000):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    batch_idx = 0

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

        batch_idx += 1
        if batch_idx == stop_batch:
            # evaluate(model, criterion, data_loader_test, device=device)
            # acc1, acc5 = evaluate(model, criterion, data_loader_test, device=device)
            # save_chk_point(model_without_ddp, optimizer, lr_scheduler, epoch, acc5)
            if isreinfoce:
                return

        if batch_idx % 500 == 0:
            acc1,acc5 = evaluate(model, criterion, data_loader_test, device=device)
            save_chk_point(model_without_ddp, optimizer, lr_scheduler, epoch, acc5)


def evaluate(model, criterion, data_loader, device, print_freq=10, isreinfoce=False, stop_batch=200):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    batch_idx = 0

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            batch_idx += 1
            if batch_idx == stop_batch:
                # evaluate(model, criterion, data_loader_test, device=device)
                if isreinfoce:
                    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg



def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def main(args, dna, ori_HW, data_loader, data_loader_test, ori_HW_dconv=[]):

    # print("==============Train==========")
    # print(pat_point, exp_point, ch_point)

    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    device = torch.device(args.device)

    print("Creating model")

    if args.dataset == "imagenet":
        if "proxyless" in args.model:
            model = torch.hub.load('mit-han-lab/ProxylessNAS', args.model)
        elif "FBNET" in args.model:
            model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'fbnetc_100')
        else:
            model =torchvision.models.__dict__[args.model](pretrained=args.pretrained)
    elif args.dataset == "cifar10":
        model = getattr(cifar10_models, args.model)(pretrained=True)

    # model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)

    if args.dataset == "imagenet":
        if args.model == "resnet18":
            pat_point, exp_point, ch_point, quant_point, comm_point = dna[0:4], dna[4], dna[5:10], dna[10:18], dna[18:21]
            HW = copy.deepcopy(ori_HW)
            HW[5] += comm_point[0]
            HW[6] += comm_point[1]
            HW[7] += comm_point[2]
            model = ss_resnet18.resnet_18_space(model, pat_point, exp_point, ch_point, quant_point, args)
        elif args.model == "mnasnet0_5":
            # pattern_3_3_idx = dna[0:4]
            # pattern_5_5_idx = dna[4:8]
            # pattern_do_or_not = dna
            # q_list = dna[8:23]
            model = ss_mnasnet0_5.mnasnet0_5_space(model, dna, args)
        elif args.model == "mnasnet1_0":
            HW_cconv = copy.deepcopy(ori_HW)
            HW_dconv = copy.deepcopy(ori_HW_dconv)
            model,ori_HW, ori_HW_dconv = ss_mnasnet1_0.mnasnet1_0_space(model,dna, HW_cconv, HW_dconv ,args)
        elif args.model == "mobilenet_v2":
            HW_cconv = copy.deepcopy(ori_HW)
            HW_dconv = copy.deepcopy(ori_HW_dconv)
            model, ori_HW, ori_HW_dconv = ss_mobilenet_v2.mobilenet_v2_space(model, dna, HW_cconv, HW_dconv,
                                                                                args)
            # model = ss_mobilenet_v2.mobilenet_v2_space(model, args)
        elif args.model == "proxyless_mobile":
            HW_cconv = copy.deepcopy(ori_HW)
            HW_dconv = copy.deepcopy(ori_HW_dconv)
            model,ori_HW, ori_HW_dconv = ss_proxyless_mobile.proxyless_mobile_space(model, dna, HW_cconv, HW_dconv ,args)
        elif args.model == "xception":
            HW_cconv = copy.deepcopy(ori_HW)
            model, ori_HW = ss_xception.xception_space(model, dna, HW_cconv, HW_dconv, args)
        else:
            print("Currently not support the given model {}".format("args.model"))
            sys.exit(0)

    elif args.dataset == "cifar10":
        if args.model == "resnet18":
            HW_cconv = copy.deepcopy(ori_HW)
            model,ori_HW = ss_resnet18_cifar.resnet_18_space(model, dna, HW_cconv, args)
        elif args.model == "big_transfer":
            HW_cconv = copy.deepcopy(ori_HW)
            model, ori_HW = ss_big_transfer.big_transfer_space(model, dna, HW_cconv, args)
        elif args.model == "mobilenet_v2":
            HW_cconv = copy.deepcopy(ori_HW)
            HW_dconv = copy.deepcopy(ori_HW_dconv)
            model, ori_HW, ori_HW_dconv = ss_mobilenet_cifar.mobilenet_v2_space(model, dna, HW_cconv, HW_dconv,
                                                                                     args)
        elif args.model == "densenet121":
            HW_cconv = copy.deepcopy(ori_HW)
            model, ori_HW = ss_densenet121_cifar.densenet121_space(model, dna, HW_cconv, args)
        elif args.model == "xception":
            HW_cconv = copy.deepcopy(ori_HW)
            model, ori_HW = ss_xception.xception_space(model, dna, HW_cconv, HW_dconv, args)
        else:
            print("Currently not support the given model {}".format("args.model"))
            sys.exit(0)


    # print(model)

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    total_lat = 0

    if args.hw_test:
        print("HW_Test")
        if args.model == "resnet18" and args.dataset=="imagenet":
            if HW[5] + HW[6] + HW[7] <= int(HW_constraints["r_Ports_BW"] / HW_constraints["BITWIDTH"]):
                total_lat = bottleneck_conv_only.get_performance(model, HW[0], HW[1], HW[2], HW[3],
                                                                 HW[4], HW[5], HW[6], HW[7], device)
            else:
                print("HW Port exceed",HW[5] + HW[6] + HW[7], int(HW_constraints["r_Ports_BW"] / HW_constraints["BITWIDTH"]))
                return 0, 0, -1
        elif args.model == "mnasnet0_5" or args.model == "mnasnet1_0" or args.model == "proxyless_mobile" or args.dataset=="cifar10":
            # print(ori_HW_dconv, ori_HW)
            total_lat = bottlenect_conv_dconv.get_performance(model, args.dataset, ori_HW_dconv, ori_HW, device)





        print("HW_Test Done")
        if total_lat>float(args.target_lat.split(" ")[1]):
            print("Latency Cannot satisfy", total_lat, float(args.target_lat.split(" ")[1]))
            return 0, 0, total_lat
        elif total_lat==-1 or total_lat==0:
            print("No latency got", total_lat, float(args.target_lat.split(" ")[1]))
            return 0, 0, -1

        print("Hardware Test Pass {}/{}".format(total_lat,float(args.target_lat.split(" ")[1])))

    else:
        if args.model == "resnet18" and args.dataset=="imagenet":
            if HW[5] + HW[6] + HW[7] <= int(HW_constraints["r_Ports_BW"] / HW_constraints["BITWIDTH"]):
                total_lat = bottleneck_conv_only.get_performance(model, HW[0], HW[1], HW[2], HW[3],
                                                                 HW[4], HW[5], HW[6], HW[7], device)
            else:
                print("HW Port exceed",HW[5] + HW[6] + HW[7], int(HW_constraints["r_Ports_BW"] / HW_constraints["BITWIDTH"]))
                total_lat = 99999999999
        elif args.model == "mnasnet0_5" or args.model == "mnasnet1_0" or args.model == "proxyless_mobile" or args.dataset=="cifar10":
            # print(ori_HW_dconv, ori_HW)
            total_lat = bottlenect_conv_dconv.get_performance(model, args.dataset, ori_HW_dconv, ori_HW, device)
        print(total_lat)
    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return 0,0,0



    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, model_without_ddp,
                        lr_scheduler, args.apex, data_loader_test, args.reinfoce, stop_batch=args.train_stop_batch)
        lr_scheduler.step()
        acc1, acc5 = evaluate(model, criterion, data_loader_test, device=device,
                              isreinfoce=args.reinfoce, stop_batch=args.test_stop_batch)

        if args.reinfoce:
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            return acc1, acc5, total_lat

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return 0,0,0


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/mnt/weiwen/ImageNet', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    # parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--cache-dataset",dest="cache_dataset",help="Cache the datasets for quicker initialization. It also serializes the transforms",
                        action="store_true",)
    parser.add_argument("--sync-bn",dest="sync_bn",help="Use sync batch norm",action="store_true",)

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # NAS related options
    parser.add_argument('--model', default='mobilenet_v2', help='model')
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        action="store_true", )
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true", )
    parser.add_argument("--rl", dest="reinfoce", help="execute reinforcement leraning", action="store_true", )
    parser.add_argument('--train_stop_batch', default=100, type=int, metavar='N',help='number of batch to terminate in training')
    parser.add_argument('--test_stop_batch', default=100000, type=int, metavar='N',help='number of batch to terminate in testing')
    parser.add_argument('-f', '--finetue_dna', default="54 33 39 44 1 1 0 1 0 30 25 24 28 24 23 12 12 7 30 17 7 12 2 2 1", help="hardware desgin of cconv", )
    parser.add_argument('-a', '--alpha', default="0.7", help="rl controller reward parameter", )
    parser.add_argument('-acc', '--target_acc', default="80 89", help="target accuracy range, determining reward", )
    parser.add_argument('-lat', '--target_lat', default="7 10", help="target latency range, determining reward", )
    parser.add_argument('-rlopt', '--rl_optimizer', default="Adam", help="optimizer of rl", )
    parser.add_argument("--hwt", dest="hw_test", help="whether test hardware", action="store_true", )
    parser.add_argument('-dc', '--dconv',default="832, 1, 32, 32, 5, 6, 10, 16",help="hardware desgin of dconv", )
    parser.add_argument('-c', '--cconv',default="130, 19, 32, 32, 3, 18, 2, 10",help="hardware desgin of cconv",)
    parser.add_argument('-d', '--dataset',default='imagenet')
    args = parser.parse_args()

    print("=" * 58)
    print("="*10,"Welcome to use automatic reverse NAS","="*10)
    print("="*11,"Your setting is listed as follows","="*12)
    print ("\t{:<20} {:<15}".format('Attribute', 'Input'))
    for k,v in vars(args).items():
        print("\t{:<20} {:<15}".format(k, v))
    print("="*12,"Exploration will start, have fun","=" * 12)
    print("=" * 58)

    print("-" * 58)
    print("-" * 10, "Search Space of Reinforcement Learning", "-" * 10)
    print("\t{:<20} {:<15}".format('Attribute', 'Search space'))


    datasets_name = args.dataset

    if datasets_name == "imagenet":
        if args.model == "resnet18":
            model_pointer = ss_resnet18
        elif args.model == "mnasnet0_5":
            model_pointer = ss_mnasnet0_5
        elif args.model == "mnasnet1_0":
            model_pointer = ss_mnasnet1_0
        elif args.model == "mobilenet_v2":
            model_pointer = ss_mobilenet_v2
        elif args.model == "proxyless_mobile":
            model_pointer = ss_proxyless_mobile
        elif args.model == "xception":
            model_pointer = ss_xception
    elif datasets_name == "cifar10":
        if args.model == "resnet18":
            model_pointer = ss_resnet18_cifar
        elif args.model == "big_transfer":
            model_pointer = ss_big_transfer
        elif args.model == "mobilenet_v2":
            model_pointer = ss_mobilenet_cifar
        elif args.model == "densenet121":
            model_pointer = ss_densenet121_cifar
        elif args.model == "xception":
            model_pointer = ss_xception

    space_name = model_pointer.get_space()[0]
    space = model_pointer.get_space()[1]
    print(space)

    for idx in range(len(space)):
        if len(space[idx])>=2:
            sp = str(min(space[idx]))+" --"+str(space[idx][1]-space[idx][0])+"--> "+str(max(space[idx]))
        elif len(space[idx])==1:
            sp = space[idx][0]
        else:
            print("[Error]: Space has no element")
            sys.exit(0)
        print("\t{:<20} {:<15}".format(space_name[idx], sp))
    print("-" * 58)

    return args

def get_data_loader(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    torch.backends.cudnn.benchmark = True

    if args.dataset=="imagenet":

        train_dir = os.path.join(args.data_path, 'train')
        val_dir = os.path.join(args.data_path, 'val')
        dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                             args.cache_dataset, args.distributed)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.workers, pin_memory=True)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size,
            sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    elif args.dataset=="cifar10":
        if args.model!="big_transfer":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]

            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
            dataset = CIFAR10(root=args.data_path, train=True, transform=transform_train)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                    drop_last=True, pin_memory=True)

            transform_val = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            dataset = CIFAR10(root=args.data_path, train=False, transform=transform_val)
            data_loader_test = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
        else:
            precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
            train_tx = transforms.Compose([
                transforms.Resize((precrop, precrop)),
                transforms.RandomCrop((crop, crop)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            val_tx = transforms.Compose([
                transforms.Resize((crop, crop)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = CIFAR10(root=args.data_path, train=True, transform=train_tx)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                     drop_last=True, pin_memory=True)
            dataset = CIFAR10(root=args.data_path, train=False, transform=val_tx)
            data_loader_test = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    return data_loader,data_loader_test

if __name__ == "__main__":
    global args
    args = parse_args()
    data_loader,data_loader_test = get_data_loader(args)
    dna = [int(x.strip()) for x in args.finetue_dna.split(" ")]
    [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p] = [int(x.strip()) for x in args.cconv.split(",")]
    HW = [Tm, Tn, Tr, Tc, Tk, W_p, I_p, O_p]
    HW2 = [int(x.strip()) for x in args.dconv.split(",")]
    print(HW,HW2)

    # print(dna,HW,HW2)
    #
    # model = torchvision.models.__dict__["mnasnet0_5"](pretrained=args.pretrained)
    # pattern_3_3_idx = dna[0:4]
    # pattern_5_5_idx = dna[4:8]
    # q_list = dna[8:23]
    # model = mnasnet0_5_space(model, pattern_3_3_idx, pattern_5_5_idx, q_list, args)

    acc1, acc5, lat = main(args, dna, HW, data_loader, data_loader_test, HW2)
    # main(args, [int(x) for x in dna.split(" ")], HW)
    #
    #
    # print(model)
    #
    # total_lat = bottlenect_conv_dconv.get_performance(model, HW2, HW)
    #
    # print("HW_Test Done")
