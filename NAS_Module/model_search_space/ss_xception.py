import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../Interface")
sys.path.append("../../Performance_Model")
import cifar10_models
from model_modify import *
import model_modify
import random
import train
import time
import datetime
import copy


## Create this space
def xception_space(model, dna, hw_cconv, hw_dconv, args):
    pass


## TODO Make this
def get_space():
    global p3size
    # p3size = 3
    # pattern_33_space = pattern_sets_generate_3()
    # p3num - len(pattern_33_space.keys())
    # print(p3num)
    space_name = ""
    space = ()

return space_name, space

## TODO VERIFY THIS
def dna_analysis(dna, logger):
    global p3size

    #pat_point, pattern_do_or_not, quant_point, comm_point = dna[0:4], dna[4:10], dna[10:103], dna[103:105]

    pattern_33_space = pattern_sets_generate_3((3, 3), p3size)

    for p in pat_point:
        logger.info("--------->Pattern 3-3 {}: {}".format(p, pattern_33_space[p].flatten()))
    logger.info("--------->Weight Pruning or Not: {}".format(pattern_do_or_not))
    logger.info("--------->Qunatization: {}".format(quant_point))
    logger.info("--------->HW: {}".format(comm_point))




if __name__ == "__main__":
       # parser = argparse.ArgumentParser('Parser User Input Arguments')
    # parser.add_argument('--device', default='cpu', help='device')
    # args = parser.parse_args()

    args = train.parse_args()
    # data_loader,data_loader_test = train.get_data_loader(args)


    model_name = "xception"
    dataset_name = "cifar10"

    hw_dconv_str = ""
    hw_cconv_str = ""
    oriDHW = [int(x.strip()) for x in hw_dconv_str.split(",")]
    oriCHW = [int(x.strip()) for x in hw_cconv_str.split(",")]

    start_time = time.time()
    count = 60
    record = {}
    latency = []
    for i in range(count):

        model = getattr(cifar10_models, model_name)(pretrained=True)


        # model = globals()["resnet18"]()

        _, space = get_space()
        dna = []
        for selection in space:
            dna.append(random.choice(selection))
        print(dna)

        ### -----   modify this "dna"  ----- #####
        ## TODO
        pat_point, exp_point, ch_point, quant_point, comm_point = dna[0:4], dna[4], dna[5:10], dna[10:18], dna[18:21]
        model,DHW,CHW = xception_space(model, dna, DHW, CHW, args)

        model = model.to(args.device)

        print("=" * 10, model_name, "performance analysis:")
        total_lat = bottlenect_conv_dconv.get_performance(model, dataset_name, DHW, CHW, args.device)
        print(total_lat)
        latency.append(total_lat)

        # acc1,acc5,_ = train.main(args, dna, HW, data_loader, data_loader_test)
        # record[i] = (acc5,total_lat)
        # print("Random {}: acc-{}, lat-{}".format(i, acc5,total_lat))
        # print(dna)
        # print("=" * 100)

    print("="*100)

    print("Min latency:",min(latency),max(latency),sum(latency)/float(len(latency)))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # print("Exploration End, using time {}".format(total_time_str))
    # for k,v in record.items():
    #     print(k,v)
    # print(min(latency), max(latency), sum(latency) / len(latency))