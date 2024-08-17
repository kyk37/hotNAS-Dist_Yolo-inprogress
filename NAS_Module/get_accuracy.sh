
# for model in 'vgg19_bn' 'wide_resnet101_2';
#for model in 'alexnet' 'densenet121' 'densenet161'  'densenet169' 'densenet201' 'squeezenet1_0' 'squeezenet1_1'  'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19' 'vgg19_bn''wide_resnet101_2' 'wide_resnet50_2'  'vgg11'  'resnet50' 'resnet101' 'resnet152' 'resnet18' 'resnet34' 'mnasnet0_5' 'mobilenet_v2'  'mnasnet0_75' 'mnasnet1_0' 'mnasnet1_3' 'shufflenet_v2_x0_5' 'shufflenet_v2_x1_0' 'shufflenet_v2_x1_5' 'shufflenet_v2_x2_0';
for model in 'resnet18';
do
	echo "******************************"
	echo $model
	python train.py --model $model  --pretrained --dataset "cifar10" --data-path "/workspace/hotNAS-CODES20/dataset" --test-only --device='cuda' -j 32 -b 256
	echo $model
	echo "=============================="
done




#python train.py --model resnet18  --pretrained --test-only --device='cuda' -j 32 -b 256
