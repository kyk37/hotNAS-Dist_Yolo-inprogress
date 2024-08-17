



#for model in 'big_transfer' 'densenet121' 'densenet169' 'resnet18' 'resnet34' 'resnet50';
#'mnasnet0_5' 'mobilenet_v2' 'mnasnet1_0' 'shufflenet_v2_x0_5' 'shufflenet_v2_x1_0' 'proxyless_cpu' 'proxyless_gpu' 'proxyless_mobile' 'FBNET';
for model in 'resnet18' 'resnet34' 'resnet50';
do
#	touch res/$model
	python explore_conv_only.py -m $model
#	python explore_conv_dconv.py -m $model > res/$model.res &
done


#for model in 'mobilenet_v2' ;
#do
#  python explore_conv_dconv.py -m $model > res/$model.res &
#done

wait
