for model in 'resnet18';
do
    echo "******************************"
    echo $model
    python train.py --model $model  --pretrained --dataset "cifar10" --data-path "/workspace/hotNAS-CODES20/dataset" --device='cuda' -j 32 -b 256
    echo $model
    echo "=============================="
done

