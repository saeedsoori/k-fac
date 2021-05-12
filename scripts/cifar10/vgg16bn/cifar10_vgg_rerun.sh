#!/bin/bash

epochs=60
device=$@

# NGD
echo 'NGD Started'
for i in 1 2 3 4 5
do
	python main.py --freq 100 --trial true --batchnorm false --dataset cifar10  --batch_size 128 --low_rank true --super_opt true --gamma 0.9 --device $device --optimizer ngd --network vgg16_bn  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  0.03 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
done
echo 'NGD Finished'

# KFAC
echo 'KBFGS-L Started'
for i in 1 2 3 4 5
do
	python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer kbfgsl  --network vgg16_bn  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  0.01 --learning_rate_decay 0.5  --damping 0.03  --weight_decay 0.003 --momentum 0.9
done
echo 'KBFGS-L Finsihed'
