#!/bin/bash

epochs=60
device=$@

# NGD
echo 'CIFAR10 MN NGD Started'
for i in 1 
do
	python main.py --freq 100 --trial true --super_opt true --batchnorm false --dataset cifar10  --batch_size 128  --device $device --optimizer ngd --network mobilenetv2  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  0.01 --learning_rate_decay 0.5  --damping 0.2  --weight_decay 0.01 --momentum 0.9
done
echo 'CIFAR10 MN NGD Finished'

# KFAC
echo 'CIFAR10 MN KBFGS Started'
for i in 1 
do
	python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer kbfgs --network mobilenetv2  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.03  --weight_decay 0.03 --momentum 0.9
done
echo 'CIFAR10 MN KBFGS Finsihed'


echo 'CIFAR100 WRN KBFGS-L Started'
for i in 1
do
	python main.py --dataset cifar100  --batch_size 128 --device $device --optimizer kbfgsl  --network wrn  --depth 28 --widen_factor 4  --epoch 60 --milestone 30,45 --learning_rate  0.003 --learning_rate_decay 0.1  --damping 0.01  --weight_decay 0.01 --momentum 0.9
done
echo 'CIFAR100 WRN KBFGS-L Finsihed'


echo 'CIFAR100 VGG KBFGS Started'
for i in 1
do
	python main.py --dataset cifar100  --batch_size 128 --device $device --optimizer kbfgs  --network vgg16_bn  --epoch 60 --milestone 30,45 --learning_rate  0.01 --learning_rate_decay 0.1  --damping 0.03  --weight_decay 0.03 --momentum 0.9
done
echo 'CIFAR100 VGG KBFGS Finsihed'

# KBFGS
echo 'CIFAR100 MN KBFGS Started'
for i in 1
do
	python main.py --dataset cifar100  --batch_size 128 --device $device --optimizer kbfgs  --network mobilenetv2  --epoch 60 --milestone 30,45 --learning_rate  0.03 --learning_rate_decay 0.1  --damping 0.03  --weight_decay 0.01 --momentum 0.9
done
echo 'CIFAR100 MN KBFGS Finsihed'
