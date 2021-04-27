#!/bin/bash

epochs=60
device=$@
# SGD
echo 'SGD Started'
for i in 1 2 3 4 
do
	python main.py --dataset cifar100  --batch_size 128  --device $device --optimizer sgd --network vgg16_bn  --epoch $epochs --milestone 30,45 --learning_rate  1e-2 --learning_rate_decay 0.1  --weight_decay 0.003 --momentum 0.9
done
echo 'SGD finished'

# NGD
echo 'NGD Started'
for i in 1 2 3 4
do
	python main.py --freq 100 --trial true --batchnorm false --dataset cifar100  --batch_size 128 --low_rank true --gamma 0.9 --device $device --optimizer ngd --network vgg16_bn  --epoch $epochs --milestone 30,45 --learning_rate  1e-2 --learning_rate_decay 0.1  --damping 0.3  --weight_decay 0.03 --momentum 0.9
done
echo 'NGD Finished'

# KFAC
echo 'KFAC Started'
for i in 1 2 3 4
do
	python main.py --dataset cifar100  --batch_size 128 --device $device --optimizer kfac --network vgg16_bn  --epoch $epochs --milestone 30,45 --learning_rate  0.003 --learning_rate_decay 0.1  --damping 0.03  --weight_decay 0.01 --momentum 0.9
done
echo 'KFAC Finsihed'

# KFAC
echo 'EKFAC Started'
for i in 1 2 3 4
do
	python main.py --dataset cifar100  --batch_size 128 --device $device --optimizer ekfac  --network vgg16_bn  --epoch $epochs --milestone 30,45 --learning_rate  0.003 --learning_rate_decay 0.1  --damping 0.03  --weight_decay 0.01 --momentum 0.9
done
echo 'EKFAC Finsihed'

# KFAC
echo 'KBFGS-L Started'
for i in 1 2 3 4
do
	python main.py --dataset cifar100  --batch_size 128 --device $device --optimizer kbfgsl  --network vgg16_bn  --epoch 25 --milestone 30,45 --learning_rate  0.003 --learning_rate_decay 0.1  --damping 0.01  --weight_decay 0.01 --momentum 0.9
done
echo 'KBFGS-L Finsihed'
