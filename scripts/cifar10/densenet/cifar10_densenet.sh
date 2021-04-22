#!/bin/bash

epochs=60
device=$@
# SGD
echo 'SGD Started'
for i in 1 2 3 4 5
do
	python main.py --dataset cifar10  --batch_size 128  --device $device --optimizer sgd --network densenet  --depth 19 --growthRate 100  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  3e-2 --learning_rate_decay 0.5  --weight_decay 0.003 --momentum 0.9
done
echo 'SGD finished'

# NGD
echo 'NGD Started'
for i in 1 2 3 4 5
do
	python main.py --freq 100 --trial true --batchnorm false --dataset cifar10  --batch_size 128  --device $device --optimizer ngd --network densenet  --depth 19 --growthRate 100  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-1 --learning_rate_decay 0.5  --damping 0.2  --weight_decay 0.003 --momentum 0.9
done
echo 'NGD Finished'

# KFAC
echo 'KFAC Started'
for i in 1 2 3 4 5
do
	python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer kfac --network densenet  --depth 19 --growthRate 100  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.03  --weight_decay 0.003 --momentum 0.9
done
echo 'KFAC Finsihed'

# KFAC
echo 'EKFAC Started'
for i in 1 2 3 4 5
do
	python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer ekfac  --network densenet  --depth 19 --growthRate 100  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.01  --weight_decay 0.003 --momentum 0.9
done
echo 'EKFAC Finsihed'

# KFAC
echo 'KBFGS-L Started'
for i in 1 2 3 4 5
do
	python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer kbfgsl  --network densenet  --depth 19 --growthRate 100  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.03  --weight_decay 0.003 --momentum 0.9
done
echo 'KBFGS-L Finsihed'
