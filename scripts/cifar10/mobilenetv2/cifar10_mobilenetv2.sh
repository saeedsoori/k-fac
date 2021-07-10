#!/bin/bash

epochs=60
device=$@
# SGD
echo 'SGD Started'
for i in 1 2 3 4 5
do
	python main.py --dataset cifar10  --batch_size 128  --device $device --optimizer sgd --network mobilenetv2  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --weight_decay 0.003 --momentum 0.9
done
echo 'SGD finished'

# NGD
echo 'NGD Started'
for i in 1 2 3 4 5
do
	python main.py --freq 100 --trial true --super_opt true --batchnorm false --dataset cifar10  --batch_size 128  --device $device --optimizer ngd --network mobilenetv2  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.2  --weight_decay 0.01 --momentum 0.9
done
echo 'NGD Finished'

# KNGD
echo 'KNGD Started'
for i in 1 2 3 4 5
do
	python main.py --freq 100 --trial true --super_opt true --batchnorm false --dataset cifar10  --batch_size 128  --device $device --optimizer kngd --network mobilenetv2  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.2  --weight_decay 0.01 --momentum 0.9
done
echo 'NGD Finished'

# KFAC
echo 'KFAC Started'
for i in 1 2 3 4 5
do
	python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer kfac --network mobilenetv2  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
done
echo 'KFAC Finsihed'

# EKFAC
echo 'EKFAC Started'
for i in 1 2 3 4 5
do
	python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer ekfac   --network mobilenetv2  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
done
echo 'EKFAC Finsihed'

# KBFGSL
echo 'KBFGS-L Started'
for i in 1 2 3 4 5
do
	python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer kbfgsl --network mobilenetv2  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.1  --weight_decay 0.003 --momentum 0.9
done
echo 'KBFGS-L Finsihed'
