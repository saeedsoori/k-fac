#!/bin/bash

epochs=60
device=$@



# KFAC
echo 'KFAC Started'
for i in 1
do
	python main.py --dataset cifar10 --TInv 100 --TCov 100 --batch_size 128 --device $device --optimizer kfac --network densenet  --depth 19 --growthRate 100  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.03  --weight_decay 0.003 --momentum 0.9
done
echo 'KFAC Finsihed'

# KFAC
echo 'EKFAC Started'
for i in 1
do
	python main.py --dataset cifar10  --TInv 100 --TCov 100 --batch_size 128 --device $device --optimizer ekfac  --network densenet  --depth 19 --growthRate 100  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.01  --weight_decay 0.003 --momentum 0.9
done
echo 'EKFAC Finsihed'

# KFAC
echo 'KBFGS Started'
for i in 1
do
	python main.py --dataset cifar10  --TInv 100 --TCov 100 --batch_size 128 --device $device --optimizer kbfgs  --network densenet  --depth 19 --growthRate 100  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  1e-2 --learning_rate_decay 0.5  --damping 0.01  --weight_decay 0.003 --momentum 0.9
done
echo 'KBFGS Finsihed'
