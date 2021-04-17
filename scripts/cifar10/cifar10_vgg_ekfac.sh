#!/bin/bash
epochs=10
device=$@
for lr in 1e-3 1e-2 1e-1
do
	for damping in 0.01 0.03 0.1 0.3
	do
		python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer ekfac --TScal 20 --network vgg16_bn  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  $lr --learning_rate_decay 0.5  --damping $damping  --weight_decay 0.003 --momentum 0.9
	done
done
