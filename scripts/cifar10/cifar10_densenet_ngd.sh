#!/bin/bash
epochs=60
device=$@
# lr range: 1e-3 1e-2 1e-1 
# damping range: 0.01 0.03 0.1 0.3 and 0.05 0.1 0.2
for lr in  1e-1 1e-2
do
	for damping in  0.1 0.05 0.2
	do
		python main.py --freq 100 --batchnorm false --dataset cifar10  --batch_size 128 --low_rank true --gamma 0.9 --device $device --optimizer ngd --network densenet  --depth 19 --growthRate 100 --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  $lr --learning_rate_decay 0.5  --damping $damping  --weight_decay 0.003 --momentum 0.9
	done
done
