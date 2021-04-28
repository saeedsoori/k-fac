#!/bin/bash
epochs=10
device=$@

for lr in 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1
do
	for damping in 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1
	do
		for wd in 1e-3 3e-3 1e-2 3e-2
		do
   			python main.py --network convnet --dataset fashion-mnist --learning_rate $lr --step_info true --momentum 0.9 --damping $damping --device $device --optimizer kfac --epoch $epochs --milestone 10,20,30,40,50 --weight_decay $wd --batch_size 128
		done
	done
done
