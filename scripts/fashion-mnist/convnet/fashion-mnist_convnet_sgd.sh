#!/bin/bash
epochs=10
device=$@

for lr in 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1
do
	for wd in 1e-3 3e-3 1e-2 3e-2
	do
		python main.py --step_info true  --dataset fashion-mnist  --batch_size 128 --device $device --optimizer sgd --network convnet --epoch $epochs --milestone 10,20,30,40,50 --learning_rate $lr --weight_decay $wd --momentum 0.9
	done
done