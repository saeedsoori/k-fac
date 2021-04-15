#!/bin/bash
epochs=10
device=$@
for lr in 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1 3 10
do
	for damping in 3e-3 1e-2 3e-2 0.1 0.3 1 3
	do
   		python main.py --network fc --dataset mnist --learning_rate $lr --step_info true --momentum 0.5 --freq 10 --damping $damping --device $device --optimizer ngd  --epoch $epochs   --weight_decay 0.003 --batch_size 128
	done
done
