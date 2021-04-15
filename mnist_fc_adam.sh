#!/bin/bash
epochs=10
device=$@
for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1 3 10
do
	for epsilon in 1e-10 1e-8 1e-6 1e-4 1e-3 1e-2 1e-1
	do
   		python main.py --network fc --dataset mnist --learning_rate $lr --step_info true --epsilon $epsilon --device $device --optimizer adam  --epoch $epochs --weight_decay 0.003 --batch_size 128
	done
done
