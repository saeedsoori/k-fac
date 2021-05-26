#!/bin/bash
epochs=60
device=$@

for optim in ngd exact_ngd kfac
do
	echo $optim
	python main.py --freq 100 --network toy --dataset fashion-mnist --learning_rate 0.03 --step_info true --damping 0.3 --momentum 0.9 --device $device --optimizer $optim --epoch $epochs --batch_size 128 --learning_rate_decay 0.5 --milestone 10,20,30,40,50 --save_inv true
done