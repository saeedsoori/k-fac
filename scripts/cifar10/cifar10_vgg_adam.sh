#!/bin/bash
epochs=60
device=$@
for lr in 1e-4 1e-3 1e-2 1e-1
do
	for epsilon in 1e-8 1e-6 1e-4
	do
		python main.py --dataset cifar10  --batch_size 128 --device $device --optimizer adam --epsilon $epsilon --network vgg16_bn  --epoch $epochs --milestone 10,20,30,40,50 --learning_rate  $lr --learning_rate_decay 0.5  --weight_decay 0.003
	done
done
